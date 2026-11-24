"""Operations store (SQLite, stdlib only): alerts, action outcomes, API keys and job runs.

Kept in its own file (database/ecx_ops.db) so the research data and the v2 action ledger are never touched.
Secrets rule: an API key is shown ONCE when created. Only its SHA-256 hash is stored, never the key itself.
"""
import hashlib
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path(os.environ.get("ECX_OPS_DB", ROOT / "database" / "ecx_ops.db"))
SEVERITIES = ["info", "warning", "critical"]
ALERT_STATES = ["open", "acknowledged", "resolved"]
KEY_SCOPES = ["read", "write"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
  alert_id TEXT PRIMARY KEY, dedupe_key TEXT NOT NULL, kind TEXT NOT NULL, metric TEXT NOT NULL,
  entity_type TEXT NOT NULL, entity_id TEXT NOT NULL, severity TEXT NOT NULL, state TEXT NOT NULL,
  title TEXT NOT NULL, detail TEXT, as_of TEXT, window_label TEXT,
  baseline_value REAL, current_value REAL, n_current INTEGER, n_baseline INTEGER, p_value REAL, effect REAL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, ack_by TEXT, ack_note TEXT, action_id TEXT);
CREATE INDEX IF NOT EXISTS idx_alert_dedupe ON alerts(dedupe_key, state);
CREATE TABLE IF NOT EXISTS alert_notes (
  note_id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id TEXT NOT NULL, author TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_note_alert ON alert_notes(alert_id);
CREATE TABLE IF NOT EXISTS outcomes (
  outcome_id TEXT PRIMARY KEY, action_id TEXT, alert_id TEXT, metric TEXT NOT NULL, entity_type TEXT NOT NULL, entity_id TEXT NOT NULL,
  action_date TEXT NOT NULL, window_days INTEGER NOT NULL, result_json TEXT NOT NULL, verdict TEXT NOT NULL,
  created_at TEXT NOT NULL, created_by TEXT);
CREATE TABLE IF NOT EXISTS api_keys (
  key_id TEXT PRIMARY KEY, name TEXT NOT NULL, key_hash TEXT NOT NULL UNIQUE, prefix TEXT NOT NULL, scope TEXT NOT NULL,
  created_at TEXT NOT NULL, created_by TEXT, last_used TEXT, revoked INTEGER NOT NULL DEFAULT 0, uses INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS job_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, summary TEXT);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def conn(db=None):
    p = Path(db or DEFAULT_DB)
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _next_id(c, table, col, prefix):
    n = c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] + 1  # nosec B608
    while c.execute(f"SELECT 1 FROM {table} WHERE {col}=?", (f"{prefix}-{n:05d}",)).fetchone():  # nosec B608
        n += 1
    return f"{prefix}-{n:05d}"


# ------------------------------------------------------------------ alerts
def upsert_alert(a, db=None):
    """Insert a new alert unless an open/acknowledged alert with the same dedupe_key exists (then refresh its numbers).
    Returns (alert_id, created: bool). A resolved alert can be raised again if the problem returns."""
    if a["severity"] not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")
    with conn(db) as c:
        ex = c.execute("SELECT alert_id FROM alerts WHERE dedupe_key=? AND state!='resolved'", (a["dedupe_key"],)).fetchone()
        t = now()
        cols = ("severity", "title", "detail", "as_of", "window_label", "baseline_value", "current_value", "n_current", "n_baseline", "p_value", "effect")
        if ex:
            c.execute("UPDATE alerts SET " + ",".join(f"{k}=?" for k in cols) + ", updated_at=? WHERE alert_id=?",  # nosec B608
                      [a.get(k) for k in cols] + [t, ex["alert_id"]])
            return ex["alert_id"], False
        aid = _next_id(c, "alerts", "alert_id", "ALT")
        c.execute("INSERT INTO alerts (alert_id,dedupe_key,kind,metric,entity_type,entity_id,severity,state,title,detail,as_of,window_label,"
                  "baseline_value,current_value,n_current,n_baseline,p_value,effect,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (aid, a["dedupe_key"], a["kind"], a["metric"], a["entity_type"], str(a["entity_id"]), a["severity"], "open", a["title"],
                   a.get("detail"), a.get("as_of"), a.get("window_label"), a.get("baseline_value"), a.get("current_value"),
                   a.get("n_current"), a.get("n_baseline"), a.get("p_value"), a.get("effect"), t, t))
        return aid, True


def get_alert(alert_id, db=None):
    with conn(db) as c:
        r = c.execute("SELECT * FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
        return dict(r) if r else None


_SEV_RANK = "CASE severity WHEN 'critical' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END"


def list_alerts(state=None, severity=None, limit=200, db=None):
    q, a = "SELECT * FROM alerts WHERE 1=1", []
    if state:
        q += " AND state=?"; a.append(state)
    if severity:
        q += " AND severity=?"; a.append(severity)
    with conn(db) as c:
        return [dict(r) for r in c.execute(q + f" ORDER BY {_SEV_RANK}, updated_at DESC, alert_id DESC LIMIT ?", a + [limit])]


def set_alert_state(alert_id, new_state, actor, note="", db=None):
    if new_state not in ALERT_STATES:
        raise ValueError(f"state must be one of {ALERT_STATES}")
    if not actor or not actor.strip():
        raise ValueError("a named person is required to change an alert")
    if new_state == "resolved" and not (note or "").strip():
        raise ValueError("a note is required when resolving an alert")
    with conn(db) as c:
        if not c.execute("SELECT 1 FROM alerts WHERE alert_id=?", (alert_id,)).fetchone():
            raise KeyError(f"alert {alert_id} not found")
        c.execute("UPDATE alerts SET state=?, updated_at=?, ack_by=?, ack_note=? WHERE alert_id=?", (new_state, now(), actor.strip(), note, alert_id))
    return get_alert(alert_id, db)


def link_action(alert_id, action_id, db=None):
    with conn(db) as c:
        c.execute("UPDATE alerts SET action_id=?, updated_at=? WHERE alert_id=?", (action_id, now(), alert_id))


def alert_counts(db=None):
    with conn(db) as c:
        out = {s: 0 for s in ALERT_STATES}
        for r in c.execute("SELECT state, COUNT(*) n FROM alerts GROUP BY state"):
            out[r["state"]] = r["n"]
        out["open_critical"] = c.execute("SELECT COUNT(*) FROM alerts WHERE state='open' AND severity='critical'").fetchone()[0]
        out["open_warning"] = c.execute("SELECT COUNT(*) FROM alerts WHERE state='open' AND severity='warning'").fetchone()[0]
        return out


# ------------------------------------------------------------------ alert notes
NOTE_MAX = 1000


def add_note(alert_id, author, body, db=None):
    """Append a comment to an alert. Notes are never edited or deleted, so they form an audit trail of who said what and when."""
    body = (body or "").strip()
    if not (author or "").strip():
        raise ValueError("a named person is required to add a note")
    if not body:
        raise ValueError("a note cannot be empty")
    if len(body) > NOTE_MAX:
        raise ValueError(f"a note can be at most {NOTE_MAX} characters")
    with conn(db) as c:
        if not c.execute("SELECT 1 FROM alerts WHERE alert_id=?", (alert_id,)).fetchone():
            raise KeyError(f"alert {alert_id} not found")
        cur = c.execute("INSERT INTO alert_notes (alert_id, author, body, created_at) VALUES (?,?,?,?)", (alert_id, author.strip(), body, now()))
        return dict(c.execute("SELECT * FROM alert_notes WHERE note_id=?", (cur.lastrowid,)).fetchone())


def list_notes(alert_id, db=None):
    """Notes for one alert, oldest first."""
    with conn(db) as c:
        return [dict(r) for r in c.execute("SELECT * FROM alert_notes WHERE alert_id=? ORDER BY note_id", (alert_id,))]


def note_counts(db=None):
    """{alert_id: number of notes} for the alert list."""
    with conn(db) as c:
        return {r["alert_id"]: r["n"] for r in c.execute("SELECT alert_id, COUNT(*) n FROM alert_notes GROUP BY alert_id")}


# ------------------------------------------------------------------ outcomes
def add_outcome(action_id, alert_id, metric, entity_type, entity_id, action_date, window_days, result, verdict, created_by, db=None):
    with conn(db) as c:
        oid = _next_id(c, "outcomes", "outcome_id", "OUT")
        c.execute("INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (oid, action_id, alert_id, metric, entity_type, str(entity_id), action_date, int(window_days),
                   json.dumps(result, default=str), verdict, now(), created_by))
    return oid


def list_outcomes(action_id=None, limit=100, db=None):
    q, a = "SELECT * FROM outcomes", []
    if action_id:
        q += " WHERE action_id=?"; a.append(action_id)
    with conn(db) as c:
        rows = [dict(r) for r in c.execute(q + " ORDER BY created_at DESC, outcome_id DESC LIMIT ?", a + [limit])]
    for r in rows:
        r["result"] = json.loads(r.pop("result_json"))
    return rows


# ------------------------------------------------------------------ API keys
def _hash(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def create_api_key(name, scope, created_by, db=None):
    """Returns (key_id, plaintext_key). The plaintext key is never stored and cannot be shown again."""
    if scope not in KEY_SCOPES:
        raise ValueError(f"scope must be one of {KEY_SCOPES}")
    if not (name or "").strip():
        raise ValueError("a key name is required")
    key = "ecx_" + secrets.token_urlsafe(32)
    with conn(db) as c:
        kid = _next_id(c, "api_keys", "key_id", "KEY")
        c.execute("INSERT INTO api_keys (key_id,name,key_hash,prefix,scope,created_at,created_by) VALUES (?,?,?,?,?,?,?)",
                  (kid, name.strip()[:80], _hash(key), key[:9], scope, now(), created_by))
    return kid, key


def verify_api_key(key, db=None):
    """Return the key row (without the hash) if the key is valid and not revoked, else None. Records usage."""
    if not key or not key.startswith("ecx_") or len(key) > 200:
        return None
    h = _hash(key)
    with conn(db) as c:
        r = c.execute("SELECT * FROM api_keys WHERE key_hash=?", (h,)).fetchone()
        if r is None or r["revoked"]:
            return None
        if not secrets.compare_digest(r["key_hash"], h):
            return None
        c.execute("UPDATE api_keys SET last_used=?, uses=uses+1 WHERE key_id=?", (now(), r["key_id"]))
        d = dict(r); d.pop("key_hash", None)
        return d


def list_api_keys(db=None):
    with conn(db) as c:
        return [{k: v for k, v in dict(r).items() if k != "key_hash"} for r in c.execute("SELECT * FROM api_keys ORDER BY created_at DESC, key_id DESC")]


def revoke_api_key(key_id, db=None):
    with conn(db) as c:
        if not c.execute("SELECT 1 FROM api_keys WHERE key_id=?", (key_id,)).fetchone():
            raise KeyError(f"key {key_id} not found")
        c.execute("UPDATE api_keys SET revoked=1 WHERE key_id=?", (key_id,))


# ------------------------------------------------------------------ job runs
def start_job(job, db=None):
    with conn(db) as c:
        cur = c.execute("INSERT INTO job_runs (job, started_at, status) VALUES (?,?,?)", (job, now(), "running"))
        return cur.lastrowid


def finish_job(job_id, status, summary, db=None):
    with conn(db) as c:
        c.execute("UPDATE job_runs SET finished_at=?, status=?, summary=? WHERE id=?", (now(), status, json.dumps(summary, default=str)[:2000], job_id))


def list_jobs(limit=20, db=None):
    with conn(db) as c:
        return [dict(r) for r in c.execute("SELECT * FROM job_runs ORDER BY id DESC LIMIT ?", (limit,))]
