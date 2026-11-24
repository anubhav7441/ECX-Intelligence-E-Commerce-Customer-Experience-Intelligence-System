"""Evidence Ledger + Action Center storage (SQLite, stdlib only). Separate DB file from v1's ecx.db.

Human-in-the-loop rules enforced here:
  * every action starts as 'proposed'; only a NAMED reviewer can approve/reject it
  * the system never executes real-world actions - 'completed' just records a human's completion note
  * every status change is written to action_history
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path(os.environ.get("ECX_APP_DB", ROOT / "database" / "ecx_app.db"))
STATUSES = ["proposed", "approved", "rejected", "in_progress", "completed"]
TRANSITIONS = {"proposed": {"approved", "rejected"}, "approved": {"in_progress", "completed", "rejected"},
               "in_progress": {"completed"}, "rejected": set(), "completed": set()}
PRIORITIES = ["Low", "Medium", "High", "Critical"]
ROLES = ["Operations Manager", "Seller Manager", "Customer Support Lead", "Logistics Manager", "Data Analyst", "Category Manager"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
  finding_id TEXT PRIMARY KEY, description TEXT NOT NULL, source_module TEXT NOT NULL, entity_type TEXT, entity_id TEXT,
  metric TEXT, model_output TEXT, stat_result TEXT, created_at TEXT NOT NULL, confidence TEXT, limitation TEXT,
  recommendation TEXT, run_id TEXT);
CREATE TABLE IF NOT EXISTS actions (
  action_id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT, priority TEXT NOT NULL, reason TEXT,
  evidence_ids TEXT, owner_role TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  completion_notes TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS action_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, action_id TEXT NOT NULL, ts TEXT NOT NULL, from_status TEXT, to_status TEXT NOT NULL,
  actor TEXT NOT NULL, note TEXT);
CREATE TABLE IF NOT EXISTS assistant_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, question TEXT, route TEXT, grounded INTEGER, refused INTEGER, answer TEXT);
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


def add_evidence(description, source_module, entity_type=None, entity_id=None, metric=None, model_output=None, stat_result=None,
                 confidence="Medium", limitation="", recommendation="", run_id=None, db=None):
    if not description or not source_module:
        raise ValueError("description and source_module are required")
    with conn(db) as c:
        fid = _next_id(c, "evidence", "finding_id", "EV")
        c.execute("INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (fid, description, source_module, entity_type, entity_id, metric,
                   json.dumps(model_output, default=str) if model_output is not None else None,
                   json.dumps(stat_result, default=str) if stat_result is not None else None,
                   now(), confidence, limitation, recommendation, run_id))
    return fid


def list_evidence(entity_id=None, run_id=None, limit=200, db=None):
    q, a = "SELECT * FROM evidence WHERE 1=1", []
    if entity_id:
        q += " AND entity_id=?"; a.append(entity_id)
    if run_id:
        q += " AND run_id=?"; a.append(run_id)
    with conn(db) as c:
        return [dict(r) for r in c.execute(q + " ORDER BY created_at DESC, finding_id DESC LIMIT ?", a + [limit])]


def propose_action(title, description, priority, reason, evidence_ids, owner_role, source="agent", db=None):
    if priority not in PRIORITIES:
        raise ValueError(f"priority must be one of {PRIORITIES}")
    if not title or not evidence_ids:
        raise ValueError("an action needs a title and at least one evidence id")
    with conn(db) as c:
        aid = _next_id(c, "actions", "action_id", "ACT")
        t = now()
        c.execute("INSERT INTO actions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (aid, title, description, priority, reason, json.dumps(list(evidence_ids)), owner_role, "proposed", t, t, "", source))
        c.execute("INSERT INTO action_history (action_id, ts, from_status, to_status, actor, note) VALUES (?,?,?,?,?,?)",
                  (aid, t, None, "proposed", source, "created as a proposal; awaiting human review"))
    return aid


def set_status(action_id, new_status, actor, note="", db=None):
    if new_status not in STATUSES:
        raise ValueError(f"unknown status '{new_status}'")
    if not actor or not actor.strip():
        raise ValueError("a named reviewer/actor is required for every status change")
    if new_status in ("approved", "rejected", "completed") and not (note or "").strip() and new_status != "approved":
        raise ValueError(f"a note is required when marking an action '{new_status}'")
    with conn(db) as c:
        r = c.execute("SELECT status FROM actions WHERE action_id=?", (action_id,)).fetchone()
        if r is None:
            raise KeyError(f"action {action_id} not found")
        cur = r["status"]
        if new_status not in TRANSITIONS[cur]:
            raise ValueError(f"cannot move from '{cur}' to '{new_status}' (allowed: {sorted(TRANSITIONS[cur]) or 'none'})")
        t = now()
        notes = note if new_status == "completed" else None
        c.execute("UPDATE actions SET status=?, updated_at=?, completion_notes=COALESCE(?, completion_notes) WHERE action_id=?",
                  (new_status, t, notes, action_id))
        c.execute("INSERT INTO action_history (action_id, ts, from_status, to_status, actor, note) VALUES (?,?,?,?,?,?)",
                  (action_id, t, cur, new_status, actor.strip(), note))
    return get_action(action_id, db=db)


def get_action(action_id, db=None):
    with conn(db) as c:
        r = c.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
        if r is None:
            return None
        a = dict(r)
        a["evidence_ids"] = json.loads(a["evidence_ids"] or "[]")
        a["history"] = [dict(h) for h in c.execute("SELECT * FROM action_history WHERE action_id=? ORDER BY id", (action_id,))]
        return a


def list_actions(status=None, db=None):
    with conn(db) as c:
        rows = c.execute("SELECT action_id FROM actions" + (" WHERE status=?" if status else "") + " ORDER BY created_at DESC, action_id DESC",  # nosec B608
                         ([status] if status else [])).fetchall()
    return [get_action(r["action_id"], db=db) for r in rows]


def log_assistant(question, route, grounded, refused, answer, db=None):
    with conn(db) as c:
        c.execute("INSERT INTO assistant_log (ts, question, route, grounded, refused, answer) VALUES (?,?,?,?,?,?)",
                  (now(), question[:500], route, int(grounded), int(refused), answer[:2000]))


def counts(db=None):
    with conn(db) as c:
        return {"evidence": c.execute("SELECT COUNT(*) FROM evidence").fetchone()[0],
                "actions": {s: c.execute("SELECT COUNT(*) FROM actions WHERE status=?", (s,)).fetchone()[0] for s in STATUSES}}
