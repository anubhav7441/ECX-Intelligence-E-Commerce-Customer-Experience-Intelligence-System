"""Users, audit log, settings and analysis history (SQLite, stdlib). Passwords are stored only as salted hashes."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from flask import current_app

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL UNIQUE COLLATE NOCASE, name TEXT NOT NULL, password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user','admin')), active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, last_login TEXT);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, user_id INTEGER, user_email TEXT, action TEXT NOT NULL, detail TEXT, ip TEXT);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS analysis_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, owner_id INTEGER NOT NULL, dataset_id TEXT NOT NULL, dataset_name TEXT,
  kind TEXT NOT NULL, analyses TEXT, status TEXT, summary TEXT);
CREATE INDEX IF NOT EXISTS ix_runs_owner ON analysis_runs(owner_id, ts);
CREATE INDEX IF NOT EXISTS ix_audit_ts ON audit_log(ts);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def conn(path=None):
    p = Path(path or current_app.config["AUTH_DB"])
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    c.executescript(SCHEMA)
    have = {r[1] for r in c.execute("PRAGMA table_info(users)")}
    for col, ddl in (("approved", "INTEGER NOT NULL DEFAULT 1"), ("verify_hash", "TEXT"), ("verify_expires", "TEXT"),      # upgrade: old accounts stay approved
                     ("verify_tries", "INTEGER NOT NULL DEFAULT 0"), ("verify_sent", "TEXT")):
        if col not in have:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {ddl}")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def user_by_email(email):
    with conn() as c:
        r = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return dict(r) if r else None


def user_by_id(uid):
    with conn() as c:
        r = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        return dict(r) if r else None


def create_user(email, name, pw_hash, role="user", approved=1):
    with conn() as c:
        cur = c.execute("INSERT INTO users (email,name,password_hash,role,active,approved,created_at) VALUES (?,?,?,?,1,?,?)", (email, name, pw_hash, role, int(approved), now()))
        return cur.lastrowid


def audit(action, detail="", user=None, ip=None):
    uid = user["id"] if user else None
    with conn() as c:
        c.execute("INSERT INTO audit_log (ts,user_id,user_email,action,detail,ip) VALUES (?,?,?,?,?,?)",
                  (now(), uid, user["email"] if user else None, action, str(detail)[:400], ip))


def get_setting(key, default=None):
    with conn() as c:
        r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(r["value"]) if r else default


def set_setting(key, value):
    with conn() as c:
        c.execute("INSERT INTO settings (key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))


def record_run(owner_id, dataset_id, dataset_name, kind, analyses, status, summary):
    with conn() as c:
        c.execute("INSERT INTO analysis_runs (ts,owner_id,dataset_id,dataset_name,kind,analyses,status,summary) VALUES (?,?,?,?,?,?,?,?)",
                  (now(), owner_id, dataset_id, dataset_name, kind, json.dumps(analyses), status, json.dumps(summary, default=str)[:4000]))


def list_runs(owner_id=None, limit=100):
    q, a = "SELECT * FROM analysis_runs", []
    if owner_id is not None:
        q += " WHERE owner_id=?"; a.append(owner_id)
    with conn() as c:
        rows = [dict(r) for r in c.execute(q + " ORDER BY id DESC LIMIT ?", a + [limit])]
    for r in rows:
        r["analyses"] = json.loads(r["analyses"] or "[]")
        r["summary"] = json.loads(r["summary"] or "{}")
    return rows
