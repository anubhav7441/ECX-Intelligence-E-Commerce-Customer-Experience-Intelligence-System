"""Administrator panel (admin-only; enforced by the global guard AND re-checked here)."""
from pathlib import Path
from flask import Blueprint, current_app, g, jsonify, redirect, render_template, request
from app.auth import db as adb, security as sec
from app.demo import service as demo_service

admin = Blueprint("admin", __name__)


def _need_admin():
    if not g.get("user") or g.user["role"] != "admin":
        return jsonify(error="Administrator access required."), 403


admin.before_request(lambda: _need_admin() if request.path.startswith("/api/admin") and current_app.config["AUTH_REQUIRED"] else None)


def _page():
    """The administrator address. An administrator sees the panel; everybody else sees only the administrator sign-in form."""
    if g.get("user") and g.user["role"] == "admin" or not current_app.config["AUTH_REQUIRED"]:
        return render_template("admin.html", active="admin")
    return render_template("admin_login.html", error=None, email="", action=current_app.config["ADMIN_PATH"] + "/login")


def _login():
    from app.auth.routes import login_user
    cfg = current_app.config
    email = (request.form.get("email") or "").strip().lower()
    key = f"{request.remote_addr}|admin|{email}"
    wait = sec.lock_remaining(key, cfg["LOGIN_MAX_FAILURES"], cfg["LOGIN_LOCK_SECONDS"])
    if wait:
        return render_template("admin_login.html", error=f"Too many failed attempts. Try again in {wait // 60 + 1} minute(s).", email=email, action=cfg["ADMIN_PATH"] + "/login"), 429
    user = adb.user_by_email(email) if email else None
    ok = sec.verify_password(user["password_hash"] if user else _DUMMY, request.form.get("password") or "") and user is not None
    if not (ok and user["role"] == "admin" and user["active"] and user.get("approved", 1)):      # one message for every kind of failure
        sec.record_failure(key)
        adb.audit("admin_login_failed", f"email={email[:80]}", None, request.remote_addr)
        return render_template("admin_login.html", error="Invalid administrator email or password.", email=email, action=cfg["ADMIN_PATH"] + "/login"), 401
    sec.clear_failures(key)
    with adb.conn() as c:
        c.execute("UPDATE users SET last_login=? WHERE id=?", (adb.now(), user["id"]))
    login_user(user)
    adb.audit("admin_login", "", user, request.remote_addr)
    return redirect(cfg["ADMIN_PATH"])


_DUMMY = sec.hash_password("not-a-real-password-0")


def register_admin_page(app):
    """The admin page URL comes from config (ECX_ADMIN_PATH, default /control-room); the old /admin address does not exist."""
    app.add_url_rule(app.config["ADMIN_PATH"], endpoint="admin.page", view_func=_page)
    app.add_url_rule(app.config["ADMIN_PATH"] + "/login", endpoint="admin.login", view_func=_login, methods=["POST"])


def _public_user(r):
    return {k: r[k] for k in ("id", "email", "name", "role", "active", "approved", "created_at", "last_login")}      # never the password hash


@admin.get("/api/admin/stats")
def stats():
    from multids import registry as R
    with adb.conn() as c:
        users = c.execute("SELECT COUNT(*) n, SUM(role='admin') a, SUM(active=0) d, SUM(approved=0) p FROM users").fetchone()
        runs = c.execute("SELECT kind, COUNT(*) n FROM analysis_runs GROUP BY kind").fetchall()
        recent = [dict(r) for r in c.execute("SELECT date(ts) day, COUNT(*) n FROM analysis_runs GROUP BY day ORDER BY day DESC LIMIT 14")]
        logins = c.execute("SELECT COUNT(*) FROM audit_log WHERE action='login'").fetchone()[0]
        fails = c.execute("SELECT COUNT(*) FROM audit_log WHERE action='login_failed'").fetchone()[0]
    ds = [d for d in R.list_all() if d["dataset_id"] != R.OLIST_ID]
    return jsonify(users=users["n"], admins=users["a"] or 0, disabled=users["d"] or 0, pending=users["p"] or 0, datasets=len(ds), runs={r["kind"]: r["n"] for r in runs},
                   activity=recent[::-1], logins=logins, failed_logins=fails, demo=demo_service.limits())


@admin.get("/api/admin/users")
def users():
    with adb.conn() as c:
        return jsonify(me=g.user["id"] if g.get("user") else None, users=[_public_user(r) for r in c.execute("SELECT * FROM users ORDER BY approved, id")])


def _admins_left(c, excluding):
    return c.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND active=1 AND id!=?", (excluding,)).fetchone()[0]


@admin.post("/api/admin/users/<int:uid>")
def update_user(uid):
    b = request.get_json(silent=True) or {}
    with adb.conn() as c:
        u = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        if not u:
            return jsonify(error="User not found."), 404
        role = b.get("role", u["role"])
        active = int(bool(b.get("active", u["active"])))
        if role not in ("user", "admin"):
            return jsonify(error="Role must be 'user' or 'admin'."), 400
        if not u["approved"]:
            return jsonify(error="Approve this account first."), 400
        if role == "admin" and u["role"] != "admin":                       # only an existing administrator can reach this endpoint
            adb.audit("admin_granted", f"user={uid} {u['email']}", g.user, request.remote_addr)
        losing_admin = u["role"] == "admin" and u["active"] and (role != "admin" or not active)
        if losing_admin and (uid == g.user["id"] or _admins_left(c, uid) == 0):
            return jsonify(error="You cannot remove or disable the last active administrator (or yourself)."), 400
        c.execute("UPDATE users SET role=?, active=? WHERE id=?", (role, active, uid))
    adb.audit("user_update", f"user={uid} role={role} active={active}", g.user, request.remote_addr)
    return jsonify(ok=True)


@admin.delete("/api/admin/users/<int:uid>")
def delete_user(uid):
    """Remove an account together with its datasets and history. Not yourself, and never the last active administrator."""
    from multids import registry as R
    with adb.conn() as c:
        u = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        if not u:
            return jsonify(error="User not found."), 404
        if g.get("user") and uid == g.user["id"]:
            return jsonify(error="You cannot delete your own account."), 400
        if u["role"] == "admin" and u["active"] and _admins_left(c, uid) == 0:
            return jsonify(error="You cannot delete the last active administrator."), 400
    removed = 0
    for d in R.list_all():
        if d.get("owner_id") == uid and not d.get("read_only") and d["dataset_id"] != R.OLIST_ID:
            try:
                R.delete(d["dataset_id"]); removed += 1
            except (FileNotFoundError, PermissionError):
                pass
    with adb.conn() as c:
        c.execute("DELETE FROM analysis_runs WHERE owner_id=?", (uid,))
        c.execute("DELETE FROM users WHERE id=?", (uid,))
    adb.audit("user_deleted", f"user={uid} {u['email']} datasets_removed={removed}", g.user, request.remote_addr)
    return jsonify(ok=True, datasets_removed=removed)


@admin.post("/api/admin/users/<int:uid>/approve")
def approve_user(uid):
    with adb.conn() as c:
        u = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        if not u:
            return jsonify(error="User not found."), 404
        c.execute("UPDATE users SET approved=1 WHERE id=?", (uid,))
    adb.audit("user_approved", f"user={uid} {u['email']}", g.user, request.remote_addr)
    return jsonify(ok=True)


@admin.post("/api/admin/users/<int:uid>/reject")
def reject_user(uid):
    """Rejecting removes a sign-up that was never approved. Approved accounts can only be disabled, never deleted here."""
    with adb.conn() as c:
        u = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        if not u:
            return jsonify(error="User not found."), 404
        if u["approved"]:
            return jsonify(error="This account is already approved. Disable it instead."), 400
        c.execute("DELETE FROM users WHERE id=?", (uid,))
    adb.audit("user_rejected", f"{u['email']}", g.user, request.remote_addr)
    return jsonify(ok=True)


@admin.get("/api/admin/datasets")
def datasets():
    from multids import registry as R
    return jsonify(datasets=[{**d, "owner_email": _owner_email(d.get("owner_id"))} for d in R.list_all()])


def _owner_email(oid):
    if not oid:
        return None
    u = adb.user_by_id(oid)
    return u["email"] if u else "(deleted user)"


@admin.delete("/api/admin/datasets/<did>")
def delete_dataset(did):
    from multids import registry as R
    if not R.valid_id(did):
        return jsonify(error="Invalid dataset id."), 404
    try:
        R.delete(did)
    except PermissionError as e:
        return jsonify(error=str(e)), 403
    except FileNotFoundError:
        return jsonify(error="Dataset not found."), 404
    adb.audit("dataset_delete_admin", did, g.user, request.remote_addr)
    return jsonify(deleted=did)


@admin.get("/api/admin/activity")
def activity():
    with adb.conn() as c:
        rows = [dict(r) for r in c.execute("SELECT r.*, u.email FROM analysis_runs r LEFT JOIN users u ON u.id=r.owner_id ORDER BY r.id DESC LIMIT 100")]
    return jsonify(activity=rows)


@admin.get("/api/admin/audit")
def audit_log():
    with adb.conn() as c:
        rows = [dict(r) for r in c.execute("SELECT id, ts, user_email, action, detail, ip FROM audit_log ORDER BY id DESC LIMIT 200")]
    return jsonify(audit=rows)


@admin.get("/api/admin/logs")
def logs():
    f = current_app.config.get("LOG_FILE")
    if not f or not Path(f).exists():
        return jsonify(lines=[], note="No log file yet.")
    with open(f, "rb") as fh:
        fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 40000))
        txt = fh.read().decode("utf-8", "replace")
    return jsonify(lines=txt.splitlines()[-200:])


@admin.route("/api/admin/demo-config", methods=["GET", "POST"])
def demo_config():
    if request.method == "POST":
        b = request.get_json(silent=True) or {}
        spec = {"demo_row_limit": (50, demo_service.HARD_MAX_ROWS), "demo_max_predictions": (0, 100), "demo_max_investigations": (0, 50)}
        for k, (lo, hi) in spec.items():
            if k in b:
                try:
                    v = int(b[k])
                except (TypeError, ValueError):
                    return jsonify(error=f"{k} must be a whole number."), 400
                if not lo <= v <= hi:
                    return jsonify(error=f"{k} must be between {lo} and {hi}."), 400
                adb.set_setting(k, str(v))
        adb.audit("demo_config", str({k: b[k] for k in spec if k in b}), g.user, request.remote_addr)
    return jsonify(demo_service.limits())
