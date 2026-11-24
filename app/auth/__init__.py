"""Authentication, roles, CSRF and the default-deny route guard.

Roles:  guest (no account; demo only)  <  user  <  admin.
Default-deny: every route is protected unless it is explicitly listed as public. This protects the existing v1/v2 pages AND APIs
without having to decorate each of them."""
import secrets
from flask import current_app, g, jsonify, redirect, request, session, url_for

PUBLIC_EXACT = {"/", "/login", "/register", "/verify", "/verify/resend", "/healthz", "/readyz", "/demo", "/favicon.ico"}
PUBLIC_PREFIX = ("/static/", "/api/demo/")
ADMIN_PREFIX = ("/admin", "/api/admin/")
UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}


def init_auth(app):
    from app.auth import db as adb, security as sec
    from app.auth.routes import auth
    app.register_blueprint(auth)

    @app.before_request
    def guard():
        g.user, g.role = None, "guest"
        uid = session.get("uid")
        if uid:
            u = adb.user_by_id(uid)
            if u and u["active"] and u.get("approved", 1):
                g.user, g.role = u, u["role"]
            else:
                session.clear()
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(24)
        cfg = current_app.config
        # ---- /api/v1/ uses API-key (Bearer) authentication handled by its own blueprint; browser cookies are never used there,
        #      so CSRF does not apply and the login gate is not needed. The key check is mandatory (see app/routes/ops.py).
        if request.path.startswith("/api/v1/") or request.path == "/metrics":   # /metrics has its own bearer-token check (off by default)
            return None
        # ---- CSRF for state-changing requests
        if cfg["CSRF_ENABLED"] and request.method in UNSAFE:
            tok = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
            if not tok or not secrets.compare_digest(str(tok), str(session.get("csrf", ""))):
                return _deny("Security token missing or expired. Reload the page and try again.", 400)
        if not cfg["AUTH_REQUIRED"]:
            return None
        path = request.path
        if path == "/admin":                                                             # the old address no longer exists
            from flask import render_template
            return render_template("error.html", code=404, message="Page not found."), 404
        admin_entry = path in (cfg["ADMIN_PATH"], cfg["ADMIN_PATH"] + "/login")          # the admin page shows its own sign-in form to non-admins
        public = path in PUBLIC_EXACT or path.startswith(PUBLIC_PREFIX) or admin_entry
        if path.startswith(ADMIN_PREFIX) and not admin_entry:
            if g.role != "admin":
                return _deny("Administrator access required.", 403 if g.user else 401)
            return None
        if public:
            return None
        if g.user is None:
            return _deny("Please log in to continue.", 401)
        if request.method in UNSAFE or path.startswith("/api/md/") and request.method == "GET" and False:
            pass
        # per-user rate limit on expensive endpoints
        if request.method in UNSAFE and path.startswith(("/api/v2/investigate", "/api/v2/simulate", "/api/md/", "/api/v2/assistant", "/api/v2/research", "/api/predict")):
            if sec.rate_limited(f"u{g.user['id']}", cfg["RATE_LIMIT_PER_MINUTE"]):
                return _deny("Too many requests. Please wait a minute.", 429)
        return None

    @app.context_processor
    def inject():
        return {"user": g.get("user"), "role": g.get("role", "guest"), "csrf_token": session.get("csrf", ""), "auth_on": current_app.config["AUTH_REQUIRED"]}

    @app.after_request
    def headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        resp.headers.setdefault("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        if not request.path.startswith("/static/"):          # pages are never cached: after logout the Back button cannot show a private page
            resp.headers["Cache-Control"] = "no-store"
        return resp


def _deny(msg, code):
    if request.path.startswith("/api/") or request.is_json or request.headers.get("X-Requested-With"):
        return jsonify(error=msg), code
    if code == 401:
        return redirect(url_for("auth.login", next=request.full_path.rstrip("?") if request.method == "GET" else None))
    from flask import render_template
    return render_template("error.html", code=code, message=msg), code
