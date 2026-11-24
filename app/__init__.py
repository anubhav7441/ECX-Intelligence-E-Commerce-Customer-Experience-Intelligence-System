"""ECX-Intelligence Flask application factory."""
from flask import Flask, jsonify, render_template
from config import Config


def _setup_logging(app):
    import logging
    import os
    from logging.handlers import RotatingFileHandler
    from pathlib import Path
    f = app.config.get("LOG_FILE")
    if f:
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        h = RotatingFileHandler(f, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        if os.environ.get("ECX_LOG_JSON") == "1":
            from app.observability import JsonFormatter
            h.setFormatter(JsonFormatter())
        else:
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        app.logger.addHandler(h)
        app.logger.setLevel(logging.INFO)


def create_app(config_object=Config):
    app = Flask(__name__)
    app.config.from_object(config_object)
    _setup_logging(app)
    from app.auth import init_auth
    init_auth(app)
    from app.observability import init_observability
    init_observability(app)

    from app.routes.pages import pages
    from app.routes.api import api
    app.register_blueprint(pages)
    app.register_blueprint(api, url_prefix="/api")
    from app.routes.v2 import v2
    app.register_blueprint(v2)
    from app.routes.multids import md
    app.register_blueprint(md)
    from app.demo.routes import demo
    app.register_blueprint(demo)
    from app.insights.routes import insights
    app.register_blueprint(insights)
    from app.admin.routes import admin, register_admin_page
    app.register_blueprint(admin)
    register_admin_page(app)
    from app.routes.recsys import rec
    app.register_blueprint(rec)
    from app.routes.ops import ops
    app.register_blueprint(ops)

    @app.get("/")
    def landing():
        from flask import g, session, request
        if g.get("user"):                      # opening the home page ends the session: the person must sign in again
            try:
                from app.auth import db as adb
                adb.audit("logout", "home page opened", g.user, request.remote_addr)
            except Exception:
                pass
            session.clear()
            g.user = None
        return render_template("landing.html", logged_in=False)

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok")

    @app.get("/readyz")
    def readyz():
        """Readiness check for a load balancer or container: are the databases writable and the data and models present?
        Returns only yes/no per check (no paths or details), and 503 if anything is missing."""
        import sqlite3
        from pathlib import Path
        base = Path(__file__).resolve().parents[1]
        checks = {}
        for name, p in {"auth_db": app.config.get("AUTH_DB"), "ops_db": app.config.get("OPS_DB") or str(base / "database" / "ecx_ops.db"),
                        "app_db": app.config.get("APP_DB") or str(base / "database" / "ecx_app.db")}.items():
            try:
                Path(p).parent.mkdir(parents=True, exist_ok=True)
                c = sqlite3.connect(p); c.execute("CREATE TABLE IF NOT EXISTS _ready (x INTEGER)"); c.commit(); c.close()
                checks[name] = True
            except Exception:
                checks[name] = False
        checks["orders_data"] = (base / "data" / "processed" / "orders_with_history.parquet").exists()
        checks["models"] = any((base / "models").glob("*.joblib")) or any((base / "models" / "portable").glob("*")) if (base / "models").exists() else False
        ok = all(checks.values())
        return jsonify(status="ready" if ok else "not_ready", checks=checks), (200 if ok else 503)

    @app.errorhandler(404)
    def not_found(e):
        from flask import request
        if request.path.startswith("/api/"):
            return jsonify(error="Not found"), 404
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.errorhandler(413)
    def too_large(e):
        return jsonify(error="File is larger than the 20 MB limit."), 413

    @app.errorhandler(500)
    def server_error(e):
        from flask import request
        if request.path.startswith("/api/"):
            app.logger.exception("Unhandled error on %s", request.path)
            return jsonify(error="Internal server error."), 500
        return render_template("error.html", code=500, message="Something went wrong on the server."), 500

    return app
