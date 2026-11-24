"""Application settings. Values can be overridden with environment variables (see .env.example)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_env_file(path=None):
    """Read KEY=VALUE lines from the .env file in the project folder into the environment (no extra package needed).
    Lines starting with # and blank lines are ignored; a value already set in the real environment is never overwritten."""
    p = Path(path) if path else BASE_DIR / ".env"
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k and k.replace("_", "").isalnum():
            os.environ.setdefault(k, v)


load_env_file()


def _secret_key():
    """Use the SECRET_KEY environment variable if set; otherwise create a random key once and keep it in database/.secret_key
    (so no key is written in the source code and sessions survive restarts)."""
    k = os.environ.get("SECRET_KEY")
    if k:
        return k
    f = BASE_DIR / "database" / ".secret_key"
    try:
        if f.exists():
            return f.read_text().strip()
        import secrets
        f.parent.mkdir(parents=True, exist_ok=True)
        k = secrets.token_hex(32)
        f.write_text(k)
        return k
    except OSError:
        import secrets
        return secrets.token_hex(32)           # read-only disk: key lasts until restart


class Config:
    SECRET_KEY = _secret_key()
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024          # 20 MB upload limit
    DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{BASE_DIR / 'database' / 'ecx.db'}")
    MAX_REVIEW_CHARS = 2000
    JSON_SORT_KEYS = False
    # ---- authentication / sessions (Phase 3). Set SECRET_KEY as an environment variable for anything beyond local development.
    AUTH_REQUIRED = os.environ.get("ECX_AUTH_REQUIRED", "1") != "0"
    CSRF_ENABLED = True
    AUTH_DB = os.environ.get("ECX_AUTH_DB", str(BASE_DIR / "database" / "ecx_auth.db"))
    # How a new sign-up becomes active:
    #   open  = immediately (no check)         email = a 6-digit code sent to the person's e-mail address
    #   admin = an administrator must approve  auto  = "email" if the ECX_SMTP_* settings are set, otherwise "open"
    SIGNUP_MODE = os.environ.get("ECX_SIGNUP_MODE", "auto").strip().lower()
    # The administrator page lives on its own URL (not /admin). Change it in .env with ECX_ADMIN_PATH=/my-secret-path
    ADMIN_PATH = "/" + os.environ.get("ECX_ADMIN_PATH", "/control-room").strip().strip("/")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.environ.get("ECX_COOKIE_SECURE", "0") == "1"   # set to 1 when served over HTTPS
    PERMANENT_SESSION_LIFETIME = 8 * 3600
    MIN_PASSWORD_LENGTH = 10
    LOGIN_MAX_FAILURES = 5
    LOGIN_LOCK_SECONDS = 300
    RATE_LIMIT_PER_MINUTE = 40                     # expensive endpoints, per user
    # ---- guest demo (Phase 4). Enforced on the backend; never applies to registered users or admins.
    DEMO_ROW_LIMIT = 500
    DEMO_MAX_PREDICTIONS = 10                      # per guest session
    DEMO_MAX_UPLOADS = 5                           # per guest session: own-data checks
    DEMO_UPLOAD_MAX_BYTES = 5 * 1024 * 1024        # guest files above 5 MB are refused
    DEMO_MAX_INVESTIGATIONS = 3                    # per guest session
    DEMO_SAMPLE_FILE = str(BASE_DIR / "data" / "demo" / "demo_sample.parquet")
    LOG_FILE = str(BASE_DIR / "logs" / "ecx.log")
    APP_DB = os.environ.get("ECX_APP_DB")        # v2 evidence ledger / action center (default: database/ecx_app.db)
    OPS_DB = os.environ.get("ECX_OPS_DB")        # operations: alerts, outcomes, API keys (default: database/ecx_ops.db)
    DIGEST_DIR = os.environ.get("ECX_DIGEST_DIR")  # where digest files are written (default: reports/digests)
    METRICS_TOKEN = os.environ.get("ECX_METRICS_TOKEN")   # unset = /metrics is disabled (404)
    API_RATE_LIMIT_PER_MINUTE = int(os.environ.get("ECX_API_RATE_LIMIT", "120"))   # per API key


class TestConfig(Config):
    """Used by the original functional tests: authentication gate OFF so v1/v2 behaviour is tested exactly as before.
    Authentication itself is tested with AuthTestConfig (tests/test_auth_flow.py)."""
    TESTING = True
    AUTH_REQUIRED = False
    SIGNUP_MODE = "open"
    CSRF_ENABLED = False
    LOG_FILE = None
    import tempfile as _t2
    AUTH_DB = os.path.join(_t2.mkdtemp(), "auth_test.db")
    import tempfile as _t
    APP_DB = os.path.join(_t.mkdtemp(), "test_app.db")
    OPS_DB = os.path.join(_t.mkdtemp(), "test_ops.db")
    DIGEST_DIR = os.path.join(_t.mkdtemp(), "digests")


class AuthTestConfig(TestConfig):
    AUTH_REQUIRED = True
    CSRF_ENABLED = True
    MIN_PASSWORD_LENGTH = 10
