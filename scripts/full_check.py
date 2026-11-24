"""One command that checks the whole project is alive:   python scripts\\full_check.py
 1. every library in requirements.txt can be imported (versions are printed)
 2. every saved model file opens
 3. the app starts and EVERY page / read-only API answers without a server error, as guest, normal user and administrator
 4. the access rules still hold (guests, normal users and administrators see only what they should)
It uses a temporary database and temporary folders, so your real users and files are never touched. Exit code 0 = all good."""
import importlib
import importlib.metadata as md
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
TMP = tempfile.mkdtemp(prefix="ecx_check_")
os.environ["MULTIDS_DIR"] = os.path.join(TMP, "ds")
PW = "Check-Pass-2026!"
fails, warns = [], []


def head(t):
    print(f"\n=== {t}")


def ok(m): print("  OK   ", m)
def bad(m): fails.append(m); print("  FAIL ", m)
def warn(m): warns.append(m); print("  note ", m)


# ---------------------------------------------------------------- 1. libraries
head("1. Libraries")
IMPORT_NAME = {"scikit-learn": "sklearn", "pyyaml": "yaml", "python-dotenv": "dotenv", "pillow": "PIL"}
for line in (ROOT / "requirements.txt").read_text().splitlines():
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([\w.\-]+)", line)
    if not m:
        continue
    name, want = m.group(1), m.group(2)
    mod = IMPORT_NAME.get(name.lower(), name.lower().replace("-", "_"))
    try:
        importlib.import_module(mod)
        have = md.version(name)
        (ok if have == want else warn)(f"{name} {have}" + ("" if have == want else f"  (project built with {want})"))
    except Exception as e:
        bad(f"{name}: cannot import ({type(e).__name__}: {str(e)[:80]})")

# ---------------------------------------------------------------- 2. models
head("2. Saved models")
try:
    import joblib
    from ml.portable import load_artifact_file          # the same loader the app uses (falls back to models/portable/)
    files = sorted((ROOT / "models").rglob("*.joblib"))
    for f in files:
        rel = str(f.relative_to(ROOT))
        try:
            joblib.load(f); ok(rel)
        except Exception as first:
            try:
                load_artifact_file(f)                                   # works only for the main XGBoost models that have a portable copy
                ok(f"{rel}  (this PC cannot unpickle it; the app uses the portable copy instead: normal on some Windows setups)")
            except Exception as second:
                ubj = ROOT / "models" / "portable" / f"{f.stem}.ubj"
                why = f"the portable copy models/portable/{f.stem}.ubj is MISSING" if not ubj.exists() else f"the portable copy exists ({ubj.stat().st_size} bytes) but also failed: {type(second).__name__}: {str(second)[:90]}"
                bad(f"{rel}: original unreadable ({type(first).__name__}: {str(first)[:50]}); {why}")
    if not files:
        warn("no .joblib model files found in models/")
except Exception as e:
    bad(f"joblib: {e}")

# ---------------------------------------------------------------- 3. app and pages
head("3. Pages and read-only APIs")
try:
    from app import create_app
    from app.auth import db as adb, security as sec
    from config import AuthTestConfig

    class C(AuthTestConfig):
        AUTH_DB = os.path.join(TMP, "auth.db")
        APP_DB = os.path.join(TMP, "app.db")
        OPS_DB = os.path.join(TMP, "ops.db")
        DIGEST_DIR = os.path.join(TMP, "digests")
        CSRF_ENABLED = False
        SIGNUP_MODE = "open"
        RATE_LIMIT_PER_MINUTE = 100000
        LOGIN_MAX_FAILURES = 100000
    app = create_app(C)
except Exception as e:
    bad(f"the app does not start: {type(e).__name__}: {e}")
    print("\nRESULT: FAILED (the app does not start)"); sys.exit(1)

with app.app_context():
    adb.create_user("user@check.local", "Check User", sec.hash_password(PW), "user")
    adb.create_user("admin@check.local", "Check Admin", sec.hash_password(PW), "admin")
ADMIN = app.config["ADMIN_PATH"]
routes = sorted({r.rule for r in app.url_map.iter_rules() if "GET" in r.methods and not r.arguments and not r.rule.startswith("/static")})
SKIP = {"/metrics"}


def client(email=None):
    c = app.test_client()
    if email:
        c.post("/login", data={"email": email, "password": PW})
    return c


def sweep(label, c, signed_in):
    """Visit every page. The home page is visited LAST because opening it deliberately ends a session (original design)."""
    seen = {}
    order = [r for r in routes if r not in SKIP and r != "/"] + ["/"]
    for r in order:
        try:
            seen[r] = c.get(r).status_code
        except Exception as e:
            bad(f"{label}: {r} crashed ({type(e).__name__}: {str(e)[:70]})"); continue
        if seen[r] >= 500:
            bad(f"{label}: {r} -> server error {seen[r]}")
        if signed_in and r != "/" and seen[r] == 401 and not r.startswith("/api/v1/"):     # /api/v1/ needs an API key by design; everything else must know the signed-in person
            bad(f"{label}: {r} -> 401, the test account was logged out")
    n200 = sum(1 for v in seen.values() if v == 200); n302 = sum(1 for v in seen.values() if v == 302)
    other = {r: v for r, v in seen.items() if v not in (200, 302) and v < 500}
    ok(f"{label}: {n200} pages opened (200), {n302} redirected, {len(other)} refused, 0 server errors" if not [f for f in fails if f.startswith(label)] else f"{label}: see failures above")
    for r, v in other.items():
        expected = (not signed_in) or r.startswith("/api/v1/") or (label.strip() == "user" and v == 403 and ("/admin" in r))
        if not expected:                                                  # guests being refused, API-key routes and admin-only routes are all correct
            warn(f"{label}: {r} -> {v}")


sweep("guest", client(), False)
sweep("user ", client("user@check.local"), True)
sweep("admin", client("admin@check.local"), True)

# ---------------------------------------------------------------- 4. access rules
head("4. Access rules")
g, u, a = client(), client("user@check.local"), client("admin@check.local")


def rule(text, cond):
    (ok if cond else bad)(text)


rule("guest is sent to the login page for private pages", g.get("/workspace").status_code == 302)
rule("guest can open the home, login, register and demo pages", all(g.get(p).status_code == 200 for p in ("/", "/login", "/register", "/demo")))
rule("guest cannot read the admin data", g.get("/api/admin/users").status_code in (401, 403))
rule("normal user can open the workspace", u.get("/workspace").status_code == 200)
rule("normal user cannot read the admin data", u.get("/api/admin/users").status_code == 403)
rule("normal user sees only the sign-in form at the admin address", b"Administrator sign-in" in u.get(ADMIN).data and b"Users, datasets, activity" not in u.get(ADMIN).data)
rule("normal user does not see an Admin Panel link", b"Admin Panel" not in u.get("/workspace").data)
rule("old /admin address does not exist", a.get("/admin").status_code == 404)
rule("administrator opens the admin panel", b"Users, datasets, activity" in a.get(ADMIN).data)
rule("administrator can read the admin data", a.get("/api/admin/users").status_code == 200)

print("\n" + "=" * 60)
print(f"RESULT: {'ALL GOOD' if not fails else 'PROBLEMS FOUND'}   |   {len(fails)} failure(s), {len(warns)} note(s)")
for f in fails:
    print("  -", f)
sys.exit(1 if fails else 0)
