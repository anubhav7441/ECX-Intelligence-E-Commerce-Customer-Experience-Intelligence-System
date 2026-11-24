"""Guest 'check my own data' (demo cap), exit/back-button guard and input validation."""
import io
import pandas as pd
import pytest
from app import create_app
from config import AuthTestConfig
from app.auth import security as sec, db as adb
from app.demo import service as demo_service


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIDS_DIR", str(tmp_path / "ds"))
    class C(AuthTestConfig):
        AUTH_DB = str(tmp_path / "auth.db")
        RATE_LIMIT_PER_MINUTE = 1000
    sec.reset_state(); demo_service.reset_state()
    a = create_app(C)
    with a.app_context():
        adb.conn().__enter__()
    return a


def tok(c):
    c.get("/demo")
    with c.session_transaction() as s:
        return s["csrf"]


def csv_bytes(n, cols=3):
    df = pd.DataFrame({f"c{i}": range(n) for i in range(cols)})
    df["v"] = [(i * 37) % 1000 + (5000 if i >= 500 else 0) for i in range(n)]
    df["city"] = ["a", "b", "c", "a"] * (n // 4) + ["a"] * (n % 4)
    return df.to_csv(index=False).encode()


def up(c, name, data, t=None):
    return c.post("/api/demo/upload", data={"file": (io.BytesIO(data), name)}, headers={"X-CSRF-Token": t or tok(c)}, content_type="multipart/form-data")


def test_demo_page_offers_own_data_check(app):
    c = app.test_client()
    html = c.get("/demo").get_data(as_text=True)
    assert 'id="uf"' in html and "Check your own data" in html and "first <span" in html


def test_file_over_500_rows_is_limited_and_says_so(app):
    c = app.test_client()
    r = up(c, "big.csv", csv_bytes(1234))
    assert r.status_code == 200
    j = r.get_json()
    assert j["rows_in_file"] == 1234 and j["rows_used"] == 500 and j["truncated"] is True
    assert "limited to 500 rows only" in j["notice"]
    assert next(c for c in j["columns"] if c["name"] == "v")["max"] < 1000          # statistics come from rows 0..499 only, never the rest


def test_small_file_is_not_marked_truncated(app):
    c = app.test_client()
    j = up(c, "small.csv", csv_bytes(40)).get_json()
    assert j["rows_used"] == 40 and j["truncated"] is False


@pytest.mark.parametrize("name,data,msg", [("x.exe", b"a,b\n1,2", "Only .csv"), ("x.csv", b"", "empty"), ("x.csv", b"a,b\n", "no data"), ("x.csv", b"\x00\x01\x02", "")])
def test_invalid_demo_files_are_refused(app, name, data, msg):
    c = app.test_client()
    r = up(c, name, data)
    assert r.status_code == 400 and msg.lower() in r.get_json()["error"].lower()


def test_demo_file_size_limit(app):
    c = app.test_client()
    big = b"a,b\n" + b"1,2\n" * (1_400_000)             # about 5.6 MB
    r = up(c, "huge.csv", big)
    assert r.status_code in (400, 413) and "5 MB" in r.get_json()["error"]


def test_demo_upload_count_is_limited(app):
    c = app.test_client()
    t = tok(c)
    codes = [up(c, "s.csv", csv_bytes(8), t).status_code for _ in range(7)]
    assert codes[:5] == [200] * 5 and codes[5] == 403


def test_demo_upload_needs_csrf_token(app):
    c = app.test_client()
    c.get("/demo")
    r = c.post("/api/demo/upload", data={"file": (io.BytesIO(b"a\n1"), "a.csv")}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_nothing_from_demo_upload_is_saved(app, tmp_path):
    c = app.test_client()
    up(c, "s.csv", csv_bytes(30))
    assert not list((tmp_path / "ds").rglob("*")) if (tmp_path / "ds").exists() else True


def test_pages_are_never_cached_and_guard_is_loaded(app):
    c = app.test_client()
    r = c.get("/login")
    assert r.headers["Cache-Control"] == "no-store"
    assert "js/guard.js" in r.get_data(as_text=True)
    assert "no-store" in c.get("/").headers["Cache-Control"]


def test_signed_in_pages_are_marked_for_back_button_logout(app):
    c = app.test_client()
    t = tok(c)
    c.post("/register", data={"email": "b@example.com", "name": "Bob Test", "password": "Str0ngPass-123", "confirm": "Str0ngPass-123", "csrf_token": t})
    html = c.get("/workspace").get_data(as_text=True)
    assert 'data-user="1"' in html
    # logging out clears the session: a protected page now redirects to login (what the Back button then hits)
    with c.session_transaction() as s:
        t = s["csrf"]
    c.post("/logout", data={"csrf_token": t})
    assert c.get("/workspace").status_code == 302


def test_guard_script_has_exit_dialog_and_back_handling():
    js = open("app/static/js/guard.js", encoding="utf-8").read()
    assert "Do you really want to exit?" in js and ">Yes<" in js and ">No<" in js and "/logout" in js
    assert "popstate" not in js        # Back must behave like a normal browser Back


def test_login_and_register_reject_bad_input(app):
    c = app.test_client()
    t = tok(c)
    r = c.post("/register", data={"email": "not-an-email", "name": "A", "password": "short", "confirm": "different", "csrf_token": t})
    assert r.status_code == 400
    r = c.post("/login", data={"email": "", "password": "", "csrf_token": t})
    assert r.status_code == 401


# ---- email validation (register) -------------------------------------------------
import pytest
from app.auth import security as _sec


@pytest.mark.parametrize("email", ["@gmail.com", "abc@gmail.com", "a.b@gmail.com", "12345678@gmail.com", "john..doe@gmail.com", ".johndoe@gmail.com",
                                   "johndoe@gmial.com", "johndoe@gmail", "john doe@example.com", "john@@example.com", "john@-example.com", "john@example.c", ""])
def test_invalid_emails_are_rejected(email):
    assert _sec.email_error(email)


@pytest.mark.parametrize("email", ["johndoe@gmail.com", "john.doe92@gmail.com", "johndoe+shop@gmail.com", "name@example.com", "a@company.co.in"])
def test_valid_emails_are_accepted(email):
    assert _sec.email_error(email) is None
