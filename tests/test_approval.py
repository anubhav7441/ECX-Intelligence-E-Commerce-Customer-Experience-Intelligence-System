"""Sign-up approval, separate admin URL and admin-only promotion."""
import pytest
from app import create_app
from app.auth import db as adb, security as sec
from app.demo import service as demo_service
from config import AuthTestConfig
from tests.test_auth_flow import PW, hdr, login, make_admin, reg


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIDS_DIR", str(tmp_path / "ds"))

    class C(AuthTestConfig):
        AUTH_DB = str(tmp_path / "auth.db")
        RATE_LIMIT_PER_MINUTE = 1000
        SIGNUP_MODE = "admin"
        ADMIN_PATH = "/control-room"
    sec.reset_state(); demo_service.reset_state()
    return create_app(C)


def uid(app, email):
    with app.app_context():
        return adb.user_by_email(email)["id"]


def test_new_signup_cannot_log_in_until_approved(app):
    c = app.test_client()
    r = reg(c)
    assert r.status_code == 302 and "/login" in r.headers["Location"]            # no auto-login
    assert c.get("/workspace").status_code == 302                                  # still a guest
    r = login(c)
    assert r.status_code == 403 and b"waiting for administrator approval" in r.data
    assert c.get("/workspace").status_code == 302
    r = login(c, pw="WrongPass-12345")                                             # wrong password reveals nothing about approval
    assert r.status_code == 401 and b"Invalid email or password" in r.data
    assert b"must approve your account" in c.get("/login?pending=1").data


def test_admin_approves_then_user_can_log_in(app):
    make_admin(app)
    u = app.test_client(); reg(u)
    a = app.test_client(); assert login(a, "admin@example.com").status_code == 302
    users = a.get("/api/admin/users").get_json()["users"]
    assert [x for x in users if x["email"] == "a@example.com"][0]["approved"] == 0
    assert a.get("/api/admin/stats").get_json()["pending"] == 1
    assert a.post(f"/api/admin/users/{uid(app, 'a@example.com')}", json={"role": "admin"}, headers=hdr(a)).status_code == 400   # approve first
    assert a.post(f"/api/admin/users/{uid(app, 'a@example.com')}/approve", headers=hdr(a)).status_code == 200
    assert login(u).status_code == 302 and u.get("/workspace").status_code == 200
    assert a.get("/api/admin/stats").get_json()["pending"] == 0


def test_admin_can_reject_a_pending_signup_only(app):
    make_admin(app)
    reg(app.test_client())
    a = app.test_client(); login(a, "admin@example.com")
    i = uid(app, "a@example.com")
    assert a.post(f"/api/admin/users/{i}/reject", headers=hdr(a)).status_code == 200
    with app.app_context():
        assert adb.user_by_email("a@example.com") is None
    reg(app.test_client(), "b@example.com", "Bob Test")
    j = uid(app, "b@example.com"); a.post(f"/api/admin/users/{j}/approve", headers=hdr(a))
    assert a.post(f"/api/admin/users/{j}/reject", headers=hdr(a)).status_code == 400     # approved accounts are never deleted this way


def test_unapproved_session_is_not_trusted(app):
    c = app.test_client(); reg(c)
    with app.app_context():
        adb.conn().__enter__()
    with c.session_transaction() as s:
        s["uid"] = uid(app, "a@example.com")                                            # even a forged session does not help
    assert c.get("/workspace").status_code == 302


def test_normal_users_cannot_approve_or_promote(app):
    make_admin(app)
    u = app.test_client(); reg(u)
    a = app.test_client(); login(a, "admin@example.com")
    a.post(f"/api/admin/users/{uid(app, 'a@example.com')}/approve", headers=hdr(a)); login(u)
    i = uid(app, "a@example.com")
    for path in (f"/api/admin/users/{i}/approve", f"/api/admin/users/{i}/reject"):
        assert u.post(path, headers=hdr(u)).status_code == 403
    assert u.post(f"/api/admin/users/{i}", json={"role": "admin"}, headers=hdr(u)).status_code == 403
    assert a.post(f"/api/admin/users/{i}", json={"role": "admin"}, headers=hdr(a)).status_code == 200   # only an admin can do it


def test_signup_always_creates_a_normal_user(app):
    reg(app.test_client(), "x@example.com", "Xavier Test")
    with app.app_context():
        assert adb.user_by_email("x@example.com")["role"] == "user"


def admin_form(c, email, pw):
    return c.post("/control-room/login", data={"email": email, "password": pw, "csrf_token": hdr(c)["X-CSRF-Token"]})


def test_admin_url_opens_admin_signin_and_then_the_panel(app):
    make_admin(app)
    c = app.test_client()                                                                # nobody logged in
    h = c.get("/control-room").data
    assert b"Administrator sign-in" in h and b"Users, datasets, activity" not in h
    assert c.get("/admin").status_code == 404                                            # the old address is gone
    assert c.get("/api/admin/users").status_code in (401, 403)                           # the data is still locked
    r = admin_form(c, "admin@example.com", PW)
    assert r.status_code == 302 and r.headers["Location"].endswith("/control-room")
    assert b"Users, datasets, activity" in c.get("/control-room").data                  # now the real panel
    assert c.get("/api/admin/users").status_code == 200


def test_admin_signin_rejects_everyone_else_with_one_message(app):
    make_admin(app)
    u = app.test_client(); reg(u)
    a = app.test_client(); login(a, "admin@example.com")
    a.post(f"/api/admin/users/{uid(app, 'a@example.com')}/approve", headers=hdr(a))
    c = app.test_client()
    msgs = []
    for email, pw in (("a@example.com", PW), ("admin@example.com", "WrongPass-12345"), ("nobody@example.com", PW)):
        r = admin_form(c, email, pw)
        assert r.status_code == 401
        msgs.append(b"Invalid administrator email or password." in r.data)
    assert all(msgs)                                                                     # a normal user's correct password is NOT enough
    assert c.get("/api/admin/users").status_code in (401, 403)


def test_admin_signin_locks_after_repeated_failures(app):
    make_admin(app)
    c = app.test_client()
    codes = [admin_form(c, "admin@example.com", "WrongPass-12345").status_code for _ in range(8)]
    assert 429 in codes
    assert admin_form(c, "admin@example.com", PW).status_code == 429


def test_normal_user_logged_in_still_gets_only_the_form(app):
    make_admin(app)
    u = app.test_client(); reg(u)
    a = app.test_client(); login(a, "admin@example.com")
    a.post(f"/api/admin/users/{uid(app, 'a@example.com')}/approve", headers=hdr(a)); login(u)
    assert b"Administrator sign-in" in u.get("/control-room").data
    assert b"Admin Panel" not in u.get("/workspace").data                                # the sidebar link is still hidden from them
    assert b"Admin Panel" in a.get("/workspace").data


def test_existing_accounts_stay_approved_after_upgrade(tmp_path):
    import sqlite3
    p = tmp_path / "old.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL UNIQUE COLLATE NOCASE, name TEXT NOT NULL, password_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user','admin')), active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, last_login TEXT)")
    c.execute("INSERT INTO users (email,name,password_hash,role,created_at) VALUES ('old@example.com','Old User','x','user','2026-01-01')")
    c.commit(); c.close()
    from flask import Flask
    a = Flask(__name__); a.config["AUTH_DB"] = str(p)
    with a.app_context():
        assert adb.user_by_email("old@example.com")["approved"] == 1


# ------------------------------------------------------------------ e-mail code verification
@pytest.fixture()
def mail_app(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIDS_DIR", str(tmp_path / "ds"))
    sent = []
    from app.auth import mailer
    monkeypatch.setattr(mailer, "send", lambda to, subject, body, env=None: sent.append((to, body)))

    class C(AuthTestConfig):
        AUTH_DB = str(tmp_path / "auth.db")
        RATE_LIMIT_PER_MINUTE = 1000
        SIGNUP_MODE = "email"
    sec.reset_state(); demo_service.reset_state()
    a = create_app(C); a.sent = sent
    return a


def code_of(app):
    import re
    return re.search(r"code is (\d{6})", app.sent[-1][1]).group(1)


def post_code(c, code):
    return c.post("/verify", data={"code": code, "csrf_token": hdr(c)["X-CSRF-Token"]})


def test_email_signup_sends_code_and_verifies(mail_app):
    c = mail_app.test_client()
    r = reg(c)
    assert r.status_code == 302 and r.headers["Location"].endswith("/verify")
    assert len(mail_app.sent) == 1 and mail_app.sent[0][0] == "a@example.com"
    assert c.get("/workspace").status_code == 302                                     # not in yet
    assert b"a***@example.com" in c.get("/verify").data                               # address is masked on screen
    r = post_code(c, code_of(mail_app))
    assert r.status_code == 302 and r.headers["Location"].endswith("/workspace")
    assert c.get("/workspace").status_code == 200
    with mail_app.app_context():
        u = adb.user_by_email("a@example.com")
        assert u["approved"] == 1 and u["verify_hash"] is None and u["role"] == "user"   # the code is not kept


def test_wrong_code_and_attempt_limit(mail_app):
    c = mail_app.test_client(); reg(c); good = code_of(mail_app)
    bad = "000000" if good != "000000" else "111111"
    for _ in range(5):
        assert post_code(c, bad).status_code == 400
    r = post_code(c, good)                                                             # even the right code is refused after 5 misses
    assert r.status_code == 400 and b"Too many wrong attempts" in r.data
    assert c.get("/workspace").status_code == 302


def test_expired_code_rejected_and_resend_works(mail_app):
    c = mail_app.test_client(); reg(c)
    with mail_app.app_context(), adb.conn() as db:
        db.execute("UPDATE users SET verify_expires='2000-01-01T00:00:00+00:00'")
    assert b"expired" in post_code(c, code_of(mail_app)).data
    r = c.post("/verify/resend", data={"csrf_token": hdr(c)["X-CSRF-Token"]})
    assert r.status_code == 429                                                        # sent less than a minute ago
    with mail_app.app_context(), adb.conn() as db:
        db.execute("UPDATE users SET verify_sent='2000-01-01T00:00:00+00:00'")
    assert c.post("/verify/resend", data={"csrf_token": hdr(c)["X-CSRF-Token"]}).status_code == 302
    assert len(mail_app.sent) == 2
    assert post_code(c, code_of(mail_app)).status_code == 302


def test_unverified_login_goes_to_verify_page(mail_app):
    c = mail_app.test_client(); reg(c)
    c2 = mail_app.test_client()
    r = login(c2)
    assert r.status_code == 302 and r.headers["Location"].endswith("/verify")
    assert c2.get("/workspace").status_code == 302
    assert login(c2, pw="WrongPass-12345").status_code == 401                          # wrong password never reaches the code page


def test_mail_failure_leaves_no_half_account(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIDS_DIR", str(tmp_path / "ds"))
    from app.auth import mailer

    def boom(*a, **k):
        raise OSError("smtp down")
    monkeypatch.setattr(mailer, "send", boom)

    class C(AuthTestConfig):
        AUTH_DB = str(tmp_path / "auth.db")
        SIGNUP_MODE = "email"
    sec.reset_state(); a = create_app(C); c = a.test_client()
    r = reg(c)
    assert r.status_code == 503 and b"could not send" in r.data
    with a.app_context():
        assert adb.user_by_email("a@example.com") is None                             # so the person can simply try again


def test_admin_can_still_approve_in_email_mode(mail_app):
    make_admin(mail_app)
    c = mail_app.test_client(); reg(c)
    a = mail_app.test_client(); login(a, "admin@example.com")
    assert a.post(f"/api/admin/users/{uid(mail_app, 'a@example.com')}/approve", headers=hdr(a)).status_code == 200
    assert login(c).status_code == 302 and c.get("/workspace").status_code == 200


def test_auto_mode_follows_mail_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIDS_DIR", str(tmp_path / "ds"))
    for v in ("ECX_SMTP_HOST", "ECX_SMTP_PORT", "ECX_SMTP_USER", "ECX_SMTP_PASSWORD", "ECX_SMTP_FROM"):
        monkeypatch.delenv(v, raising=False)

    class C(AuthTestConfig):
        AUTH_DB = str(tmp_path / "auth.db")
        SIGNUP_MODE = "auto"
    sec.reset_state(); a = create_app(C); c = a.test_client()
    r = reg(c)
    assert r.status_code == 302 and r.headers["Location"].endswith("/workspace")      # no mail set up: sign-up works at once
    for v, x in (("ECX_SMTP_HOST", "h"), ("ECX_SMTP_PORT", "587"), ("ECX_SMTP_USER", "u"), ("ECX_SMTP_PASSWORD", "p"), ("ECX_SMTP_FROM", "f@example.com")):
        monkeypatch.setenv(v, x)
    from app.auth import mailer, routes
    monkeypatch.setattr(mailer, "send", lambda *a, **k: None)
    with a.test_request_context():
        assert routes.signup_mode() == "email"


def test_unverified_signup_cannot_squat_someones_email(mail_app):
    squatter = mail_app.test_client(); reg(squatter, pw="Squatter-Pass-99")            # types a victim's address, never verifies
    owner = mail_app.test_client()
    r = reg(owner, pw="Owner-Pass-12345")
    assert r.status_code == 302 and r.headers["Location"].endswith("/verify")           # the real owner is not blocked
    assert post_code(owner, code_of(mail_app)).status_code == 302
    assert login(squatter, pw="Squatter-Pass-99").status_code == 401                    # the squatter's password no longer works
    assert login(squatter, pw="Owner-Pass-12345").status_code == 302


def test_verified_account_cannot_be_taken_over(mail_app):
    a = mail_app.test_client(); reg(a); post_code(a, code_of(mail_app))
    r = reg(mail_app.test_client(), pw="Other-Pass-123456")
    assert r.status_code == 400 and b"already exists" in r.data


def test_signup_form_is_rate_limited_in_email_mode(mail_app):
    c = mail_app.test_client()
    codes = [reg(c, f"user{i}@example.com", f"User Number{i}").status_code for i in range(7)]
    assert codes[:5] == [302] * 5 and codes[5:] == [429, 429]
    assert len(mail_app.sent) == 5                                                       # no e-mails beyond the limit


# ------------------------------------------------------------------ delete user, lean admin menu
def _upload(c, name="x.csv"):
    import io
    import pandas as pd
    b = pd.DataFrame({"a": range(30), "b": [x * 2 for x in range(30)]}).to_csv(index=False).encode()
    return c.post("/api/md/upload", data={"file": (io.BytesIO(b), name), "csrf_token": hdr(c)["X-CSRF-Token"]}, headers=hdr(c), content_type="multipart/form-data")


def test_admin_can_delete_a_user_with_their_files(app):
    make_admin(app)
    u = app.test_client(); reg(u)
    a = app.test_client(); login(a, "admin@example.com")
    i = uid(app, "a@example.com")
    a.post(f"/api/admin/users/{i}/approve", headers=hdr(a)); login(u)
    assert _upload(u).status_code in (200, 201)
    assert a.get("/api/admin/datasets").get_json()["datasets"]
    before = [d for d in a.get("/api/admin/datasets").get_json()["datasets"] if d.get("owner_id") == i]
    assert before
    r = a.delete(f"/api/admin/users/{i}", headers=hdr(a))
    assert r.status_code == 200 and r.get_json()["datasets_removed"] == len(before)
    with app.app_context():
        assert adb.user_by_email("a@example.com") is None
    assert not [d for d in a.get("/api/admin/datasets").get_json()["datasets"] if d.get("owner_id") == i]
    assert login(u).status_code == 401                                                   # the deleted person can no longer sign in
    assert a.delete(f"/api/admin/users/{i}", headers=hdr(a)).status_code == 404


def test_admin_cannot_delete_self_or_last_admin_and_users_cannot_delete(app):
    make_admin(app)
    a = app.test_client(); login(a, "admin@example.com")
    me = uid(app, "admin@example.com")
    r = a.delete(f"/api/admin/users/{me}", headers=hdr(a))
    assert r.status_code == 400 and b"own account" in r.data
    u = app.test_client(); reg(u)
    a.post(f"/api/admin/users/{uid(app, 'a@example.com')}/approve", headers=hdr(a)); login(u)
    assert u.delete(f"/api/admin/users/{me}", headers=hdr(u)).status_code == 403
    assert u.delete(f"/api/admin/users/{uid(app, 'a@example.com')}", headers=hdr(u)).status_code == 403
    with app.app_context():
        adb.create_user("second@example.com", "Second Admin", sec.hash_password(PW), "admin")
    b = app.test_client(); login(b, "second@example.com")
    s = uid(app, "second@example.com")
    assert a.delete(f"/api/admin/users/{s}", headers=hdr(a)).status_code == 200          # another admin can be deleted while one remains
    # the only remaining active administrator is "me" (the self-check already protects it); a second admin who is disabled does not count as one
    with app.app_context():
        adb.create_user("third@example.com", "Third Admin", sec.hash_password(PW), "admin")
    t = uid(app, "third@example.com")
    assert a.post(f"/api/admin/users/{t}", json={"active": False}, headers=hdr(a)).status_code == 200
    assert a.delete(f"/api/admin/users/{t}", headers=hdr(a)).status_code == 200            # a disabled admin can be removed freely


def test_admin_login_lands_on_admin_page_with_lean_menu(app):
    make_admin(app)
    a = app.test_client()
    r = login(a, "admin@example.com")
    assert r.status_code == 302 and r.headers["Location"].endswith("/control-room")
    h = a.get("/control-room").data
    assert b"Admin Panel" in h and b"API keys" in h and b"Settings" in h
    for gone in (b"My Data", b"Order Risk Check", b"What-If Planner", b"More tools", b"Product Recommendations"):
        assert gone not in h                                                              # the administrator does not see the user menu
    u = app.test_client(); reg(u)
    a.post(f"/api/admin/users/{uid(app, 'a@example.com')}/approve", headers=hdr(a)); login(u)
    hu = u.get("/workspace").data
    assert b"My Data" in hu and b"More tools" in hu and b"Admin Panel" not in hu          # normal users: unchanged
