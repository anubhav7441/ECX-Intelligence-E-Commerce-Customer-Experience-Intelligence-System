"""Authentication, roles, ownership, demo and admin tests (AUTH_REQUIRED=True)."""
import io
import pandas as pd
import pytest
from app import create_app
from config import AuthTestConfig
from app.auth import security as sec, db as adb
from app.demo import service as demo_service

PW = "Str0ngPass-123"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIDS_DIR", str(tmp_path / "ds"))
    class C(AuthTestConfig):
        AUTH_DB = str(tmp_path / "auth.db")
        RATE_LIMIT_PER_MINUTE = 1000
    sec.reset_state(); demo_service.reset_state()
    a = create_app(C)
    with a.app_context():
        adb.conn().__enter__()                       # create schema
    return a


def token(c):
    c.get("/login")
    with c.session_transaction() as s:
        return s["csrf"]


def reg(c, email="a@example.com", name="Alice Test", pw=PW):
    t = token(c)
    return c.post("/register", data={"email": email, "name": name, "password": pw, "confirm": pw, "csrf_token": t})


def login(c, email="a@example.com", pw=PW):
    t = token(c)
    return c.post("/login", data={"email": email, "password": pw, "csrf_token": t})


def make_admin(app, email="admin@example.com"):
    with app.app_context():
        adb.create_user(email, "Admin User", sec.hash_password(PW), "admin")


def csv_bytes():
    return pd.DataFrame({"a": range(30), "b": [x * 2 for x in range(30)]}).to_csv(index=False).encode()


def hdr(c):
    with c.session_transaction() as s:
        t = s.get("csrf")
    if t is None:
        c.get("/healthz")
        with c.session_transaction() as s:
            t = s["csrf"]
    return {"X-CSRF-Token": t}


# ---- journey 1-3: guest landing, demo, restricted access
def test_landing_public_with_buttons(app):
    r = app.test_client().get("/")
    assert r.status_code == 200
    h = r.get_data(as_text=True)
    assert "Explore demo" in h and "/login" in h and "/register" in h


def test_guest_blocked_from_everything_else(app):
    c = app.test_client()
    for u in ["/dashboard", "/predict", "/multidata", "/command", "/workspace", "/history", "/account"]:
        r = c.get(u)
        assert r.status_code == 302 and "/login" in r.headers["Location"], u
    for u in ["/api/kpis", "/api/md/datasets", "/api/v2/anomalies", "/api/history"]:
        assert c.get(u).status_code == 401, u
    h = c.get("/control-room").data                      # a normal user only ever sees the administrator sign-in form, never the panel
    assert b"Administrator sign-in" in h and b"Users, datasets, activity" not in h
    assert c.get("/api/admin/users").status_code in (401, 403)


def test_demo_page_and_row_cap(app):
    c = app.test_client()
    assert "DEMO MODE" in c.get("/demo").get_data(as_text=True)
    s = c.get("/api/demo/summary").get_json()
    assert s["n_rows"] <= 500 and s["row_limit"] == 500
    assert s["kpis"]["orders"] == s["n_rows"]


def test_demo_cap_follows_admin_setting_and_hard_max(app):
    with app.app_context():
        adb.set_setting("demo_row_limit", "100")
    c = app.test_client()
    assert c.get("/api/demo/summary").get_json()["n_rows"] == 100
    with app.app_context():
        adb.set_setting("demo_row_limit", "999999")
    assert c.get("/api/demo/summary").get_json()["row_limit"] <= demo_service.HARD_MAX_ROWS


def test_demo_prediction_limit_enforced_on_backend(app):
    c = app.test_client()
    oid = c.get("/api/demo/summary").get_json()["orders_preview"][0]["order_id"]
    codes = []
    for _ in range(12):
        r = c.post("/api/demo/predict", json={"order_prefix": oid}, headers=hdr(c))
        codes.append(r.status_code)
    assert codes.count(200) == 10 and codes[-1] == 403
    # clearing the cookie does not reset the counter (per-IP counter)
    c2 = app.test_client()
    assert c2.post("/api/demo/predict", json={"order_prefix": oid}, headers=hdr(c2)).status_code == 403


def test_demo_predict_rejects_orders_outside_sample(app):
    c = app.test_client()
    r = c.post("/api/demo/predict", json={"order_prefix": "00000000"}, headers=hdr(c))
    assert r.status_code == 404


def test_demo_investigation_limit(app):
    c = app.test_client()
    sellers = c.get("/api/demo/sellers").get_json()["sellers"]
    ok = [c.post("/api/demo/investigate", json={"seller_prefix": sellers[0]["full"]}, headers=hdr(c)).status_code for _ in range(4)]
    assert ok == [200, 200, 200, 403]


def test_demo_limits_do_not_apply_to_registered_users(app):
    c = app.test_client(); reg(c)
    r = c.get("/api/md/datasets")
    assert r.status_code == 200
    for _ in range(15):
        assert c.post("/api/md/upload", data={"file": (io.BytesIO(csv_bytes()), "x.csv")}, headers=hdr(c), content_type="multipart/form-data").status_code == 201


# ---- journey 4-5: register, login
def test_register_login_logout_flow(app):
    c = app.test_client()
    r = reg(c)
    assert r.status_code == 302 and "/workspace" in r.headers["Location"]
    assert c.get("/workspace").status_code == 200
    c.post("/logout", data={"csrf_token": hdr(c)["X-CSRF-Token"]})
    assert c.get("/workspace").status_code == 302
    r = login(c)
    assert r.status_code == 302 and c.get("/api/md/datasets").status_code == 200


def test_validation_and_duplicates(app):
    c = app.test_client()
    assert reg(c, email="bad").status_code == 400
    assert reg(c, pw="short1").status_code == 400
    assert reg(c, pw="alllettersnonumber").status_code == 400
    t = token(c)
    assert c.post("/register", data={"email": "q@example.com", "name": "Q Q", "password": PW, "confirm": "different1234", "csrf_token": t}).status_code == 400
    assert reg(c, email="Dup@Example.com").status_code == 302
    c2 = app.test_client()
    r = reg(c2, email="dup@example.com")
    assert r.status_code == 400 and "already exists" in r.get_data(as_text=True)


def test_password_is_hashed_not_plain(app):
    c = app.test_client(); reg(c)
    with app.app_context():
        h = adb.user_by_email("a@example.com")["password_hash"]
    assert PW not in h and h.startswith(("scrypt:", "pbkdf2:"))


def test_wrong_password_generic_message_and_lockout(app):
    c = app.test_client(); reg(c)
    c.post("/logout", data={"csrf_token": hdr(c)["X-CSRF-Token"]})
    for _ in range(5):
        r = login(c, pw="WrongPass-999")
        assert r.status_code == 401 and "Invalid email or password" in r.get_data(as_text=True)
    assert login(c).status_code == 429               # locked even with the right password
    r = login(c, email="nobody@example.com", pw="x")
    assert r.status_code in (401, 429)


def test_open_redirect_blocked(app):
    c = app.test_client(); reg(c, email="r@example.com")
    c.post("/logout", data={"csrf_token": hdr(c)["X-CSRF-Token"]})
    t = token(c)
    r = c.post("/login", data={"email": "r@example.com", "password": PW, "csrf_token": t, "next": "https://evil.example/x"})
    assert r.status_code == 302 and "evil.example" not in r.headers["Location"]


def test_csrf_required_on_post(app):
    c = app.test_client()
    c.get("/login")
    assert c.post("/login", data={"email": "a@example.com", "password": PW}).status_code == 400
    assert c.post("/api/demo/predict", json={}).status_code == 400


def test_session_cookie_flags(app):
    c = app.test_client()
    r = c.get("/login")
    ck = r.headers.get("Set-Cookie", "")
    assert "HttpOnly" in ck and "SameSite=Lax" in ck


def test_security_headers(app):
    r = app.test_client().get("/")
    assert r.headers["X-Frame-Options"] == "DENY" and "Content-Security-Policy" in r.headers


# ---- journey 6-9: upload, analysis, report, history + ownership
def upload(c, name="x.csv"):
    return c.post("/api/md/upload", data={"file": (io.BytesIO(csv_bytes()), name)}, headers=hdr(c), content_type="multipart/form-data")


def test_upload_ownership_isolation(app):
    a, b = app.test_client(), app.test_client()
    reg(a, email="a@example.com"); reg(b, email="b@example.com", name="Bob Test")
    did = upload(a).get_json()["dataset_id"]
    assert did in [d["dataset_id"] for d in a.get("/api/md/datasets").get_json()["datasets"]]
    assert did not in [d["dataset_id"] for d in b.get("/api/md/datasets").get_json()["datasets"]]
    assert b.get(f"/api/md/{did}").status_code == 404
    assert b.delete(f"/api/md/{did}", headers=hdr(b)).status_code == 404
    assert b.post(f"/api/md/{did}/report", headers=hdr(b)).status_code == 404
    assert a.get(f"/api/md/{did}").status_code == 200


def test_delete_own_dataset(app):
    a = app.test_client(); reg(a)
    did = upload(a).get_json()["dataset_id"]
    assert a.delete(f"/api/md/{did}", headers=hdr(a)).status_code == 200
    assert a.get(f"/api/md/{did}").status_code == 404


def test_upload_validation(app):
    a = app.test_client(); reg(a)
    r = a.post("/api/md/upload", data={"file": (io.BytesIO(b"hello"), "x.exe")}, headers=hdr(a), content_type="multipart/form-data")
    assert r.status_code == 400
    assert a.post("/api/md/upload", data={}, headers=hdr(a), content_type="multipart/form-data").status_code == 400


def test_history_records_runs_per_user(app):
    a, b = app.test_client(), app.test_client()
    reg(a, email="a@example.com"); reg(b, email="b@example.com", name="Bob Test")
    did = upload(a).get_json()["dataset_id"]
    sug = a.get(f"/api/md/{did}").get_json()["suggestion"]
    assert "mapping" in sug or sug is not None
    with app.app_context():
        adb.record_run(adb.user_by_email("a@example.com")["id"], did, "x", "analysis", ["trend"], "ok", {})
    assert len(a.get("/api/history").get_json()["runs"]) == 1
    assert len(b.get("/api/history").get_json()["runs"]) == 0


def test_rate_limit_on_expensive_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIDS_DIR", str(tmp_path / "ds"))
    class C(AuthTestConfig):
        AUTH_DB = str(tmp_path / "auth.db")
        RATE_LIMIT_PER_MINUTE = 3
    sec.reset_state()
    a = create_app(C); c = a.test_client(); reg(c)
    codes = [c.post("/api/md/upload", data={}, headers=hdr(c), content_type="multipart/form-data").status_code for _ in range(6)]
    assert 429 in codes


# ---- journey 11-12: admin
def test_regular_user_cannot_use_admin(app):
    c = app.test_client(); reg(c)
    r = c.get("/control-room")
    assert r.status_code == 200 and b"Administrator sign-in" in r.data and b"Users, datasets, activity" not in r.data
    assert c.get("/api/admin/users").status_code == 403
    assert c.post("/api/admin/users/1", json={"role": "admin"}, headers=hdr(c)).status_code == 403


def test_admin_login_and_management(app):
    make_admin(app)
    u = app.test_client(); reg(u)
    c = app.test_client(); assert login(c, "admin@example.com").status_code == 302
    assert c.get("/control-room").status_code == 200
    users = c.get("/api/admin/users").get_json()["users"]
    assert all("password_hash" not in x for x in users) and len(users) == 2
    uid = [x["id"] for x in users if x["email"] == "a@example.com"][0]
    assert c.post(f"/api/admin/users/{uid}", json={"active": False}, headers=hdr(c)).status_code == 200
    assert login(app.test_client()).status_code == 401                 # disabled user cannot log in
    assert c.post(f"/api/admin/users/{uid}", json={"role": "bogus"}, headers=hdr(c)).status_code == 400
    assert c.get("/api/admin/stats").status_code == 200 and c.get("/api/admin/audit").status_code == 200
    assert c.get("/api/admin/datasets").status_code == 200 and c.get("/api/admin/logs").status_code == 200


def test_last_admin_cannot_be_removed(app):
    make_admin(app)
    c = app.test_client(); login(c, "admin@example.com")
    me = [x for x in c.get("/api/admin/users").get_json()["users"] if x["email"] == "admin@example.com"][0]["id"]
    assert c.post(f"/api/admin/users/{me}", json={"role": "user"}, headers=hdr(c)).status_code == 400
    assert c.post(f"/api/admin/users/{me}", json={"active": False}, headers=hdr(c)).status_code == 400


def test_admin_demo_config_validation(app):
    make_admin(app)
    c = app.test_client(); login(c, "admin@example.com")
    assert c.post("/api/admin/demo-config", json={"demo_row_limit": 10}, headers=hdr(c)).status_code == 400
    assert c.post("/api/admin/demo-config", json={"demo_row_limit": "abc"}, headers=hdr(c)).status_code == 400
    r = c.post("/api/admin/demo-config", json={"demo_row_limit": 200, "demo_max_predictions": 2}, headers=hdr(c))
    assert r.status_code == 200 and r.get_json()["rows"] == 200
    g = app.test_client()
    assert g.get("/api/demo/summary").get_json()["n_rows"] == 200


def test_admin_can_see_other_users_datasets(app):
    make_admin(app)
    u = app.test_client(); reg(u)
    did = upload(u).get_json()["dataset_id"]
    c = app.test_client(); login(c, "admin@example.com")
    assert c.get(f"/api/md/{did}").status_code == 200


def test_audit_log_records_events(app):
    c = app.test_client(); reg(c)
    with app.app_context():
        with adb.conn() as k:
            acts = {r["action"] for r in k.execute("SELECT action FROM audit_log")}
    assert "register" in acts


def test_account_settings(app):
    c = app.test_client(); reg(c)
    assert c.post("/api/account/profile", json={"name": "New Name"}, headers=hdr(c)).status_code == 200
    assert c.post("/api/account/profile", json={"name": "x"}, headers=hdr(c)).status_code == 400
    assert c.post("/api/account/password", json={"current": "bad", "new": "Another-Pass-99", "confirm": "Another-Pass-99"}, headers=hdr(c)).status_code == 400
    assert c.post("/api/account/password", json={"current": PW, "new": "Another-Pass-99", "confirm": "Another-Pass-99"}, headers=hdr(c)).status_code == 200
    c.post("/logout", data={"csrf_token": hdr(c)["X-CSRF-Token"]})
    assert login(c, pw="Another-Pass-99").status_code == 302


def test_olist_analysis_still_allowed_for_users(app):
    c = app.test_client(); reg(c)
    assert c.get("/dashboard").status_code == 200
    assert c.get("/api/kpis").status_code == 200
    assert c.get("/risk").status_code == 200


def test_full_journey_upload_map_run_report_history(app):
    """Journeys 6-9 end to end on a (synthetic) retail sample: upload -> mapping -> analysis -> PDF -> history -> download."""
    from pathlib import Path
    samples = sorted((Path(__file__).resolve().parents[1] / "samples").glob("*retail*.xlsx"))
    if not samples:
        pytest.skip("no sample file")
    c = app.test_client(); reg(c)
    r = c.post("/api/md/upload", data={"file": (io.BytesIO(samples[0].read_bytes()), samples[0].name)}, headers=hdr(c), content_type="multipart/form-data")
    assert r.status_code == 201
    did = r.get_json()["dataset_id"]
    d = c.get(f"/api/md/{did}").get_json()
    mp = d["suggestion"].get("mapping") or {}
    assert c.post(f"/api/md/{did}/mapping", json={"mapping": mp}, headers=hdr(c)).status_code == 200
    avail = [x["id"] for x in c.get(f"/api/md/{did}").get_json()["capabilities"] if x.get("available")]
    assert avail
    assert c.post(f"/api/md/{did}/run", json={"analyses": avail[:2]}, headers=hdr(c)).status_code == 200
    assert c.post(f"/api/md/{did}/report", headers=hdr(c)).status_code == 200
    runs = c.get("/api/history").get_json()["runs"]
    assert {x["kind"] for x in runs} == {"analysis", "report"}
    pdf = c.get(f"/api/md/{did}/files/report.pdf")
    assert pdf.status_code == 200 and pdf.data[:4] == b"%PDF"
    other = app.test_client(); reg(other, email="o@example.com", name="Other User")
    assert other.get(f"/api/md/{did}/files/report.pdf").status_code == 404


# ---- plain-English advice (Problem / Why / What to do)
def _cards_ok(cards):
    assert cards
    for c in cards:
        for k in ("title", "problem", "why", "solutions", "evidence", "priority", "confidence"):
            assert c.get(k), (c.get("key"), k)
        assert isinstance(c["solutions"], list) and len(c["solutions"]) >= 1


def test_insights_require_login(app):
    c = app.test_client()
    assert c.get("/insights").status_code == 302
    assert c.get("/api/insights/olist").status_code == 401


def test_olist_insights_have_problem_why_solution(app):
    c = app.test_client(); reg(c)
    j = c.get("/api/insights/olist").get_json()
    _cards_ok(j["cards"])
    assert j["cards"][0]["key"] == "late_vs_reviews"            # headline finding comes first
    assert "62%" in j["cards"][0]["problem"] or "%" in j["cards"][0]["problem"]


def test_dataset_insights_need_mapping_and_respect_ownership(app):
    from pathlib import Path
    sample = Path(__file__).resolve().parents[1] / "samples" / "SYNTHETIC_online_retail_like.xlsx"
    a, b = app.test_client(), app.test_client()
    reg(a); reg(b, email="b@example.com", name="Bob Test")
    did = a.post("/api/md/upload", data={"file": (io.BytesIO(sample.read_bytes()), sample.name)}, headers=hdr(a), content_type="multipart/form-data").get_json()["dataset_id"]
    r = a.get(f"/api/insights/dataset/{did}").get_json()
    assert r["ready"] is False and "mapping" in r["message"].lower()
    mp = {k: (v["column"] if isinstance(v, dict) else v) for k, v in a.get(f"/api/md/{did}").get_json()["suggestion"]["suggested"].items() if v}
    assert a.post(f"/api/md/{did}/mapping", json={"mapping": mp}, headers=hdr(a)).status_code == 200
    r = a.get(f"/api/insights/dataset/{did}").get_json()
    assert r["ready"] and r["cards"]
    _cards_ok(r["cards"])
    assert b.get(f"/api/insights/dataset/{did}").status_code == 404


def test_report_pdf_contains_advice(app):
    from pathlib import Path
    sample = Path(__file__).resolve().parents[1] / "samples" / "SYNTHETIC_online_retail_like.xlsx"
    a = app.test_client(); reg(a)
    did = a.post("/api/md/upload", data={"file": (io.BytesIO(sample.read_bytes()), sample.name)}, headers=hdr(a), content_type="multipart/form-data").get_json()["dataset_id"]
    mp = {k: (v["column"] if isinstance(v, dict) else v) for k, v in a.get(f"/api/md/{did}").get_json()["suggestion"]["suggested"].items() if v}
    a.post(f"/api/md/{did}/mapping", json={"mapping": mp}, headers=hdr(a))
    a.post(f"/api/md/{did}/run", json={"analyses": ["sales_overview", "product_analysis"]}, headers=hdr(a))
    assert a.post(f"/api/md/{did}/report", headers=hdr(a)).status_code == 200
    pdf = a.get(f"/api/md/{did}/files/report.pdf").data
    try:
        from pypdf import PdfReader
    except ImportError:
        pytest.skip("pypdf not installed")
    txt = " ".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "Why it is happening" in txt and "What you can do" in txt


def test_page_fonts_are_bundled_and_pages_render(app):
    c = app.test_client(); reg(c)
    for f in ("sora-latin-wght-normal", "manrope-latin-wght-normal"):
        assert c.get(f"/static/fonts/{f}.woff2").status_code == 200, f
    for u in ["/workspace", "/insights", "/multidata", "/risk", "/simulate", "/assistant", "/history", "/account"]:
        r = c.get(u); assert r.status_code == 200 and b"fonts.css" in r.data, u


def test_home_page_signs_the_person_out_but_other_pages_keep_session(app):
    c = app.test_client(); reg(c)
    assert c.get("/workspace").status_code == 200
    assert c.get("/multidata").status_code == 200            # moving between pages keeps the session (normal Back works)
    assert c.get("/workspace").status_code == 200
    assert c.get("/").status_code == 200                      # opening the home page ends the session
    assert c.get("/workspace").status_code == 302


def test_visuals_and_one_click_report(app):
    import io as _io
    c = app.test_client(); reg(c)
    t = token(c)
    rng = __import__("numpy").random.default_rng(0); n = 300
    df = pd.DataFrame({"order_date": pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 330, n), unit="D"), "product": rng.choice(list("abcdefgh"), n),
                       "quantity": rng.integers(1, 5, n), "unit_price": rng.uniform(1, 50, n), "region": rng.choice(["N", "S", "E"], n), "rating": rng.integers(1, 6, n)})
    b = _io.BytesIO(); df.to_csv(b, index=False); b.seek(0)
    did = c.post("/api/md/upload", data={"file": (b, "s.csv"), "csrf_token": t}, headers={"X-CSRF-Token": t}).get_json()["dataset_id"]
    assert c.post(f"/api/md/{did}/download-report", headers={"X-CSRF-Token": t}).status_code == 400     # mapping not confirmed yet
    sug = c.get(f"/api/md/{did}").get_json()["suggestion"]["suggested"]
    assert c.post(f"/api/md/{did}/mapping", json={"mapping": {k: v["column"] for k, v in sug.items()}}, headers={"X-CSRF-Token": t}).status_code == 200
    v = c.get(f"/api/md/{did}/visuals").get_json()
    assert len(v["charts"]) >= 3 and all(x["explain"] and x["why"] for x in v["charts"])
    assert c.get(f"/api/md/{did}/files/{v['charts'][0]['file']}").status_code == 200
    r = c.post(f"/api/md/{did}/download-report", headers={"X-CSRF-Token": t}).get_json()
    assert r["ok"] and c.get(r["url"]).data[:4] == b"%PDF"
