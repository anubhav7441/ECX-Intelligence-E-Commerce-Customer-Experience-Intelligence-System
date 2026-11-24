"""The 'Data City' look (design G) on the public pages: assets are served, markup is in place, real figures are shown, security stays strict."""
import pytest
from app import create_app
from app.auth import security as sec, db as adb
from app.demo import service as demo_service
from config import AuthTestConfig


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIDS_DIR", str(tmp_path / "ds"))

    class C(AuthTestConfig):
        AUTH_DB = str(tmp_path / "auth.db")
        RATE_LIMIT_PER_MINUTE = 1000
    sec.reset_state(); demo_service.reset_state()
    a = create_app(C)
    with a.app_context():
        adb.conn().__enter__()
    return a.test_client()


def test_landing_has_iso_chart_with_real_figures(client):
    h = client.get("/").get_data(as_text=True)
    assert 'class="iso ' in h and h.count('class="bx"') == 6 and "css/city.css" in h
    assert "80.2%" in h and "62.4%" in h and "9.3%" in h and "95,824" in h      # figures come from the Olist data
    assert "Explore demo" in h and "Upload my file" in h and "/register" in h


@pytest.mark.parametrize("path", ["/login", "/register"])
def test_auth_pages_have_mini_chart_and_keep_form(client, path):
    h = client.get(path).get_data(as_text=True)
    assert "iso mini" in h and "css/city.css" in h
    assert 'name="csrf_token"' in h and 'name="email"' in h and 'name="password"' in h


def test_assets_are_served(client):
    for p in ("/static/css/city.css", "/static/fonts/chakra-petch-latin-700-normal.woff2", "/static/fonts/inter-latin-wght-normal.woff2"):
        assert client.get(p).status_code == 200, p


def test_no_webgl_or_outside_hosts(client):
    h = client.get("/").get_data(as_text=True)
    assert "three.module" not in h and "<canvas" not in h and "fonts.googleapis" not in h
    csp = client.get("/").headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp and "http" not in csp.replace("'self'", "")


def test_app_pages_do_not_load_the_landing_style(client):
    assert "city.css" not in client.get("/demo").get_data(as_text=True)


def test_gskin_is_loaded_on_app_and_public_pages(client):
    assert "css/gskin.css" in client.get("/").get_data(as_text=True)
    assert "css/gskin.css" in client.get("/login").get_data(as_text=True)
    assert "css/gskin.css" in client.get("/demo").get_data(as_text=True)
    assert client.get("/static/css/gskin.css").status_code == 200
    assert client.get("/static/fonts/chakra-petch-latin-600-normal.woff2").status_code == 200


def test_logged_in_pages_use_gskin(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIDS_DIR", str(tmp_path / "ds2"))

    class C(AuthTestConfig):
        AUTH_DB = str(tmp_path / "auth2.db")
        RATE_LIMIT_PER_MINUTE = 1000
    sec.reset_state(); demo_service.reset_state()
    a = create_app(C)
    with a.app_context():
        adb.conn().__enter__()
    c = a.test_client()
    c.get("/register")
    with c.session_transaction() as s:
        t = s["csrf"]
    r = c.post("/register", data={"name": "Skin Tester", "email": "skin.tester@example.com", "password": "Str0ngPass-123", "confirm": "Str0ngPass-123", "csrf_token": t})
    assert r.status_code in (200, 302)
    h = c.get("/workspace").get_data(as_text=True)
    assert "css/gskin.css" in h and "sidebar" in h and "Operations Center" in h and "More tools" in h


def test_demo_page_does_not_link_to_itself_in_the_top_bar():
    """On the demo page the top bar must not offer 'Explore demo' (you are already there); it offers Sign in instead."""
    from app import create_app
    from config import TestConfig
    h = create_app(TestConfig).test_client().get("/demo").get_data(as_text=True)
    top = h[h.index('<header class="pnav">'):h.index("</header>")]
    assert "Explore demo" not in top and "Sign in" in top and "Create free account" in top
