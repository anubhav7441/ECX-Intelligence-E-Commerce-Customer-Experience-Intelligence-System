import io
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["/dashboard", "/data", "/reviews", "/predict", "/explain", "/sellers", "/products", "/geography", "/research", "/recommendations"]


@pytest.mark.parametrize("path", PAGES)
def test_pages_render(client, path):
    r = client.get(path)
    assert r.status_code == 200 and b"ECX-" in r.data


def test_404_page_and_api(client):
    assert client.get("/nope").status_code == 404
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.is_json


def test_health_ok(client):
    j = client.get("/api/health").get_json()
    assert j["status"] == "ok" and j["database"]["tables"]["orders"] == 99441


def test_health_degraded_when_database_missing(tmp_path):
    from app import create_app
    from config import TestConfig

    class BadCfg(TestConfig):
        DATABASE_URL = f"sqlite:///{tmp_path / 'empty.db'}"
    c = create_app(BadCfg).test_client()
    r = c.get("/api/health")
    j = r.get_json()
    assert r.status_code == 200 and j["status"] == "degraded" and j["database"]["ok"] is False
    assert c.get("/data").status_code == 200          # page still renders with a clear message


def test_kpis_match_saved_results(client):
    saved = json.load(open(ROOT / "research" / "results" / "bi_results.json"))["kpis"]
    j = client.get("/api/kpis").get_json()
    assert j["total_orders_all_statuses"] == 99441 == saved["total_orders_all_statuses"]
    assert abs(j["revenue_items_delivered"] - saved["revenue_items_delivered"]) < 0.01


def test_sellers_endpoint_filters_and_limit(client):
    all_ = client.get("/api/sellers").get_json()
    assert len(all_) == 413 and all(r["orders"] >= 50 for r in all_)
    seg = client.get("/api/sellers?segment=PRIORITY PROBLEM&limit=5").get_json()
    assert len(seg) == 5 and all(r["segment"] == "PRIORITY PROBLEM" for r in seg)
    assert client.get("/api/sellers?limit=abc").status_code == 400


def test_categories_states_monthly(client):
    assert len(client.get("/api/states").get_json()) == 24
    assert all(r["orders"] >= 100 for r in client.get("/api/categories").get_json())
    assert len(client.get("/api/monthly").get_json()) > 10


def test_research_and_recommendations(client):
    j = client.get("/api/research").get_json()
    assert any(v["id"] == "H1" for v in j["verdicts"])
    rec = client.get("/api/recommendations").get_json()
    assert len(rec["recommendations"]) >= 5 and all(r["evidence"] for r in rec["recommendations"])


def test_review_analyze_valid_and_invalid(client):
    ok = client.post("/api/review/analyze", json={"text": "Produto não chegou"})
    assert ok.status_code == 200 and "late_delivery" in ok.get_json()["complaint_categories"]
    assert client.post("/api/review/analyze", json={"text": "   "}).status_code == 400
    assert client.post("/api/review/analyze", json={}).status_code == 400
    assert client.post("/api/review/analyze", json={"text": 5}).status_code == 400
    assert client.post("/api/review/analyze", json={"text": "a" * 5000}).status_code == 400
    assert client.post("/api/review/analyze", data="not json", content_type="text/plain").status_code == 400


def test_predict_endpoint(client):
    sellers = client.get("/api/options").get_json()["sellers"]
    body = {"items_price": 100, "freight_value": 15, "estimated_days": 20, "seller_id": sellers[0],
            "customer_state": "MG", "category": "toys", "payment_type": "boleto"}
    r = client.post("/api/predict", json=body)
    assert r.status_code == 200 and "contributions" in r.get_json()
    assert client.post("/api/predict", json={**body, "items_price": -4}).status_code == 400
    assert client.post("/api/predict", json=[1, 2]).status_code == 400
    assert client.post("/api/predict", data="x", content_type="text/plain").status_code == 400


def test_explain_endpoint(client, analytic):
    oid = analytic.order_id.iloc[-1]
    r = client.get(f"/api/explain/{oid}")
    assert r.status_code == 200 and r.get_json()["order_id"] == oid
    assert client.get("/api/explain/zzzz").status_code == 404
    assert client.get("/api/explain/bad-id!").status_code == 400


def test_upload_endpoint(client):
    ok = client.post("/api/upload", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "t.csv")}, content_type="multipart/form-data")
    assert ok.status_code == 200 and ok.get_json()["rows"] == 1
    assert client.post("/api/upload", data={}, content_type="multipart/form-data").status_code == 400
    bad = client.post("/api/upload", data={"file": (io.BytesIO(b"x"), "t.txt")}, content_type="multipart/form-data")
    assert bad.status_code == 400
    empty = client.post("/api/upload", data={"file": (io.BytesIO(b""), "t.csv")}, content_type="multipart/form-data")
    assert empty.status_code == 400


def test_upload_too_large_returns_413(client):
    big = io.BytesIO(b"a\n" + b"1\n" * (11 * 1024 * 1024))
    r = client.post("/api/upload", data={"file": (big, "big.csv")}, content_type="multipart/form-data")
    assert r.status_code == 413 and r.is_json


def test_figure_route_only_serves_png(client):
    assert client.get("/figures/eda_score_distribution.png").status_code == 200
    assert client.get("/figures/../../config.py").status_code == 404
    assert client.get("/figures/x.txt").status_code == 404


def test_xss_text_not_executed_in_api_json(client):
    r = client.post("/api/review/analyze", json={"text": "<script>alert(1)</script> produto quebrado"})
    assert r.status_code == 200 and r.is_json
