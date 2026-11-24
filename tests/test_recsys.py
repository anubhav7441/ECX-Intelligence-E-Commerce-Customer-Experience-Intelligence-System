"""Recommendation Intelligence tests (10 scenarios from the brief). Skipped if recsys data has not been built."""
import numpy as np
import pandas as pd
import pytest
from recsys import service as S
from recsys.evaluate import metrics_for
from recsys.models import RecModel, MODELS

pytestmark = pytest.mark.skipif(not S.available(), reason="run python -m recsys.data first")


@pytest.fixture(scope="module")
def svc():
    return S.get_service()


def _cust(svc, tier, n=1):
    return svc.customers(tier, n)[0]["customer_id"]


def test_established_customer_personalised(svc):
    r = svc.recommend(_cust(svc, "established"), 10)
    assert r["tier"] == "established" and len(r["recommendations"]) == 10
    bought = {h["product_id"] for h in r["history"]}
    assert not bought & {x["product_id"] for x in r["recommendations"]}      # never re-recommend purchases
    assert all(x["reasons"] for x in r["recommendations"])                    # every rec has a reason
    assert "NOT purchase probabilities" in r["score_note"]


def test_limited_history_is_labelled(svc):
    r = svc.recommend(_cust(svc, "limited"), 10)
    assert r["tier"] == "limited" and "fallback" in r["note"].lower()
    assert all(isinstance(x["fallback"], bool) for x in r["recommendations"])


def test_new_customer_popularity_fallback(svc):
    r = svc.recommend_new(None, 5)
    assert r["tier"] == "new" and all(x["fallback"] and x["source"] == "Popularity fallback" for x in r["recommendations"])
    cat = svc.categories()[0]["category"]
    r2 = svc.recommend_new(cat, 5)
    assert all(x["category"] == cat for x in r2["recommendations"])
    assert svc.recommend_new("no_such_category") is None


def test_product_with_enough_data_similar(svc):
    pid = svc.search_products("", 1)[0]["product_id"]
    r = svc.similar(pid, 5)
    assert len(r["similar"]) == 5 and pid not in {x["product_id"] for x in r["similar"]} and not r["warnings"]
    sims = [x["content_similarity"] for x in r["similar"]]; assert sims == sorted(sims, reverse=True)


def test_product_with_little_data_warns(svc):
    j = int(np.where(svc.m.buyers == 1)[0][0])
    r = svc.similar(svc.m.items[j], 5)
    assert r["warnings"] and len(r["similar"]) == 5


def test_frequently_bought_together_uses_real_baskets(svc):
    m = svc.m; row = m.rules.sort_values("pair_count", ascending=False).iloc[0]
    r = svc.together(row.antecedent, 5)
    assert r["together"][0]["level"] == "product"
    # recompute support/confidence/lift for this pair directly from the basket table
    bk = svc.bk[svc.bk.order_purchase_timestamp < m.cutoff]; n = bk.order_id.nunique()
    a = set(bk[bk.product_id == row.antecedent].order_id); b = set(bk[bk.product_id == row.consequent].order_id)
    assert row.pair_count == len(a & b) and abs(row.confidence - len(a & b) / len(a)) < 1e-9
    assert abs(row.lift - (len(a & b) / len(a)) / (len(b) / n)) < 1e-9 and abs(row.support - len(a & b) / n) < 1e-12


def test_together_falls_back_to_category(svc):
    j = int(np.where(svc.m.buyers == 1)[0][0]); r = svc.together(svc.m.items[j], 5)
    assert r["note"] is None or r["together"][0]["level"] == "category"


def test_metrics_hand_computed():
    m = metrics_for({"u": ["a", "b", "c"]}, {"u": {"b", "z"}}, 3)
    assert m["P"][0] == pytest.approx(1 / 3) and m["R"][0] == pytest.approx(0.5)
    assert m["N"][0] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert m["AP"][0] == pytest.approx((0.5 * 1) / 2)


def test_temporal_leakage_free_model_split(svc):
    inter = svc.inter; cut = pd.Timestamp("2018-06-01")
    m = RecModel(inter, svc.bk, svc.prods, cutoff=cut)
    train = inter[inter.first_time < cut]
    assert m.n_users == train.customer_unique_id.nunique() and m.n_items == train.product_id.nunique()
    assert m.X.sum() == len(train)


def test_evaluation_report_is_real_and_consistent():
    e = S.RecService.evaluation()
    if e is None: pytest.skip("evaluation not run")
    t = e["test_results"]; assert set(t) >= {"popularity", "item_cf", "als", "content", "hybrid"}
    assert sum(e["hybrid_weights_chosen_on_validation"].values()) == pytest.approx(1.0)
    assert e["protocol"]["test"]["eligible_customers"] > 0
    for v in t.values(): assert 0 <= v["NDCG@10"] <= 1 and v["R@10_ci95"][0] <= v["R@10"] <= v["R@10_ci95"][1] + 1e-9


def test_all_models_score_all_items(svc):
    h = svc.m.history_idx(_cust(svc, "established")); s = svc.m.scores(h)
    assert set(s) == set(MODELS) and all(len(v) == svc.m.n_items and np.isfinite(v).all() for v in s.values())


def test_api_responses(client, svc):
    cid = _cust(svc, "established"); pid = svc.search_products("", 1)[0]["product_id"]
    assert client.get("/api/rec/status").get_json()["ready"]
    assert client.get(f"/api/rec/customer/{cid}?k=3").status_code == 200
    assert client.get(f"/api/rec/similar/{pid}").status_code == 200 and client.get(f"/api/rec/together/{pid}").status_code == 200
    assert client.get("/api/rec/new").status_code == 200 and client.get("/api/rec/evaluation").status_code in (200, 404)
    assert client.get(f"/api/rec/explain/{cid}/{pid}").status_code == 200
    for bad in ("/api/rec/customer/zzz", f"/api/rec/customer/{cid}?model=nope", "/api/rec/similar/not-an-id"):
        r = client.get(bad); assert r.status_code == 400 and r.is_json
    assert client.get("/api/rec/customer/" + "0" * 32).status_code == 404
    assert client.get(f"/api/rec/customer/{cid}?k=9999").get_json()["k"] == 50


def test_page_and_existing_modules_still_work(client):
    assert client.get("/recsys").status_code == 200
    for p in ("/dashboard", "/predict", "/risk", "/simulate", "/multidata", "/insights"): assert client.get(p).status_code == 200, p
    assert client.get("/api/health").status_code in (200, 503)


def test_recsys_requires_login_when_auth_on():
    from app import create_app
    from config import AuthTestConfig
    c = create_app(AuthTestConfig).test_client()
    assert c.get("/api/rec/status").status_code == 401 and c.get("/recsys", follow_redirects=False).status_code == 302
