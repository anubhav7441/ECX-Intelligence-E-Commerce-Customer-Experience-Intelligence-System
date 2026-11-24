import json
from pathlib import Path
import pytest
from app.services import predict_service as ps

ROOT = Path(__file__).resolve().parents[1]
BASE = None


def base():
    o = ps.options()
    return {"items_price": 120, "freight_value": 20, "estimated_days": 24, "seller_id": o["sellers"][0],
            "customer_state": "SP", "category": "bed_bath_table", "payment_type": "credit_card"}


def test_model_files_exist_and_load():
    from ml.explainability.explain import load_artifact
    for tag in ["S0", "S0_R", "S1", "S1_R"]:
        art = load_artifact(tag)
        assert art["model"] is not None and art["threshold"] > 0


def test_prediction_returns_valid_structure():
    r = ps.predict(base())
    assert 0 <= r["risk_score"] <= 1 and r["mode"] == "checkout" and len(r["contributions"]) > 0
    assert 0 <= r["observed_rate_in_band"] <= 1


def test_later_delivery_raises_risk():
    fast = ps.predict({**base(), "delivery_days": 8})["risk_score"]
    slow = ps.predict({**base(), "delivery_days": 45})["risk_score"]
    assert ps.predict({**base(), "delivery_days": 8})["mode"] == "delivery"
    assert slow > fast


@pytest.mark.parametrize("bad", [{}, {"items_price": "abc"}, {"items_price": -1}, {"items_price": 1e9}])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(ps.InputError):
        ps.predict({**base(), **bad} if bad else {})


def test_unknown_category_state_seller_rejected():
    for k, v in [("category", "nope"), ("customer_state", "ZZ"), ("seller_id", "x" * 32), ("payment_type", "bitcoin")]:
        with pytest.raises(ps.InputError):
            ps.predict({**base(), k: v})


def test_missing_optional_fields_are_ok():
    r = ps.predict(base())  # no weight / distance / delivery
    assert r["probability_dissatisfied"] == r["probability_dissatisfied"]


def test_results_file_sane():
    R = json.load(open(ROOT / "ml" / "prediction" / "results" / "ml_results.json"))
    for fs, d in R["results"].items():
        for m, r in d.items():
            assert 0.5 < r["test"]["roc_auc"] < 1.0, (fs, m)
    assert R["selected"]["S1+R delivery-time + history"]["test"]["roc_auc"] > R["selected"]["S0 checkout-time"]["test"]["roc_auc"]


def test_explain_unknown_order():
    with pytest.raises(ps.InputError):
        ps.explain_order("does-not-exist")


def test_sentiment_model_loads_and_predicts():
    from app.services.nlp_service import sentiment_model
    m = sentiment_model()
    assert m.predict(["pessimo nao recomendo nao chegou"])[0] == "negative"
    assert m.predict(["excelente produto recomendo muito bom"])[0] == "positive"
