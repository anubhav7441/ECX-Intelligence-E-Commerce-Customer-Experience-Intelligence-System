"""Tests for ECX-Intelligence 2.0 modules (risk, anomaly, NLP v2, store, agents, assistant, simulation, API)."""
import numpy as np
import pandas as pd
import pytest
from ecx2 import store, tools

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture()
def db(tmp_path):
    return str(tmp_path / "t.db")


@pytest.fixture(scope="module")
def flagged_seller():
    return tools.top_sellers("dissatisfied_rate", 1)["rows"][0]["seller_id"]


# ---------------------------------------------------------------- risk
def test_risk_features_have_no_outcome_leakage():
    from ecx2.risk import train
    forbidden = {"delivery_days", "delay_days", "carrier_transit_days", "is_late", "review_score", "dissatisfied"}
    for _, (num, cat) in train.STAGES.items():
        assert not forbidden & set(num + cat)


def test_late_history_uses_only_earlier_deliveries():
    from ecx2.risk.train import add_late_history
    t0 = pd.Timestamp("2018-01-01")
    d = pd.DataFrame({"primary_seller_id": ["s"] * 3, "primary_category": ["c"] * 3, "customer_state": ["SP"] * 3,
                      "order_purchase_timestamp": [t0, t0 + pd.Timedelta(days=1), t0 + pd.Timedelta(days=30)],
                      "order_delivered_customer_date": [t0 + pd.Timedelta(days=10), t0 + pd.Timedelta(days=40), t0 + pd.Timedelta(days=50)],
                      "is_late": [1, 0, 0]})
    r = add_late_history(d, prior=0.1)
    assert r.seller_late_hist_n.tolist() == [0, 0, 1]      # order 3 sees only order 1 (delivered day 10 < purchase day 30)
    assert r.seller_late_hist.iloc[0] == pytest.approx(0.1)


def test_predict_risk_stages_and_validation():
    from ecx2.risk.predict import predict_risk
    a = predict_risk({"items_price": 100, "freight_value": 20, "estimated_days": 20, "customer_state": "SP", "category": "bed_bath_table"})
    assert a["ok"] and a["stage"].startswith("A0") and a["band"] in ("Low", "Medium", "High")
    b = predict_risk({"items_price": 100, "freight_value": 20, "estimated_days": 20, "approval_hours": 3, "seller_handling_days": 2})
    assert b["ok"] and b["stage"].startswith("A1")
    bad = predict_risk({"items_price": -5, "estimated_days": "abc"})
    assert not bad["ok"] and len(bad["errors"]) == 2


def test_risk_bands_are_ordered_on_test():
    from ecx2.common import load_json
    for s, v in load_json("risk_results.json")["selected"].items():
        o = v["bands"]["observed_test_late_rate"]
        assert o["Low"]["late_rate"] < o["High"]["late_rate"], s


# ---------------------------------------------------------------- anomaly
def test_binomial_detector_recovers_planted_anomaly():
    from ecx2.anomaly.detect import _binom_flags
    rng = np.random.default_rng(0)
    rows = [(f"s{i}", rng.random(200) < 0.07) for i in range(40)]
    rows.append(("planted", rng.random(200) < 0.30))
    df = pd.concat([pd.DataFrame({"sid": s, "late": v.astype(int)}) for s, v in rows])
    f = _binom_flags(df.groupby("sid"), "late", 0.07, "t")
    flagged = set(f[(f.p_adj < 0.05) & (f.observed > f.expected * 1.25)].entity)
    assert "planted" in flagged and len(flagged) <= 4      # planted found, few false positives (FDR)


def test_anomaly_output_is_non_accusatory_and_complete():
    a = tools.anomalies(None, None, 300)
    assert a["ok"] and a["n"] > 0
    for r in a["rows"][:50]:
        assert r["why_flagged"] and r["recommended_investigation"] and "not proof" in r["caveat"].lower()
        assert not any(w in (r["why_flagged"] + r["possible_explanation"]).lower() for w in ("fraud", "fraudulent", "scam"))


# ---------------------------------------------------------------- NLP v2
def test_nlp_v2_examples():
    from ecx2.nlp.issues import analyse_v2
    r = analyse_v2("Produto chegou quebrado e a entrega atrasou muito, péssimo!")
    assert r["main_issue"] == "damaged_product" and r["language"] == "pt" and r["keywords"] and 0 < r["confidence"] <= 0.95
    assert "late_delivery" in r["complaint_categories"]
    praise = analyse_v2("Ótima qualidade, recomendo")
    assert praise["main_issue"] == "other"                  # v1 matched 'ma qualidade' inside 'otima qualidade'


def test_nlp_v2_edge_cases():
    from ecx2.nlp.issues import analyse_v2
    assert analyse_v2("")["ok"] is False and analyse_v2("   ")["ok"] is False and analyse_v2(None)["ok"] is False
    en = analyse_v2("The product never arrived and it was terrible")
    assert en["language"] == "en" and en["confidence"] < 0.4 and any("English" in c for c in en["caveats"])


# ---------------------------------------------------------------- store / ledger / action workflow
def test_action_workflow_and_human_approval(db):
    e = store.add_evidence("finding", "test", db=db)
    a = store.propose_action("Do X", "d", "High", "because", [e], "Operations Manager", db=db)
    assert store.get_action(a, db=db)["status"] == "proposed"
    with pytest.raises(ValueError):
        store.set_status(a, "approved", "", db=db)             # no named reviewer
    with pytest.raises(ValueError):
        store.set_status(a, "completed", "Dr A", "x", db=db)    # cannot skip approval
    store.set_status(a, "approved", "Dr A", db=db)
    with pytest.raises(ValueError):
        store.set_status(a, "completed", "Dr A", "", db=db)     # completion note required
    done = store.set_status(a, "completed", "Dr A", "fixed carrier contract", db=db)
    assert done["status"] == "completed" and done["completion_notes"] == "fixed carrier contract"
    assert [h["to_status"] for h in done["history"]] == ["proposed", "approved", "completed"]
    with pytest.raises(ValueError):
        store.set_status(a, "rejected", "Dr A", "late", db=db)  # terminal state
    with pytest.raises(KeyError):
        store.set_status("ACT-99999", "approved", "x", db=db)


def test_action_requires_evidence_and_valid_priority(db):
    with pytest.raises(ValueError):
        store.propose_action("t", "d", "High", "r", [], "role", db=db)
    e = store.add_evidence("f", "m", db=db)
    with pytest.raises(ValueError):
        store.propose_action("t", "d", "Urgent!!", "r", [e], "role", db=db)


# ---------------------------------------------------------------- agents
def test_seller_investigation_is_structured_and_linked(db, flagged_seller):
    from ecx2.agents.system import Supervisor
    out = Supervisor(db).investigate("seller", flagged_seller)
    assert out["ok"]
    rep = out["report"]
    assert [r.agent for r in rep.reports] == ["Seller Investigation Agent", "Delivery Risk Agent", "Customer Review Agent",
                                              "Research Validation Agent", "Action Planning Agent"]
    ledger = {e["finding_id"] for e in store.list_evidence(run_id=rep.run_id, db=db)}
    for r in rep.reports:
        assert r.task and r.recommended_next_step
        for f in r.findings:
            assert f.evidence_id in ledger
    for a in rep.actions:
        assert store.get_action(a.action_id, db=db)["status"] == "proposed" and set(a.evidence_ids) <= ledger
    assert rep.human_review_required


def test_unknown_inputs_are_handled(db):
    from ecx2.agents.system import Supervisor
    assert not Supervisor(db).investigate("seller", "zzzz")["ok"]
    assert not Supervisor(db).investigate("seller", "ffffffffffffffff")["ok"]
    assert not Supervisor(db).investigate("platform_month", "1999-01")["ok"]
    assert not Supervisor(db).investigate("galaxy", "x")["ok"]


def test_agent_survives_tool_failure(db, flagged_seller, monkeypatch):
    from ecx2.agents.system import Supervisor
    def boom(*a, **k):
        raise RuntimeError("tool exploded")
    monkeypatch.setattr(tools, "seller_complaints", boom)
    out = Supervisor(db).investigate("seller", flagged_seller)
    assert out["ok"]
    rv = next(r for r in out["report"].reports if r.agent == "Customer Review Agent")
    assert rv.status == "partial" or rv.status == "failed"
    assert any("tool exploded" in e for e in rv.errors)
    assert len(out["report"].reports) == 5                    # other agents still ran


# ---------------------------------------------------------------- assistant
def test_assistant_grounded_and_refuses(db):
    from ecx2.assistant.core import answer
    k = answer("What is the late rate?", db=db)
    assert k["grounded"] and "6.7%" in k["answer"] and k["citations"]
    r = answer("Will the stock market rise tomorrow?", db=db)
    assert r["refused"] and not r["grounded"] and "Supported question types" in r["answer"]
    t = answer("Does late delivery affect dissatisfaction?", db=db)
    assert t["route"] == "association" and "odds ratio" in t["answer"] and "causation" in t["answer"]
    assert answer("Tell me about seller ffffff", db=db)["refused"]
    assert not answer("", db=db)["ok"] and not answer("x" * 600, db=db)["ok"]


def test_assistant_works_without_llm_and_guards_llm_numbers(db):
    from ecx2.assistant.core import answer
    base = answer("What is the late rate?", db=db, llm=lambda p: (_ for _ in ()).throw(ConnectionError("no server")))
    assert base["ok"] and not base["used_llm"]                 # LLM down -> deterministic answer
    lie = answer("What is the late rate?", db=db, llm=lambda p: "The late rate is 99.9% which is huge")
    assert not lie["used_llm"] and "99.9" not in lie["answer"]  # invented number rejected
    ok = answer("What is the late rate?", db=db, llm=lambda p: "About 6.7% of orders are late.")
    assert ok["used_llm"]


def test_statistical_tests_are_python_computed():
    r = tools.test_association("late delivery", "dissatisfaction")
    d = tools._d()
    assert r["rate_exposed"] == pytest.approx(d[d.is_late == 1].dissatisfied.mean())
    assert r["odds_ratio"] > 10 and r["verdict"].startswith("Supported")
    assert not tools.test_association("nonsense", "dissatisfaction")["ok"]
    assert not tools.test_association("late delivery", "late delivery")["ok"]


# ---------------------------------------------------------------- simulation
def test_simulation_validation_and_direction():
    from ecx2.sim.engine import simulate
    for bad in ({}, {"late_orders_fixed_pct": 500}, {"delivery_days_reduction": "x"}, {"bogus": 1, "delivery_days_reduction": 1}):
        assert not simulate(bad)["ok"]
    r = simulate({"late_orders_fixed_pct": 100})
    assert r["ok"] and "HYPOTHETICAL" in r["label"]
    assert r["simulated"]["late_rate"] < r["baseline"]["late_rate"]
    assert r["change"]["expected_dissatisfied_pp"] < 0
    lo, hi = r["change"]["expected_dissatisfied_ci95_pp"]; assert lo <= r["change"]["expected_dissatisfied_pp"] <= hi
    assert not simulate({"delivery_days_reduction": 1}, seller_id="ffffffffff")["ok"]


def test_simulation_no_change_for_zero_effect_freight():
    from ecx2.sim.engine import simulate
    r = simulate({"freight_discount_pct": 1})
    assert abs(r["change"]["late_rate_pp"]) < 1e-9


# ---------------------------------------------------------------- API / pages
PAGES = ["/command", "/risk", "/voice", "/anomalies", "/investigate", "/assistant", "/simulate", "/actions", "/evaluation"]


@pytest.mark.parametrize("url", PAGES)
def test_v2_pages_render(client, url):
    assert client.get(url).status_code == 200


def test_v1_pages_still_work(client):
    for u in ["/dashboard", "/data", "/reviews", "/predict", "/explain", "/sellers", "/products", "/geography", "/research", "/recommendations"]:
        assert client.get(u).status_code == 200
    assert client.get("/api/health").status_code == 200


def test_v2_api(client):
    assert client.post("/api/v2/risk", json={"items_price": 90, "freight_value": 15, "estimated_days": 20}).json["band"]
    assert client.post("/api/v2/risk", json={"items_price": -1}).status_code == 400
    assert client.post("/api/v2/risk", data="notjson").status_code == 400
    assert client.post("/api/v2/voice", json={"text": ""}).status_code == 400
    assert client.post("/api/v2/voice", json={"text": "entrega atrasou"}).json["main_issue"] == "late_delivery"
    assert client.post("/api/v2/assistant", json={"question": "what is the late rate"}).json["grounded"]
    assert client.post("/api/v2/simulate", json={"delivery_days_reduction": 2}).json["label"].startswith("HYPOTHETICAL")
    assert client.post("/api/v2/simulate", json={}).status_code == 400
    assert client.post("/api/v2/research", json={"exposure": "late delivery", "outcome": "dissatisfaction"}).status_code == 200
    assert client.post("/api/v2/research", json={"exposure": "x", "outcome": "y"}).status_code == 400
    assert client.get("/api/v2/anomalies?limit=abc").status_code == 400
    assert client.get("/api/v2/anomalies?limit=5").json["rows"]
    assert client.get("/api/v2/actions?status=bogus").status_code == 400


def test_api_action_approval_flow(client):
    r = client.post("/api/v2/actions", json={"title": "Check carrier X", "reason": "late rate high in region", "priority": "High"})
    assert r.status_code == 201
    aid = r.json["action_id"]
    assert client.post(f"/api/v2/actions/{aid}/status", json={"status": "approved", "actor": ""}).status_code == 400
    assert client.post(f"/api/v2/actions/{aid}/status", json={"status": "approved", "actor": "Reviewer A"}).json["status"] == "approved"
    assert client.post("/api/v2/actions/NOPE/status", json={"status": "approved", "actor": "A"}).status_code == 404
    assert client.post("/api/v2/actions", json={"reason": "x"}).status_code == 400
    assert client.get("/api/v2/evidence").json["evidence"]


def test_api_investigate(client, flagged_seller):
    r = client.post("/api/v2/investigate", json={"target_type": "seller", "target_id": flagged_seller})
    assert r.status_code == 200 and r.json["reports"] and r.json["human_review_required"]
    assert client.post("/api/v2/investigate", json={"target_type": "seller", "target_id": "nothex!"}).status_code == 400
