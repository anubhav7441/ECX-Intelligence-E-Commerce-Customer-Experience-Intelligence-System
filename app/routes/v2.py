"""ECX-Intelligence 2.0 routes (additive blueprint; v1 routes unchanged)."""
from flask import Blueprint, current_app, jsonify, render_template, request
from ecx2 import store, tools
from ecx2.common import V2_DATA, clean, load_json

v2 = Blueprint("v2", __name__)


def db():
    return current_app.config.get("APP_DB")


def err(msg, code=400):
    return jsonify(error=msg if isinstance(msg, str) else "; ".join(msg)), code


def body():
    b = request.get_json(silent=True)
    return b if isinstance(b, dict) else None


# ------------------------------------------------------------------ pages
@v2.route("/command")
def command():
    try:
        k = tools.platform_summary()
    except FileNotFoundError:
        k = None
    an = load_json("anomaly_summary.json") or {}
    return render_template("command.html", active="command", k=k, an=an, cnt=store.counts(db()), risk=load_json("risk_results.json"))


@v2.route("/risk")
def risk_page():
    from ecx2.risk.predict import RANGES
    ref = load_json("risk_reference.json") or {}
    return render_template("risk.html", active="risk", res=load_json("risk_results.json"), states=sorted(ref.get("state", {})),
                           cats=sorted(ref.get("category", {})), ranges=RANGES)


@v2.route("/anomalies")
def anomalies_page():
    return render_template("anomalies.html", active="anomalies", summ=load_json("anomaly_summary.json"))


@v2.route("/investigate")
def investigate_page():
    return render_template("investigate.html", active="investigate")


@v2.route("/assistant")
def assistant_page():
    from ecx2.assistant.core import SUPPORTED
    return render_template("assistant.html", active="assistant", supported=SUPPORTED, exposures=sorted(tools.EXPOSURES), outcomes=sorted(tools.OUTCOMES))


@v2.route("/simulate")
def simulate_page():
    from ecx2.sim.engine import PARAMS
    return render_template("simulate.html", active="simulate", params=PARAMS)


@v2.route("/actions")
def actions_page():
    return render_template("actions.html", active="actions", roles=store.ROLES, priorities=store.PRIORITIES)


@v2.route("/voice")
def voice_page():
    return render_template("voice.html", active="voice", res=load_json("nlp_v2_results.json"))


@v2.route("/evaluation")
def evaluation_page():
    from app.services import data_service as ds
    return render_template("evaluation.html", active="evaluation", risk=load_json("risk_results.json"), nlp=load_json("nlp_v2_results.json"),
                           agent=load_json("agent_eval.json"), an=load_json("anomaly_summary.json"), ml=ds.clean(ds.ml()),
                           registry=load_json("model_registry.json"), exp=load_json("experiments.json"))


# ------------------------------------------------------------------ API
@v2.get("/api/v2/health")
def health():
    from ecx2.common import V2_MODELS
    return jsonify(risk_models=(V2_MODELS / "delivery_risk_A0.joblib").exists(), anomalies=(V2_DATA / "anomalies.csv").exists(),
                   nlp_v2=load_json("nlp_v2_results.json") is not None, agent_eval=load_json("agent_eval.json") is not None, store=store.counts(db()))


@v2.post("/api/v2/risk")
def api_risk():
    b = body()
    if b is None:
        return err("Send a JSON object.")
    from ecx2.risk.predict import predict_risk
    r = predict_risk(b)
    return (jsonify(clean(r)), 200) if r.get("ok") else err(r["errors"])


@v2.get("/api/v2/risk/sellers")
def api_risk_sellers():
    import pandas as pd
    p = V2_DATA / "risk_scores_test.parquet"
    if not p.exists():
        return err("Risk scores missing - run python -m ecx2.risk.train", 404)
    r = pd.read_parquet(p)
    g = r.groupby("primary_seller_id").agg(n=("order_id", "size"), mean_risk=("risk_A0", "mean"), late_rate=("is_late", "mean")).query("n>=20")
    g = g.sort_values("mean_risk", ascending=False).head(15).reset_index()
    return jsonify(clean({"source": "test-period orders (2018-06 to 2018-08), checkout-stage model", "rows": g.round(4).to_dict("records")}))


@v2.get("/api/v2/anomalies")
def api_anomalies():
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        return err("limit must be an integer")
    r = tools.anomalies(request.args.get("entity") or None, request.args.get("detector") or None, max(1, min(limit, 500)))
    return (jsonify(clean(r)), 200) if r["ok"] else err(r["error"], 404)


@v2.post("/api/v2/voice")
def api_voice():
    b = body() or {}
    t = b.get("text")
    if not isinstance(t, str) or not t.strip():
        return err("Provide a non-empty 'text' string.")
    if len(t) > current_app.config.get("MAX_REVIEW_CHARS", 2000):
        return err("Text is too long.")
    from app.services.nlp_service import sentiment_model
    from ecx2.nlp.issues import analyse_v2
    return jsonify(clean(analyse_v2(t, sentiment_model())))


@v2.post("/api/v2/investigate")
def api_investigate():
    b = body()
    if b is None:
        return err("Send a JSON object.")
    from ecx2.agents.system import Supervisor
    out = Supervisor(db()).investigate(b.get("target_type", "seller"), str(b.get("target_id", "")), b.get("question"))
    if not out["ok"]:
        return err(out["error"])
    return jsonify(clean(out["report"].model_dump()))


@v2.post("/api/v2/assistant")
def api_assistant():
    b = body()
    if b is None:
        return err("Send a JSON object.")
    from ecx2.assistant.core import answer
    r = answer(b.get("question"), db=db())
    return (jsonify(clean(r)), 200) if r.get("ok") else err(r["error"])


@v2.post("/api/v2/research")
def api_research():
    b = body()
    if b is None:
        return err("Send a JSON object.")
    r = tools.test_association(b.get("exposure"), b.get("outcome"))
    return (jsonify(clean(r)), 200) if r["ok"] else err(r["error"])


@v2.post("/api/v2/simulate")
def api_simulate():
    b = body()
    if b is None:
        return err("Send a JSON object.")
    from ecx2.sim.engine import simulate
    f = {k: b.pop(k, None) or None for k in ("seller_id", "state", "category")}
    r = simulate(b, **f)
    return (jsonify(clean(r)), 200) if r["ok"] else err(r["errors"])


@v2.get("/api/v2/actions")
def api_actions():
    st = request.args.get("status")
    if st and st not in store.STATUSES:
        return err("unknown status")
    return jsonify(clean({"actions": store.list_actions(st, db=db()), "counts": store.counts(db())}))


@v2.post("/api/v2/actions")
def api_action_create():
    b = body()
    if b is None:
        return err("Send a JSON object.")
    try:
        ev = b.get("evidence_ids") or []
        if not ev:   # manual actions must still cite evidence: create a ledger entry from the reviewer's rationale
            ev = [store.add_evidence(b.get("reason") or "Manual action rationale", "Action Center (manual)", None, None, None, None, None, "Low",
                                     "Entered by a human reviewer; not machine-verified.", b.get("title", ""), None, db=db())]
        aid = store.propose_action(b.get("title"), b.get("description", ""), b.get("priority", "Medium"), b.get("reason", ""), ev,
                                   b.get("owner_role", "Data Analyst"), source=b.get("actor") or "manual", db=db())
    except ValueError as e:
        return err(str(e))
    return jsonify(store.get_action(aid, db=db())), 201


@v2.post("/api/v2/actions/<action_id>/status")
def api_action_status(action_id):
    b = body()
    if b is None:
        return err("Send a JSON object.")
    try:
        return jsonify(store.set_status(action_id, b.get("status"), b.get("actor", ""), b.get("note", ""), db=db()))
    except KeyError:
        return err("action not found", 404)
    except ValueError as e:
        return err(str(e))


@v2.get("/api/v2/evidence")
def api_evidence():
    return jsonify(clean({"evidence": store.list_evidence(request.args.get("entity") or None, request.args.get("run_id") or None, 200, db=db())}))
