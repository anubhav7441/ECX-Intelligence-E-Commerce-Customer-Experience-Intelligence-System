"""REST API (JSON). All data comes from verified, pre-computed results or the trained models."""
from flask import Blueprint, current_app, jsonify, request
from app.services import data_service as ds
from app.services import predict_service as ps
from app.services.upload_service import UploadError, validate_csv

api = Blueprint("api", __name__)


def err(msg, code=400):
    return jsonify(error=msg), code


@api.get("/health")
def health():
    db = ds.db_status()
    return jsonify(status="ok" if db["ok"] else "degraded", database=db)


@api.get("/kpis")
def kpis():
    return jsonify(ds.clean(ds.bi()["kpis"]))


@api.get("/monthly")
def monthly():
    return jsonify(ds.monthly())


@api.get("/sellers")
def sellers():
    seg = request.args.get("segment")
    try:
        limit = int(request.args.get("limit", 0)) or None
    except ValueError:
        return err("limit must be an integer")
    return jsonify(ds.sellers(seg, limit))


@api.get("/categories")
def categories():
    return jsonify(ds.categories(request.args.get("segment")))


@api.get("/states")
def states():
    return jsonify(ds.states())


@api.get("/research")
def research():
    s = ds.stats()
    return jsonify(ds.clean({"verdicts": s["verdicts"], "H1": s["H1_late_vs_ontime"], "descriptive": s["descriptive"]}))


@api.get("/recommendations")
def recommendations():
    b = ds.bi()
    return jsonify(ds.clean({"recommendations": b["recommendations"], "findings": b["findings"]}))


@api.get("/options")
def options():
    return jsonify(ps.options())


@api.post("/review/analyze")
def review_analyze():
    from ml.sentiment_analysis.review_nlp import analyse
    from app.services.nlp_service import sentiment_model
    body = request.get_json(silent=True) or {}
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return err("Provide a non-empty 'text' string.")
    if len(text) > current_app.config["MAX_REVIEW_CHARS"]:
        return err(f"Text is longer than {current_app.config['MAX_REVIEW_CHARS']} characters.")
    r = analyse(text)
    probs = sentiment_model().predict_proba([r["norm_text"]])[0]
    classes = list(sentiment_model().classes_)
    r["model_sentiment"] = {c: round(float(p), 4) for c, p in zip(classes, probs)}
    r["model_sentiment_label"] = classes[int(probs.argmax())]
    r["aspects"] = {k: v for k, v in r["aspects"].items() if v}
    r.pop("norm_text")
    return jsonify(r)


@api.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return err("Send a JSON object.")
    try:
        return jsonify(ds.clean(ps.predict(body)))
    except ps.InputError as e:
        return err(str(e))


@api.get("/explain/<order_id>")
def explain(order_id):
    if not order_id.isalnum() or len(order_id) > 64:
        return err("Invalid order id.")
    try:
        return jsonify(ds.clean(ps.explain_order(order_id)))
    except ps.InputError as e:
        return err(str(e), 404)


@api.post("/upload")
def upload():
    f = request.files.get("file")
    if f is None:
        return err("No file was sent (form field name must be 'file').")
    try:
        return jsonify(validate_csv(f.filename, f.read()))
    except UploadError as e:
        return err(str(e))
