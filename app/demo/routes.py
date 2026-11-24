"""Guest demo blueprint: /demo page and /api/demo/* (public, heavily limited, backend-enforced)."""
from flask import Blueprint, jsonify, render_template, request, current_app
from app.demo import service as S
from app.auth import security as sec

demo = Blueprint("demo", __name__)


def _limit_ip():
    return sec.rate_limited(f"demo|{request.remote_addr}", 60)


@demo.route("/demo")
def page():
    return render_template("demo.html", active="demo", limits=S.limits())


@demo.get("/api/demo/summary")
def summary():
    if _limit_ip():
        return jsonify(error="Too many requests. Please wait a minute."), 429
    return jsonify(S.summary())


@demo.post("/api/demo/predict")
def predict():
    if _limit_ip():
        return jsonify(error="Too many requests. Please wait a minute."), 429
    b = request.get_json(silent=True) or {}
    oid = str(b.get("order_prefix", "")).strip().lower()
    L = S.limits()
    if S.used("predictions") >= L["predictions"]:
        return jsonify(error=f"Demo limit reached ({L['predictions']} predictions). Register a free account for unlimited predictions.", limit=True), 403
    if len(oid) < 8:
        return jsonify(error="Enter at least 8 characters of an order id from the preview table."), 400
    ids = [i for i in S.sample().order_id if i.startswith(oid)]
    if len(ids) != 1:
        return jsonify(error="No unique order in the demo sample matches. Copy the id from the preview table."), 404
    from app.services import predict_service as ps, data_service as ds
    try:
        res = ds.clean(ps.explain_order(ids[0], top=5))
    except ps.InputError as e:
        return jsonify(error=str(e)), 404
    S.use("predictions")
    res["demo_note"] = "Demo: top 5 drivers only. These are model attributions (associations), not proven causes. Registered users see the full explanation."
    return jsonify(res)


@demo.post("/api/demo/investigate")
def investigate():
    if _limit_ip():
        return jsonify(error="Too many requests. Please wait a minute."), 429
    b = request.get_json(silent=True) or {}
    L = S.limits()
    if S.used("investigations") >= L["investigations"]:
        return jsonify(error=f"Demo limit reached ({L['investigations']} investigations). Register a free account for full AI investigations.", limit=True), 403
    r = S.seller_investigation(b.get("seller_prefix"))
    if not r["ok"]:
        return jsonify(error=r["error"]), 400
    S.use("investigations")
    return jsonify(r)


@demo.post("/api/demo/upload")
def upload():
    """Guest 'check my own data': first N rows only (N = demo row limit), nothing stored, limited uses per session."""
    if _limit_ip():
        return jsonify(error="Too many requests. Please wait a minute."), 429
    L = S.limits()
    cap = int(current_app.config.get("DEMO_MAX_UPLOADS", 5))
    if S.used("uploads") >= cap:
        return jsonify(error=f"Demo limit reached ({cap} file checks). Register a free account to analyse your own data without this limit.", limit=True), 403
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="Choose a .csv or .xlsx file first."), 400
    r = S.check_own_data(f.filename, f.read(int(current_app.config.get("DEMO_UPLOAD_MAX_BYTES", 5 * 1024 * 1024)) + 1))
    if not r["ok"]:
        return jsonify(error=r["error"]), 400
    S.use("uploads")
    r["usage"] = {"used": S.used("uploads"), "max": cap}
    return jsonify(r)


@demo.errorhandler(413)
def too_big(e):
    return jsonify(error="File is too large for demo mode (limit 5 MB)."), 413


@demo.get("/api/demo/sellers")
def sellers():
    return jsonify(sellers=S.top_sellers())
