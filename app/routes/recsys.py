"""Recommendation Intelligence Center: page + /api/rec/* endpoints (additive blueprint)."""
import re
from flask import Blueprint, jsonify, render_template, request
from recsys import service as S

rec = Blueprint("rec", __name__)
_ID = re.compile(r"^[0-9a-f]{32}$")


def _err(msg, code=400):
    return jsonify(error=msg), code


def _svc():
    return S.get_service() if S.available() else None


def _k(default=10):
    try:
        return max(1, min(int(request.args.get("k", default)), 50))
    except ValueError:
        return default


@rec.route("/recsys")
def page():
    return render_template("recsys.html", active="recsys")


@rec.route("/api/rec/status")
def status():
    ok = S.available()
    return jsonify(ready=ok, hybrid_weights=(_svc().weights if ok else None), evaluation_available=S.RecService.evaluation() is not None)


@rec.route("/api/rec/customers")
def customers():
    s = _svc()
    if not s: return _err("recommendation data not built; run python -m recsys.data", 503)
    t = request.args.get("tier")
    if t not in (None, "", "established", "limited"): return _err("tier must be established or limited")
    return jsonify(customers=s.customers(t or None, n=_k(20)))


@rec.route("/api/rec/customer/<cid>")
def customer(cid):
    s = _svc()
    if not s: return _err("recommendation data not built", 503)
    if not _ID.match(cid): return _err("invalid customer id")
    try:
        r = s.recommend(cid, _k(), request.args.get("model", "hybrid"))
    except ValueError as e:
        return _err(str(e))
    return (jsonify(r) if r else _err("unknown customer; use /api/rec/new for customers with no history", 404))


@rec.route("/api/rec/new")
def new_customer():
    s = _svc()
    if not s: return _err("recommendation data not built", 503)
    r = s.recommend_new(request.args.get("category") or None, _k())
    return (jsonify(r) if r else _err("unknown category", 404))


@rec.route("/api/rec/categories")
def categories():
    s = _svc()
    return jsonify(categories=s.categories()) if s else _err("recommendation data not built", 503)


@rec.route("/api/rec/products")
def products():
    s = _svc()
    if not s: return _err("recommendation data not built", 503)
    return jsonify(products=s.search_products(request.args.get("q", "")[:40], _k(25)))


@rec.route("/api/rec/similar/<pid>")
def similar(pid):
    s = _svc()
    if not s: return _err("recommendation data not built", 503)
    if not _ID.match(pid): return _err("invalid product id")
    r = s.similar(pid, _k())
    return jsonify(r) if r else _err("unknown product", 404)


@rec.route("/api/rec/together/<pid>")
def together(pid):
    s = _svc()
    if not s: return _err("recommendation data not built", 503)
    if not _ID.match(pid): return _err("invalid product id")
    r = s.together(pid, _k())
    return jsonify(r) if r else _err("unknown product", 404)


@rec.route("/api/rec/insights")
def insights():
    s = _svc()
    return jsonify(s.insights(_k(15))) if s else _err("recommendation data not built", 503)


@rec.route("/api/rec/explain/<cid>/<pid>")
def explain(cid, pid):
    s = _svc()
    if not s: return _err("recommendation data not built", 503)
    if not (_ID.match(cid) and _ID.match(pid)): return _err("invalid id")
    r = S.explain(s, cid, pid, request.args.get("model", "hybrid"))
    return jsonify(r) if r else _err("unknown customer or product", 404)


@rec.route("/api/rec/evaluation")
def evaluation():
    e = S.RecService.evaluation()
    return jsonify(e) if e else _err("evaluation not run; python -m recsys.evaluate", 404)
