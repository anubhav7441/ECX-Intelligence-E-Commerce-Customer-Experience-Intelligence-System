"""Insights & Advice: page + JSON APIs (login required through the global guard; dataset ownership checked)."""
from flask import Blueprint, jsonify, render_template
from app.insights import olist, dataset
from multids import registry as R

insights = Blueprint("insights", __name__)


@insights.route("/insights")
def page():
    return render_template("insights.html", active="insights")


@insights.get("/api/insights/olist")
def api_olist():
    try:
        return jsonify(ready=True, title="Olist marketplace (2016-2018 public data)", summary=olist.summary(), cards=olist.build())
    except FileNotFoundError:
        return jsonify(error="The Olist data files are not installed (extract the data ZIP)."), 404


@insights.get("/api/insights/dataset/<did>")
def api_dataset(did):
    from app.routes.multids import _check
    bad = _check(did)
    if bad:
        return bad
    if did == R.OLIST_ID:
        return api_olist()
    r = dataset.build(did)
    r["title"] = (R.meta(did) or {}).get("name", did)
    return jsonify(r)
