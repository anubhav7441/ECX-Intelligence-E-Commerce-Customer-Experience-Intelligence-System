"""HTML pages."""
from pathlib import Path
from flask import Blueprint, abort, render_template, send_from_directory
from app.services import data_service as ds
from app.services import predict_service as ps

pages = Blueprint("pages", __name__)
FIG = Path(__file__).resolve().parents[2] / "docs" / "figures"


@pages.route("/figures/<path:name>")
def figure(name):
    if not name.endswith(".png"):
        abort(404)
    return send_from_directory(FIG, name)


@pages.route("/dashboard")
def dashboard():
    b, s = ds.bi(), ds.stats()
    return render_template("dashboard.html", active="dashboard", kpis=ds.clean(b["kpis"]), monthly=ds.monthly(),
                           score_dist=s["descriptive"]["score_distribution_pct"], bands=s["delay_band_table"],
                           h1=ds.clean(s["H1_late_vs_ontime"]), cx=ds.clean(b["cx_score"]), findings=ds.clean(b["findings"]))


@pages.route("/data")
def data_management():
    return render_template("data.html", active="data", db=ds.db_status(), prep=ds.prep_report())


@pages.route("/reviews")
def reviews():
    n, e = ds.nlp(), ds.nlp_eval()
    return render_template("reviews.html", active="reviews", nlp=ds.clean(n), ev=ds.clean(e))


@pages.route("/predict")
def predict():
    m = ds.ml()
    return render_template("predict.html", active="predict", opts=ps.options(), ml=ds.clean(m), cal=ps.calibration())


@pages.route("/explain")
def explain():
    return render_template("explain.html", active="explain", g=ds.clean(ds.shap_global()))


@pages.route("/sellers")
def sellers():
    b = ds.bi()
    return render_template("sellers.html", active="sellers", rows=ds.sellers(), info=ds.clean(b["sellers"]))


@pages.route("/products")
def products():
    return render_template("products.html", active="products", rows=ds.categories(), info=ds.clean(ds.bi()["categories"]))


@pages.route("/geography")
def geography():
    return render_template("geography.html", active="geography", states=ds.states(), regions=ds.regions(), geo=ds.clean(ds.bi()["geography"]))


@pages.route("/research")
def research():
    s = ds.stats()
    return render_template("research.html", active="research", s=ds.clean(s), ml=ds.clean(ds.ml()), nlp=ds.clean(ds.nlp()),
                           ev=ds.clean(ds.nlp_eval()), journey=ds.clean(ds.bi()["journey"]))


@pages.route("/recommendations")
def recommendations():
    b = ds.bi()
    return render_template("recommendations.html", active="recommendations", recs=ds.clean(b["recommendations"]),
                           findings=ds.clean(b["findings"]))
