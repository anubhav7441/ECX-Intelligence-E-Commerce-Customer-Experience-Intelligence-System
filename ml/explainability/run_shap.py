"""Global + local SHAP analysis for the delivery-time model (S1+R) and the checkout-time model (S0+R).
Run from project root:  python -m ml.explainability.run_shap
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from ml.plotting import BLUE, ORANGE, INK2
from ml.prediction.features import FEATURE_GROUPS
from ml.explainability.explain import load_artifact, shap_matrix, aggregate, explain_rows

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "figures"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
NICE = {"delay_days": "Delay vs estimate (days)", "delivery_days": "Delivery time (days)", "is_late": "Late flag",
        "estimated_days": "Estimated delivery (days)", "carrier_transit_days": "Carrier transit (days)",
        "seller_handling_days": "Seller handling (days)", "approval_hours": "Payment approval (h)",
        "seller_hist_neg": "Seller past dissatisfaction", "seller_hist_mean": "Seller past mean score",
        "seller_hist_late_cmp": "Seller past late-complaints", "seller_hist_n": "Seller past #reviews",
        "cat_hist_neg": "Category past dissatisfaction", "cat_hist_mean": "Category past mean score",
        "state_hist_neg": "State past dissatisfaction", "log_items_price": "Item price (log)",
        "log_freight_value": "Freight (log)", "freight_ratio": "Freight ratio", "n_items": "Items in order",
        "distance_km": "Seller-customer distance (km)", "primary_category": "Product category",
        "customer_state": "Customer state", "seller_state": "Seller state", "main_payment_type": "Payment type"}


def nice(f):
    return NICE.get(f, f)


def run(tag, title):
    art = load_artifact(tag)
    full = pd.read_parquet(ROOT / "data" / "processed" / "orders_with_history.parquet")
    n_test = json.load(open(ROOT / "ml" / "prediction" / "results" / "ml_results.json"))["split"]["n_test"]
    te = full.iloc[-n_test:].reset_index(drop=True)
    X, sv, base = shap_matrix(art, te)
    agg = aggregate(sv, art["feature_names"], art["cat_cols"])
    imp = agg.abs().mean().sort_values(ascending=False)
    # grouped importance
    g = {k: float(sum(imp.get(f, 0) for f in fs)) for k, fs in FEATURE_GROUPS.items()}
    tot = sum(g.values()); grp = {k: v / tot * 100 for k, v in sorted(g.items(), key=lambda x: -x[1]) if v > 0}
    # bar chart of top 15 original features
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh([nice(f) for f in top.index], top.values, color=BLUE)
    ax.set_xlabel("Mean |SHAP value| (log-odds of dissatisfied)"); ax.set_title(f"What drives the prediction - {title}")
    ax.grid(axis="y", visible=False); fig.savefig(FIG / f"shap_importance_{tag}.png"); plt.close(fig)
    # grouped
    fig, ax = plt.subplots(figsize=(7, 3.6)); items = list(grp.items())[::-1]
    ax.barh([k for k, _ in items], [v for _, v in items], color=[ORANGE if k == "Delivery performance" else BLUE for k, _ in items])
    ax.set_xlabel("Share of total mean |SHAP| (%)"); ax.set_title(f"Importance by factor group - {title}"); ax.grid(axis="y", visible=False)
    fig.savefig(FIG / f"shap_groups_{tag}.png"); plt.close(fig)
    # beeswarm on numeric features
    num = art["num_cols"]; names = list(art["feature_names"]); ix = [names.index(c) for c in num]
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv[:, ix], pd.DataFrame(X[:, ix], columns=[nice(c) for c in num]), max_display=14, show=False, plot_size=None)
    plt.title(f"SHAP summary (numeric features) - {title}"); plt.savefig(FIG / f"shap_beeswarm_{tag}.png"); plt.close("all")
    # dependence for delay
    if "delay_days" in num:
        j = names.index("delay_days"); xs = X[:, j]; m = (xs > -30) & (xs < 40)
        fig, ax = plt.subplots(figsize=(7, 4)); ax.scatter(xs[m], sv[m, j], s=4, alpha=.25, color=BLUE)
        ax.axvline(0, color=ORANGE, lw=1.2); ax.set_xlabel("Delay vs estimated date (days; >0 = late)"); ax.set_ylabel("SHAP (log-odds)")
        ax.set_title("Effect of delay on predicted dissatisfaction"); fig.savefig(FIG / f"shap_dependence_delay_{tag}.png"); plt.close(fig)
    # local explanations: top-risk true positive, confident-satisfied, false negative
    p = art["model"].predict_proba(X)[:, 1]; y = te.dissatisfied.values
    picks = {"highest_risk_true_dissatisfied": int(np.argmax(np.where(y == 1, p, -1))),
             "lowest_risk_true_satisfied": int(np.argmin(np.where(y == 0, p, 2))),
             "missed_dissatisfied_false_negative": int(np.argmin(np.where(y == 1, p, 2)))}
    local = {}
    for k, i in picks.items():
        e = explain_rows(art, te.iloc[[i]], top=8)[0]; e["order_id"] = te.order_id.iloc[i]; e["actual_dissatisfied"] = int(y[i]); local[k] = e
        c = e["contributions"][::-1]
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.barh([f"{nice(d['feature'])} = {d['value'] if not isinstance(d['value'], float) else round(d['value'], 2)}" for d in c],
                [d["shap_logodds"] for d in c], color=[ORANGE if d["shap_logodds"] > 0 else BLUE for d in c])
        ax.axvline(0, color=INK2, lw=.8)
        ax.set_title(f"{k.replace('_', ' ')}  (P(dissatisfied)={e['probability_dissatisfied']:.2f}, actual={int(y[i])})", fontsize=10)
        ax.set_xlabel("SHAP (log-odds)  orange = raises risk, blue = lowers risk"); ax.grid(axis="y", visible=False)
        fig.savefig(FIG / f"shap_local_{tag}_{k}.png"); plt.close(fig)
    rank = {f: r + 1 for r, f in enumerate(imp.index)}
    res = {"model": art["model_name"], "feature_set": art["feature_set"], "n_explained": int(len(te)),
           "top15": [{"feature": f, "mean_abs_shap": float(v)} for f, v in imp.head(15).items()],
           "group_share_pct": grp, "delivery_feature_ranks": {f: rank.get(f) for f in ["delay_days", "delivery_days", "is_late", "carrier_transit_days", "estimated_days"] if f in rank},
           "local_examples": local}
    (OUT / f"shap_{tag}.json").write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    for tag, title in [("S1_R", "delivery-time model"), ("S0_R", "checkout-time model")]:
        r = run(tag, title)
        print(tag, r["model"], "| top5:", [(d["feature"], round(d["mean_abs_shap"], 3)) for d in r["top15"][:5]])
        print("   groups %:", {k: round(v, 1) for k, v in r["group_share_pct"].items()})
        print("   delivery ranks:", r["delivery_feature_ranks"])
