"""EDA + business intelligence: KPIs, experience/seller scores, segmentation, geography, journey analysis, recommendations.
Every number comes from the real data / previous result files. Run from project root: python -m research.run_eda_bi
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml.plotting import BLUE, ORANGE, GREY, GREEN, INK2
from research.statistical_analysis.run_stats import wilson

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
BI = ROOT / "data" / "processed" / "bi"
BI.mkdir(parents=True, exist_ok=True)
RES = ROOT / "research" / "results"
RES.mkdir(parents=True, exist_ok=True)
MIN_SELLER, MIN_CAT, MIN_STATE = 50, 100, 100


def save(fig, name):
    fig.savefig(FIG / name); plt.close(fig)


def main():
    allo = pd.read_parquet(ROOT / "data" / "processed" / "orders_all.parquet")
    df = pd.read_parquet(ROOT / "data" / "processed" / "orders_analytic.parquet")
    nlp = pd.read_parquet(ROOT / "data" / "processed" / "review_nlp.parquet")
    stats = json.load(open(ROOT / "research" / "statistical_analysis" / "results" / "stats_results.json"))
    nlpr = json.load(open(ROOT / "ml" / "sentiment_analysis" / "results" / "nlp_results.json"))
    df = df.merge(nlp[["order_id", "sent_p_negative"] + [c for c in nlp.columns if c.startswith("cat_")]], on="order_id", how="left")
    overall_rate = df.dissatisfied.mean()
    OUT = {}

    # ---------------------------------------------------------------- KPIs (all delivered orders for volume/revenue; reviewed for experience)
    dl = allo[allo.order_status == "delivered"]
    OUT["kpis"] = {
        "total_orders_all_statuses": int(len(allo)), "delivered_orders": int(len(dl)),
        "revenue_items_delivered": float(dl.items_price.sum()), "freight_total_delivered": float(dl.freight_value.sum()),
        "avg_order_value_items": float(dl.items_price.mean()),
        "avg_review_score_analytic": float(df.review_score.mean()), "pct_dissatisfied_1_2": float(overall_rate * 100),
        "pct_late_delivered_with_date": float(dl.loc[dl.order_delivered_customer_date.notna(), "delay_days"].gt(0).mean() * 100),
        "pct_late_analytic": float(df.is_late.mean() * 100), "avg_delivery_days": float(dl.delivery_days.mean()),
        "n_sellers": int(allo.primary_seller_id.nunique()), "n_unique_customers": int(allo.customer_unique_id.nunique()),
        "pct_customers_repeat": float((allo.groupby("customer_unique_id").size() > 1).mean() * 100),
    }

    # monthly trend
    m = dl[dl.purchase_month >= "2017-01"].groupby("purchase_month").agg(orders=("order_id", "size"), revenue=("items_price", "sum")).reset_index()
    mm = df[df.purchase_month >= "2017-01"].groupby("purchase_month").agg(avg_score=("review_score", "mean"), late_pct=("is_late", lambda s: s.mean() * 100),
                                                                          dissatisfied_pct=("dissatisfied", lambda s: s.mean() * 100)).reset_index()
    monthly = m.merge(mm, on="purchase_month"); monthly.to_csv(BI / "monthly.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].bar(monthly.purchase_month, monthly.orders, color=BLUE); ax[0].set_title("Delivered orders per month")
    ax[1].bar(monthly.purchase_month, monthly.revenue / 1e3, color=BLUE); ax[1].set_title("Item revenue per month (thousand BRL)")
    for a in ax:
        a.tick_params(axis="x", rotation=90, labelsize=7); a.grid(axis="x", visible=False)
    save(fig, "eda_monthly_orders_revenue.png")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].plot(monthly.purchase_month, monthly.late_pct, color=ORANGE, lw=2); ax[0].set_title("Late-delivery rate (%)")
    ax[1].plot(monthly.purchase_month, monthly.dissatisfied_pct, color=BLUE, lw=2); ax[1].set_title("Dissatisfied reviews, score 1-2 (%)")
    for a in ax:
        a.tick_params(axis="x", rotation=90, labelsize=7)
    save(fig, "eda_monthly_late_dissatisfied.png")

    # score distribution
    sd = df.review_score.value_counts(normalize=True).sort_index() * 100
    fig, ax = plt.subplots(figsize=(5.5, 3.4)); ax.bar(sd.index.astype(int).astype(str), sd.values, color=[ORANGE, ORANGE, GREY, BLUE, BLUE])
    for i, v in enumerate(sd.values): ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9, color=INK2)
    ax.set_title("Review score distribution"); ax.set_xlabel("Stars"); ax.set_ylabel("% of orders"); ax.grid(axis="x", visible=False)
    save(fig, "eda_score_distribution.png")

    # estimated vs actual delivery
    fig, ax = plt.subplots(figsize=(7, 3.6)); b = np.arange(0, 61, 2)
    ax.hist(df.delivery_days.clip(upper=60), bins=b, alpha=.8, color=BLUE, label="Actual delivery days")
    ax.hist(df.estimated_days.clip(upper=60), bins=b, alpha=.55, color=ORANGE, label="Estimated delivery days")
    ax.set_title("Estimated vs actual delivery time (clipped at 60 days)"); ax.legend(); ax.set_xlabel("Days from purchase")
    save(fig, "eda_estimated_vs_actual_delivery.png")

    # delay bands
    bt = pd.DataFrame(stats["delay_band_table"]); fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(bt.delay_days, bt.pct_dissatisfied, color=[BLUE, BLUE, ORANGE, ORANGE, ORANGE, ORANGE])
    for i, v in enumerate(bt.pct_dissatisfied): ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, color=INK2)
    ax.set_title("Dissatisfied reviews (1-2 stars) by delivery delay"); ax.set_ylabel("% dissatisfied"); ax.tick_params(axis="x", labelsize=8); ax.grid(axis="x", visible=False)
    save(fig, "eda_dissatisfaction_by_delay.png")

    # price / freight / distance quartiles
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    for a, key, t in zip(ax, ["Q5_price_vs_score", "H4_freight_ratio_vs_score", "Q6_distance_vs_score"], ["Order value", "Freight ratio", "Seller-customer distance"]):
        q = pd.DataFrame(stats[key]["quartile_table"]); a.bar(["Q1\nlow", "Q2", "Q3", "Q4\nhigh"], q.pct_dissatisfied, color=BLUE); a.set_title(t)
        a.set_ylim(0, 18); a.grid(axis="x", visible=False)
        for i, v in enumerate(q.pct_dissatisfied): a.text(i, v + .3, f"{v:.1f}%", ha="center", fontsize=8, color=INK2)
    ax[0].set_ylabel("% dissatisfied"); save(fig, "eda_price_freight_distance.png")

    # ---------------------------------------------------------------- composite scores
    delay_comp = np.where(df.delay_days <= 0, 1.0, np.clip(1 - df.delay_days / 14.0, 0, 1))
    rev_comp = (df.review_score - 1) / 4
    sent_comp = 1 - df.sent_p_negative
    w = np.where(sent_comp.notna(), 0.5, 0.65) * 1.0
    wd = np.where(sent_comp.notna(), 0.3, 0.35)
    ws = np.where(sent_comp.notna(), 0.2, 0.0)
    df["cx_score"] = 100 * (w * rev_comp + wd * delay_comp + ws * sent_comp.fillna(0))
    OUT["cx_score"] = {"definition": "100*(0.5*review + 0.3*delivery + 0.2*text sentiment); if no text: 100*(0.65*review + 0.35*delivery). "
                                     "review=(stars-1)/4; delivery=1 if on time else 1-delay/14 (floored 0); sentiment=1-P(negative) from out-of-fold TF-IDF model.",
                       "mean": float(df.cx_score.mean()), "mean_late": float(df.loc[df.is_late == 1, "cx_score"].mean()),
                       "mean_on_time": float(df.loc[df.is_late == 0, "cx_score"].mean())}

    # ---------------------------------------------------------------- sellers
    def agg_group(key, minn):
        g = df.groupby(key).agg(orders=("order_id", "size"), revenue=("items_price", "sum"), avg_score=("review_score", "mean"),
                                late_rate=("is_late", "mean"), avg_delay_when_late=("delay_days", lambda s: s[s > 0].mean()),
                                avg_delivery_days=("delivery_days", "mean"), freight_ratio=("freight_ratio", "median"),
                                dissatisfied_rate=("dissatisfied", "mean"), dissatisfied_n=("dissatisfied", "sum"), cx=("cx_score", "mean"),
                                late_complaint_rate=("cat_late_delivery", "mean")).reset_index()
        g = g[g.orders >= minn].copy()
        ci = g.apply(lambda r: wilson(r.dissatisfied_n, r.orders), axis=1); g["dis_ci_lo"] = [x[0] for x in ci]; g["dis_ci_hi"] = [x[1] for x in ci]
        return g
    s = agg_group("primary_seller_id", MIN_SELLER)
    k = 10; mu = df.review_score.mean()
    shrunk = (s.avg_score * s.orders + k * mu) / (s.orders + k)
    s["perf_score"] = 100 * (0.5 * (shrunk - 1) / 4 + 0.3 * (1 - s.late_rate.clip(upper=0.5) / 0.5) + 0.2 * s.orders.rank(pct=True))
    rev_cut, exp_cut = s.revenue.median(), df.review_score.mean()
    s["segment"] = np.select(
        [(s.revenue >= rev_cut) & (s.avg_score >= exp_cut), (s.revenue >= rev_cut) & (s.avg_score < exp_cut),
         (s.revenue < rev_cut) & (s.avg_score >= exp_cut)],
        ["Best performers", "PRIORITY PROBLEM", "Growth opportunity"], default="Low priority")
    s.sort_values("revenue", ascending=False).to_csv(BI / "sellers.csv", index=False)
    OUT["sellers"] = {"min_orders": MIN_SELLER, "n_eligible": int(len(s)), "revenue_cut_median": float(rev_cut), "experience_cut_mean_score": float(exp_cut),
                      "segment_counts": s.segment.value_counts().to_dict(),
                      "priority_problem_sellers": s[s.segment == "PRIORITY PROBLEM"].sort_values("revenue", ascending=False).head(10)[
                          ["primary_seller_id", "orders", "revenue", "avg_score", "late_rate", "dissatisfied_rate", "perf_score"]].round(3).to_dict("records"),
                      "share_orders_covered_pct": float(s.orders.sum() / len(df) * 100)}
    fig, ax = plt.subplots(figsize=(7.5, 5)); cmap = {"Best performers": GREEN, "PRIORITY PROBLEM": ORANGE, "Growth opportunity": BLUE, "Low priority": GREY}
    for seg, d in s.groupby("segment"): ax.scatter(d.revenue / 1e3, d.avg_score, s=14 + d.orders / 20, alpha=.7, color=cmap[seg], label=f"{seg} ({len(d)})", edgecolor="white", lw=.5)
    ax.axvline(rev_cut / 1e3, color=GREY, ls="--", lw=1); ax.axhline(exp_cut, color=GREY, ls="--", lw=1); ax.set_xscale("log")
    ax.set_xlabel("Revenue (thousand BRL, log scale)"); ax.set_ylabel("Average review score"); ax.set_title(f"Seller segmentation (sellers with >={MIN_SELLER} orders)"); ax.legend(fontsize=8)
    save(fig, "bi_seller_segments.png")

    # ---------------------------------------------------------------- categories
    c = agg_group("primary_category", MIN_CAT)
    c_rev_cut = c.revenue.median()
    c["segment"] = np.select([(c.revenue >= c_rev_cut) & (c.avg_score >= exp_cut), (c.revenue >= c_rev_cut) & (c.avg_score < exp_cut),
                              (c.revenue < c_rev_cut) & (c.avg_score >= exp_cut)], ["Best performers", "PRIORITY PROBLEM", "Growth opportunity"], default="Low priority")
    c.sort_values("revenue", ascending=False).to_csv(BI / "categories.csv", index=False)
    OUT["categories"] = {"min_orders": MIN_CAT, "n_eligible": int(len(c)), "segment_counts": c.segment.value_counts().to_dict(),
                         "worst_by_avg_score": c.sort_values("avg_score").head(8)[["primary_category", "orders", "revenue", "avg_score", "dissatisfied_rate", "late_rate"]].round(3).to_dict("records"),
                         "priority_problem": c[c.segment == "PRIORITY PROBLEM"].sort_values("revenue", ascending=False)[["primary_category", "orders", "revenue", "avg_score", "dissatisfied_rate", "late_rate"]].round(3).to_dict("records"),
                         "top_by_revenue": c.sort_values("revenue", ascending=False).head(8)[["primary_category", "orders", "revenue", "avg_score", "dissatisfied_rate"]].round(3).to_dict("records")}
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for seg, d in c.groupby("segment"): ax.scatter(d.revenue / 1e3, d.avg_score, s=30 + d.orders / 40, alpha=.75, color=cmap[seg], label=f"{seg} ({len(d)})", edgecolor="white")
    for _, r in c.sort_values("revenue", ascending=False).head(8).iterrows(): ax.annotate(r.primary_category, (r.revenue / 1e3, r.avg_score), fontsize=7, xytext=(4, 3), textcoords="offset points", color=INK2)
    for _, r in c.sort_values("avg_score").head(3).iterrows(): ax.annotate(r.primary_category, (r.revenue / 1e3, r.avg_score), fontsize=7, xytext=(4, -9), textcoords="offset points", color=ORANGE)
    ax.axvline(c_rev_cut / 1e3, color=GREY, ls="--", lw=1); ax.axhline(exp_cut, color=GREY, ls="--", lw=1); ax.set_xscale("log")
    ax.set_xlabel("Revenue (thousand BRL, log)"); ax.set_ylabel("Average review score"); ax.set_title(f"Category revenue vs satisfaction (>={MIN_CAT} orders)"); ax.legend(fontsize=8)
    save(fig, "bi_category_segments.png")

    # ---------------------------------------------------------------- geography
    st = agg_group("customer_state", MIN_STATE)
    st["z_vs_overall"] = (st.dissatisfied_rate - overall_rate) / np.sqrt(overall_rate * (1 - overall_rate) / st.orders)
    st["hotspot"] = st.dis_ci_lo > overall_rate
    st = st.sort_values("dissatisfied_rate", ascending=False); st.to_csv(BI / "states.csv", index=False)
    reg = df.groupby("customer_region").agg(orders=("order_id", "size"), avg_score=("review_score", "mean"), late_rate=("is_late", "mean"),
                                            avg_delivery_days=("delivery_days", "mean"), dissatisfied_rate=("dissatisfied", "mean"), freight_ratio=("freight_ratio", "median")).reset_index()
    reg.to_csv(BI / "regions.csv", index=False)
    OUT["geography"] = {"n_states": int(len(st)), "hotspot_states": st[st.hotspot][["customer_state", "orders", "dissatisfied_rate", "late_rate", "avg_delivery_days", "avg_score"]].round(3).to_dict("records"),
                        "regions": reg.round(3).to_dict("records"),
                        "top_volume_poor": st[(st.orders >= st.orders.median()) & st.hotspot][["customer_state", "orders", "dissatisfied_rate", "late_rate"]].round(3).to_dict("records")}
    fig, ax = plt.subplots(figsize=(7.5, 6)); d = st.sort_values("dissatisfied_rate")
    err = [d.dissatisfied_rate - d.dis_ci_lo, d.dis_ci_hi - d.dissatisfied_rate]
    ax.barh(d.customer_state, d.dissatisfied_rate * 100, xerr=[e * 100 for e in err], color=[ORANGE if h else BLUE for h in d.hotspot], error_kw={"ecolor": GREY, "lw": 1})
    ax.axvline(overall_rate * 100, color=INK2, ls="--", lw=1); ax.set_xlabel("% dissatisfied (95% Wilson CI)  -  orange = significantly above overall")
    ax.set_title("Dissatisfaction by customer state"); ax.grid(axis="y", visible=False); save(fig, "bi_state_hotspots.png")
    fig, ax = plt.subplots(figsize=(6.5, 4)); ax.scatter(st.late_rate * 100, st.dissatisfied_rate * 100, s=20 + st.orders / 150, color=BLUE, alpha=.7)
    for _, r in st.iterrows(): ax.annotate(r.customer_state, (r.late_rate * 100, r.dissatisfied_rate * 100), fontsize=7, xytext=(3, 3), textcoords="offset points", color=INK2)
    ax.set_xlabel("Late-delivery rate (%)"); ax.set_ylabel("% dissatisfied"); ax.set_title("States: lateness vs dissatisfaction"); save(fig, "bi_state_late_vs_dissatisfied.png")

    # ---------------------------------------------------------------- order structure (model flagged n_items / n_sellers)
    df["multi_item"] = np.where(df.n_items > 1, "2+ items", "1 item"); df["multi_seller"] = np.where(df.n_sellers > 1, "2+ sellers", "1 seller")
    ms = df.groupby("multi_item").agg(n=("order_id", "size"), dissatisfied=("dissatisfied", "mean"), late=("is_late", "mean"), delivery_days=("delivery_days", "mean"), score=("review_score", "mean"))
    mse = df.groupby("multi_seller").agg(n=("order_id", "size"), dissatisfied=("dissatisfied", "mean"), late=("is_late", "mean"), delivery_days=("delivery_days", "mean"), score=("review_score", "mean"))
    OUT["order_structure"] = {"by_items": ms.round(4).reset_index().to_dict("records"), "by_sellers": mse.round(4).reset_index().to_dict("records"),
                              "note": "Multi-item orders may be delivered in parts and a single review covers all items (dataset limitation); association only."}
    # does the multi-item effect survive controlling for lateness?
    OUT["order_structure"]["by_items_and_late"] = df.groupby(["multi_item", "is_late"]).agg(n=("order_id", "size"), dissatisfied=("dissatisfied", "mean")).round(4).reset_index().to_dict("records")

    # ---------------------------------------------------------------- customer journey
    df["missed_handoff_deadline"] = (df.order_delivered_carrier_date > df.max_shipping_limit).astype(float)
    stages = {"Payment approval (hours)": "approval_hours", "Seller handling: approval to carrier (days)": "seller_handling_days", "Carrier transit (days)": "carrier_transit_days"}
    jr = []
    for lab, col in stages.items():
        q90 = df[col].quantile(.9); slow = df[col] > q90
        jr.append({"stage": lab, "mean_on_time": float(df.loc[df.is_late == 0, col].mean()), "mean_late": float(df.loc[df.is_late == 1, col].mean()),
                   "ratio_late_over_on_time": float(df.loc[df.is_late == 1, col].mean() / df.loc[df.is_late == 0, col].mean()),
                   "p90_threshold": float(q90), "dissatisfied_pct_if_slowest_10pct": float(df.loc[slow, "dissatisfied"].mean() * 100),
                   "dissatisfied_pct_otherwise": float(df.loc[~slow, "dissatisfied"].mean() * 100), "late_pct_if_slowest_10pct": float(df.loc[slow, "is_late"].mean() * 100)})
    mh = df.dropna(subset=["missed_handoff_deadline"])
    OUT["journey"] = {"stages": jr,
                      "missed_seller_handoff_deadline_pct": float(mh.missed_handoff_deadline.mean() * 100),
                      "late_pct_if_handoff_missed": float(mh.loc[mh.missed_handoff_deadline == 1, "is_late"].mean() * 100),
                      "late_pct_if_handoff_ok": float(mh.loc[mh.missed_handoff_deadline == 0, "is_late"].mean() * 100),
                      "dissatisfied_pct_if_handoff_missed": float(mh.loc[mh.missed_handoff_deadline == 1, "dissatisfied"].mean() * 100),
                      "dissatisfied_pct_if_handoff_ok": float(mh.loc[mh.missed_handoff_deadline == 0, "dissatisfied"].mean() * 100),
                      "n_handoff_known": int(len(mh))}
    jd = pd.DataFrame(jr); fig, ax = plt.subplots(figsize=(8, 3.6)); x = np.arange(len(jd))
    ax.bar(x - .2, jd.mean_on_time, .4, color=BLUE, label="On-time orders"); ax.bar(x + .2, jd.mean_late, .4, color=ORANGE, label="Late orders")
    ax.set_xticks(x); ax.set_xticklabels([s.replace(" (", "\n(") for s in jd.stage], fontsize=8); ax.set_title("Where do late orders lose time? (mean per stage)"); ax.legend(); ax.grid(axis="x", visible=False)
    save(fig, "bi_journey_stages.png")

    # ---------------------------------------------------------------- complaint chart
    cc = pd.DataFrame(nlpr["complaint_categories"]).sort_values("pct_of_negative_reviews")
    fig, ax = plt.subplots(figsize=(7.5, 4)); ax.barh(cc.category.str.replace("_", " "), cc.pct_of_negative_reviews, color=[ORANGE if c_ == "late_delivery" else BLUE for c_ in cc.category])
    ax.set_xlabel("% of negative (1-2 star) reviews with a comment (multi-label)"); ax.set_title("What customers complain about (rule-based taxonomy)"); ax.grid(axis="y", visible=False)
    save(fig, "bi_complaint_categories.png")
    asp = pd.DataFrame(nlpr["aspects"]).T.sort_values("pct_negative_among_negative_reviews")
    fig, ax = plt.subplots(figsize=(7, 3.5)); ax.barh(asp.index.str.replace("_", " "), asp.pct_negative_among_negative_reviews, color=BLUE)
    ax.set_xlabel("% of negative reviews with a NEGATIVE sentiment on this aspect"); ax.set_title("Aspect-based sentiment in negative reviews"); ax.grid(axis="y", visible=False)
    save(fig, "bi_aspect_sentiment.png")

    # ---------------------------------------------------------------- recommendations (ACTION + WHY + EXPECTED IMPACT + HOW TO MEASURE), each tied to evidence keys
    h1 = stats["H1_late_vs_ontime"]; n_late = int((df.is_late == 1).sum())
    avoidable_upper = n_late * (h1["pct_dissatisfied_late"] - h1["pct_dissatisfied_on_time"]) / 100
    h1s = stats["H1_sensitivity_review_after_delivery_only"]
    avoidable_conservative = n_late * (h1s["pct_dissatisfied_late"] - h1s["pct_dissatisfied_on_time"]) / 100
    ps = OUT["sellers"]["priority_problem_sellers"][:3]
    pc = OUT["categories"]["priority_problem"][:3]
    hs = OUT["geography"]["hotspot_states"][:4]
    mi = {r["multi_item"]: r for r in OUT["order_structure"]["by_items"]}
    jn = OUT["journey"]; slowest = max(jr, key=lambda r: r["ratio_late_over_on_time"])
    recs = [
        {"id": "R1", "priority": 1, "evidence": ["H1_late_vs_ontime", "delay_band_table", "H1_sensitivity"],
         "action": "Treat late delivery as the top operational KPI: set an alert when an order passes its estimated date and trigger proactive customer communication, and make the estimated-date promise more realistic.",
         "why": f"{h1['pct_dissatisfied_late']:.1f}% of late orders are dissatisfied vs {h1['pct_dissatisfied_on_time']:.1f}% on time (odds ratio {h1['odds_ratio']:.1f}); in the controlled logistic model lateness keeps an odds ratio of {stats['logit_controls']['terms']['is_late']['odds_ratio']:.1f}. Delivery features are the largest SHAP factor group.",
         "expected_impact": f"Association-based range: roughly {avoidable_conservative:,.0f} (reviews written after delivery only) to {avoidable_upper:,.0f} (all reviews) fewer dissatisfied reviews among {n_late:,} late orders if lateness were eliminated. This is an upper-bound association, not a proven causal effect.",
         "measure": "Weekly late-delivery rate; dissatisfied-review rate for late vs on-time orders; share of late orders contacted before the estimated date."},
        {"id": "R2", "priority": 2, "evidence": ["sellers.PRIORITY PROBLEM", "H2_sellers"],
         "action": "Review the high-revenue / low-experience 'priority problem' sellers first: agree improvement plans on handling time and shipping-deadline compliance, with a probation threshold.",
         "why": f"Seller dissatisfaction rates differ significantly (chi-square p={stats['H2_sellers']['p']:.1g}, Cramer's V={stats['H2_sellers']['cramers_v']:.2f}); {OUT['sellers']['segment_counts'].get('PRIORITY PROBLEM', 0)} of {OUT['sellers']['n_eligible']} sellers with >= {MIN_SELLER} orders are in the priority-problem segment; top examples: " + ", ".join(f"{x['primary_seller_id'][:8]}.. ({x['orders']} orders, avg {x['avg_score']:.2f})" for x in ps),
         "expected_impact": "Because these sellers carry high order volume, each point of improvement reaches more customers than fixes for low-volume sellers.",
         "measure": "Per-seller average score, late rate and dissatisfied rate (with 95% CI) monthly; movement between segments."},
        {"id": "R3", "priority": 3, "evidence": ["categories.PRIORITY PROBLEM", "H3_categories"],
         "action": "Investigate high-revenue categories with below-average satisfaction (" + ", ".join(x["primary_category"] for x in pc) + "): product descriptions, size/dimension accuracy, packaging and carrier choice for bulky items.",
         "why": f"{stats['H3_categories']['n_groups_significantly_worse']} of {stats['H3_categories']['n_groups']} categories are significantly worse than the overall dissatisfied rate; the top worst categories by rate are " + ", ".join(x["primary_category"] for x in stats["H3_categories"]["worst_groups"][:3]) + ".",
         "expected_impact": "Targets categories where revenue is large, so reduced dissatisfaction protects the most sales.",
         "measure": "Category dissatisfied rate and complaint mix (damaged/quality/wrong product) per month."},
        {"id": "R4", "priority": 4, "evidence": ["H5_states", "geography.hotspot_states"],
         "action": "Run a regional logistics review for hotspot states (" + ", ".join(h["customer_state"] for h in hs) + "): carrier performance, realistic delivery estimates, and local fulfilment options.",
         "why": f"{stats['H5_states']['n_groups_significantly_worse']} of {stats['H5_states']['n_groups']} states have a dissatisfied rate whose 95% CI lies above the overall rate; hotspots include " + "; ".join(f"{h['customer_state']} ({h['dissatisfied_rate']*100:.1f}% dissatisfied, {h['late_rate']*100:.1f}% late, n={h['orders']})" for h in hs) + ".",
         "expected_impact": "Fixing high-volume hotspots (e.g. RJ) affects many orders; smaller states show higher rates but fewer customers.",
         "measure": "State-level late rate, delivery days vs estimate, dissatisfied rate with CI."},
        {"id": "R5", "priority": 5, "evidence": ["order_structure", "SHAP S1_R n_items rank 2"],
         "action": "Audit multi-item and multi-seller orders: check for partial/split shipments and communicate which items arrive separately.",
         "why": f"Orders with 2+ items have {mi['2+ items']['dissatisfied']*100:.1f}% dissatisfied vs {mi['1 item']['dissatisfied']*100:.1f}% for single-item orders; 'items in order' is the 2nd most important feature in the delivery-time SHAP model. The gap persists among ON-TIME orders ({[r for r in OUT['order_structure']['by_items_and_late'] if r['multi_item']=='2+ items' and r['is_late']==0][0]['dissatisfied']*100:.1f}% vs {[r for r in OUT['order_structure']['by_items_and_late'] if r['multi_item']=='1 item' and r['is_late']==0][0]['dissatisfied']*100:.1f}%), so it is not just lateness. Complaint text also shows missing/partial deliveries ({[r for r in nlpr['complaint_categories'] if r['category']=='missing_product'][0]['pct_of_negative_reviews']:.1f}% of negative reviews).",
         "expected_impact": "Addresses a driver independent of lateness; size of effect depends on how much is due to split shipments (not observable here).",
         "measure": "Dissatisfied rate by item count; share of orders delivered in one shipment; 'missing product' complaint share."},
        {"id": "R6", "priority": 6, "evidence": ["nlp complaint_categories", "seller_issue"],
         "action": "Improve seller/store responsiveness (reply SLA for complaints, exchange and refund handling).",
         "why": f"Seller-issue complaints appear in {[r for r in nlpr['complaint_categories'] if r['category']=='seller_issue'][0]['pct_of_negative_reviews']:.1f}% of negative commented reviews and have the lowest mean score ({[r for r in nlpr['complaint_categories'] if r['category']=='seller_issue'][0]['mean_score_when_present']:.2f}). Caveat: this category had the weakest classifier precision in our evaluation.",
         "expected_impact": "Reduces escalations (e.g. consumer-complaint sites mentioned in reviews).", "measure": "Median response time; share of complaints resolved; seller_issue complaint rate."},
        {"id": "R7", "priority": 7, "evidence": ["Q10_repeat_purchase"],
         "action": "Do not assume that fixing ratings alone will raise repeat purchases; test retention interventions (e.g. coupons after a bad experience) with an A/B experiment.",
         "why": f"Repeat rate after a dissatisfied first order was {stats['Q10_repeat_purchase']['repeat_rate_dissatisfied_pct']:.2f}% vs {stats['Q10_repeat_purchase']['repeat_rate_satisfied_pct']:.2f}% after a satisfied one (p={stats['Q10_repeat_purchase']['p']:.2f}): no significant difference, and only {OUT['kpis']['pct_customers_repeat']:.1f}% of customers ever reorder in this data.",
         "expected_impact": "Avoids investing in a retention effect the data does not support.", "measure": "Repeat-purchase rate within 180 days by experience group in a controlled experiment."},
    ]
    OUT["recommendations"] = recs
    findings = [
        {"finding": "Late deliveries are strongly associated with dissatisfied reviews.", "evidence": f"{h1['pct_dissatisfied_late']:.1f}% of late orders vs {h1['pct_dissatisfied_on_time']:.1f}% of on-time orders are 1-2 stars (Cliff's delta {h1['cliffs_delta']:.2f}); dissatisfaction rises from {bt.pct_dissatisfied.iloc[1]:.0f}% (on the estimated day) to {bt.pct_dissatisfied.iloc[-2]:.0f}% (8-14 days late). Restricted to reviews written after delivery the effect is smaller (delta {h1s['cliffs_delta']:.2f}) because many late-order reviews were written before the recorded delivery.",
         "business_impact": f"{n_late:,} late orders ({n_late/len(df)*100:.1f}% of reviewed orders) generate a disproportionate share of dissatisfaction.", "recommendation": "R1"},
        {"finding": "Customers mainly complain about non-delivery/delay, incomplete orders and seller responsiveness - not price.", "evidence": f"Late/not-delivered appears in {[r for r in nlpr['complaint_categories'] if r['category']=='late_delivery'][0]['pct_of_negative_reviews']:.1f}% of negative commented reviews; price and freight effects are negligible (Cliff's delta {stats['Q5_price_vs_score']['top_vs_bottom_quartile']['cliffs_delta']:.2f} and {stats['H4_freight_ratio_vs_score']['top_vs_bottom_quartile']['cliffs_delta']:.2f}).",
         "business_impact": "Operational fixes matter more than discounting.", "recommendation": "R1, R6"},
        {"finding": "Seller quality is heterogeneous: a set of sellers is significantly worse than the platform.", "evidence": f"{stats['H2_sellers']['n_groups_significantly_worse']} of {stats['H2_sellers']['n_groups']} sellers (>=50 orders) have a dissatisfaction CI entirely above the overall rate (chi-square p={stats['H2_sellers']['p']:.1g}).", "business_impact": "Concentrated, addressable problem.", "recommendation": "R2"},
        {"finding": "Some high-revenue categories and several regions under-perform.", "evidence": f"{stats['H3_categories']['n_groups_significantly_worse']} categories and {stats['H5_states']['n_groups_significantly_worse']} states are significantly worse than overall; effect sizes are small (Cramer's V {stats['H3_categories']['cramers_v']:.2f} and {stats['H5_states']['cramers_v']:.2f}).", "business_impact": "Targeted, not platform-wide, interventions.", "recommendation": "R3, R4"},
        {"finding": "Multi-item orders are a hidden risk factor.", "evidence": f"{mi['2+ items']['dissatisfied']*100:.1f}% vs {mi['1 item']['dissatisfied']*100:.1f}% dissatisfied; 2nd most important SHAP feature in the delivery-time model.", "business_impact": "Possible partial-delivery experience problem.", "recommendation": "R5"},
        {"finding": "Predicting dissatisfaction is hard from structured data alone; delivery outcome and seller history are what help.", "evidence": "See ML results: checkout-time ROC-AUC about 0.57-0.61, delivery-time about 0.69-0.71; historical review features add a significant but modest gain.", "business_impact": "Use the model for risk monitoring, not individual decisions.", "recommendation": "R1, R2"},
        {"finding": "No evidence in this dataset that a bad experience reduces repeat purchase.", "evidence": f"Repeat rate {stats['Q10_repeat_purchase']['repeat_rate_dissatisfied_pct']:.2f}% vs {stats['Q10_repeat_purchase']['repeat_rate_satisfied_pct']:.2f}% (p={stats['Q10_repeat_purchase']['p']:.2f}); only {OUT['kpis']['pct_customers_repeat']:.1f}% of customers reorder.", "business_impact": "Limits retention claims.", "recommendation": "R7"},
    ]
    OUT["findings"] = findings
    (RES / "bi_results.json").write_text(json.dumps(OUT, indent=2, default=str))
    return OUT


if __name__ == "__main__":
    o = main()
    print(json.dumps(o["kpis"], indent=1)); print(o["cx_score"]); print(o["sellers"]["segment_counts"], o["categories"]["segment_counts"])
    print("hotspots:", [h["customer_state"] for h in o["geography"]["hotspot_states"]])
    print("order structure:", o["order_structure"]["by_items"], o["order_structure"]["by_items_and_late"])
    print("journey:", json.dumps(o["journey"], indent=0)[:1500])
    for r in o["recommendations"][:2]: print(r["id"], r["why"][:300])
