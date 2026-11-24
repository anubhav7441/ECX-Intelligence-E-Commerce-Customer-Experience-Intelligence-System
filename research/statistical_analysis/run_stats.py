"""Statistical analysis (Phase 6 / PDF Phases 8-10). Every number is computed from the real analytic dataset.

Outputs: research/statistical_analysis/results/stats_results.json  and  hypothesis_verdicts.csv
Run from project root:  python -m research.statistical_analysis.run_stats
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)
ALPHA = 0.05


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def boot_mean_diff(a, b, n_boot=2000):
    """Bootstrap CI for mean(a)-mean(b) using multinomial resampling on integer scores 1..5."""
    vals = np.arange(1, 6)
    pa = np.bincount(a.astype(int), minlength=6)[1:] / len(a)
    pb = np.bincount(b.astype(int), minlength=6)[1:] / len(b)
    ma = RNG.multinomial(len(a), pa, size=n_boot) @ vals / len(a)
    mb = RNG.multinomial(len(b), pb, size=n_boot) @ vals / len(b)
    d = ma - mb
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def mwu(a, b):
    """Mann-Whitney U (two-sided) + Cliff's delta (= 2*AUC-1) for group a vs b."""
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    delta = 2 * u / (len(a) * len(b)) - 1
    return float(u), float(p), float(delta)


def fp(p):
    return "<1e-300" if p == 0 else f"{p:.2g}"


def cliff_label(d):
    d = abs(d)
    return "negligible" if d < .147 else "small" if d < .33 else "medium" if d < .474 else "large"


def chi2_rate(df, flag_col, outcome="dissatisfied"):
    ct = pd.crosstab(df[flag_col], df[outcome])
    chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    return ct, float(chi2), float(p), float(np.sqrt(chi2 / n))


def main():
    df = pd.read_parquet(ROOT / "data" / "processed" / "orders_analytic.parquet")
    R = {"n_analytic_orders": int(len(df))}

    # ------------------------------------------------------------------ descriptives
    R["descriptive"] = {
        "mean_review_score": float(df.review_score.mean()),
        "median_review_score": float(df.review_score.median()),
        "pct_dissatisfied_1_2": float(df.dissatisfied.mean() * 100),
        "pct_late": float(df.is_late.mean() * 100),
        "mean_delivery_days": float(df.delivery_days.mean()),
        "median_delivery_days": float(df.delivery_days.median()),
        "mean_estimated_days": float(df.estimated_days.mean()),
        "mean_delay_days_when_late": float(df.loc[df.is_late == 1, "delay_days"].mean()),
        "score_distribution_pct": (df.review_score.value_counts(normalize=True).sort_index() * 100).round(2).to_dict(),
    }

    # ------------------------------------------------------------------ H1: late vs on-time (main + sensitivity)
    def late_test(d, label):
        a = d.loc[d.is_late == 1, "review_score"].values
        b = d.loc[d.is_late == 0, "review_score"].values
        u, p, delta = mwu(a, b)
        lo, hi = boot_mean_diff(a, b)
        ct, chi2, p_chi, phi = chi2_rate(d, "is_late")
        k1, n1 = int(((d.is_late == 1) & (d.dissatisfied == 1)).sum()), int((d.is_late == 1).sum())
        k0, n0 = int(((d.is_late == 0) & (d.dissatisfied == 1)).sum()), int((d.is_late == 0).sum())
        # odds ratio with Woolf CI
        orr = (k1 / (n1 - k1)) / (k0 / (n0 - k0))
        se = np.sqrt(1 / k1 + 1 / (n1 - k1) + 1 / k0 + 1 / (n0 - k0))
        return {
            "label": label, "n_late": n1, "n_on_time": n0,
            "mean_score_late": float(a.mean()), "mean_score_on_time": float(b.mean()),
            "mean_diff": float(a.mean() - b.mean()), "mean_diff_ci95": [lo, hi],
            "mannwhitney_U": u, "p_value": p, "cliffs_delta": delta, "effect_size_label": cliff_label(delta),
            "pct_dissatisfied_late": k1 / n1 * 100, "pct_dissatisfied_on_time": k0 / n0 * 100,
            "chi2": chi2, "chi2_p": p_chi, "phi": phi,
            "odds_ratio": float(orr), "odds_ratio_ci95": [float(np.exp(np.log(orr) - 1.96 * se)),
                                                          float(np.exp(np.log(orr) + 1.96 * se))],
        }
    R["H1_late_vs_ontime"] = late_test(df, "all analytic orders")
    R["H1_sensitivity_review_after_delivery_only"] = late_test(df[df.review_after_delivery], "reviews created on/after delivery date")

    # dose-response: delay bands
    bands = pd.cut(df.delay_days, [-np.inf, -1, 0, 3, 7, 14, np.inf],
                   labels=["early (>=1d)", "on estimated day", "1-3d late", "4-7d late", "8-14d late", ">14d late"])
    g = df.groupby(bands, observed=True).agg(n=("order_id", "size"), mean_score=("review_score", "mean"),
                                             pct_dissatisfied=("dissatisfied", lambda s: s.mean() * 100))
    R["delay_band_table"] = g.round(3).reset_index().astype({"delay_days": str}).to_dict(orient="records")
    rho, p = stats.spearmanr(df.delay_days, df.review_score)
    R["spearman_delay_vs_score"] = {"rho": float(rho), "p": float(p)}

    # ------------------------------------------------------------------ Q5 price, H4 freight, Q6 distance (Spearman + quartile contrast)
    def monotone(col, label):
        d = df[[col, "review_score", "dissatisfied"]].dropna()
        rho, p = stats.spearmanr(d[col], d.review_score)
        q = pd.qcut(d[col], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"], duplicates="drop")
        t = d.groupby(q, observed=True).agg(n=("review_score", "size"), mean_score=("review_score", "mean"),
                                            pct_dissatisfied=("dissatisfied", lambda s: s.mean() * 100)).round(3)
        top, bot = d[q == t.index[-1]], d[q == t.index[0]]
        u, pm, delta = mwu(top.review_score.values, bot.review_score.values)
        return {"label": label, "n": int(len(d)), "spearman_rho": float(rho), "p": float(p),
                "quartile_table": t.reset_index().rename(columns={col: "quartile"}).astype({"quartile": str}).to_dict(orient="records"),
                "top_vs_bottom_quartile": {"mannwhitney_p": pm, "cliffs_delta": delta, "label": cliff_label(delta)}}
    R["Q5_price_vs_score"] = monotone("items_price", "order item value")
    R["H4_freight_ratio_vs_score"] = monotone("freight_ratio", "freight ratio (freight/price)")
    R["Q6_distance_vs_score"] = monotone("distance_km", "seller-customer distance (km)")
    R["delivery_days_vs_score"] = monotone("delivery_days", "actual delivery days")

    # ------------------------------------------------------------------ multivariable logistic regression (controls)
    m = df[["dissatisfied", "is_late", "items_price", "freight_ratio", "distance_km", "delivery_days", "n_items"]].dropna().copy()
    m["log_price"] = np.log1p(m.items_price)
    m["freight_ratio"] = m.freight_ratio.clip(upper=m.freight_ratio.quantile(.99))
    m["distance_100km"] = m.distance_km / 100
    X = sm.add_constant(m[["is_late", "log_price", "freight_ratio", "distance_100km", "delivery_days", "n_items"]])
    fit = sm.Logit(m.dissatisfied, X).fit(disp=0)
    ci = fit.conf_int()
    R["logit_controls"] = {
        "n": int(len(m)), "pseudo_r2": float(fit.prsquared),
        "terms": {k: {"odds_ratio": float(np.exp(fit.params[k])), "ci95": [float(np.exp(ci.loc[k, 0])), float(np.exp(ci.loc[k, 1]))],
                      "p": float(fit.pvalues[k])} for k in fit.params.index if k != "const"},
    }

    # ------------------------------------------------------------------ H2: sellers heterogeneity ; H3: categories ; H5: regions
    def heterogeneity(group_col, min_n, label):
        g = df.groupby(group_col).agg(n=("order_id", "size"), k=("dissatisfied", "sum")).reset_index()
        g = g[g.n >= min_n]
        overall = df.dissatisfied.mean()
        ct = np.array([g.k, g.n - g.k])
        chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
        v = float(np.sqrt(chi2 / (g.n.sum() * (min(ct.shape) - 1))))
        g["rate"] = g.k / g.n
        lo_hi = g.apply(lambda r: wilson(r.k, r.n), axis=1)
        g["ci_lo"] = [x[0] for x in lo_hi]; g["ci_hi"] = [x[1] for x in lo_hi]
        worse = g[g.ci_lo > overall].sort_values("rate", ascending=False)
        return {"label": label, "min_orders": min_n, "n_groups": int(len(g)), "orders_covered": int(g.n.sum()),
                "overall_rate_pct": float(overall * 100), "chi2": float(chi2), "dof": int(dof), "p": float(p),
                "cramers_v": v, "n_groups_significantly_worse": int(len(worse)),
                "worst_groups": worse.head(10)[[group_col, "n", "rate", "ci_lo", "ci_hi"]].round(4).to_dict(orient="records")}
    R["H2_sellers"] = heterogeneity("primary_seller_id", 50, "sellers with >=50 orders")
    R["H3_categories"] = heterogeneity("primary_category", 100, "categories with >=100 orders")
    R["H5_states"] = heterogeneity("customer_state", 100, "customer states with >=100 orders")
    R["H5_regions"] = heterogeneity("customer_region", 100, "customer macro-regions")

    # ------------------------------------------------------------------ Q10: repeat purchase after bad experience
    # Only first orders; purchases before 2018-03-01 so every customer has a >=6 month follow-up window in the data.
    first = df[(df.order_rank == 1) & (df.order_purchase_timestamp < "2018-03-01")]
    ct, chi2, p, phi = chi2_rate(first.assign(repeat=first.has_later_order), "dissatisfied", "repeat")
    ks = first.groupby("dissatisfied").agg(n=("order_id", "size"), repeat_rate=("has_later_order", "mean"))
    k1, n1 = int(first[first.dissatisfied == 1].has_later_order.sum()), int((first.dissatisfied == 1).sum())
    k0, n0 = int(first[first.dissatisfied == 0].has_later_order.sum()), int((first.dissatisfied == 0).sum())
    R["Q10_repeat_purchase"] = {
        "n_first_orders": int(len(first)), "repeat_rate_dissatisfied_pct": k1 / n1 * 100, "repeat_rate_satisfied_pct": k0 / n0 * 100,
        "n_dissatisfied": n1, "n_satisfied": n0, "repeaters_dissatisfied": k1, "repeaters_satisfied": k0,
        "ci95_dissatisfied": [x * 100 for x in wilson(k1, n1)], "ci95_satisfied": [x * 100 for x in wilson(k0, n0)],
        "chi2": chi2, "p": p, "phi": phi,
        "overall_pct_customers_with_repeat_order": float((df.groupby("customer_unique_id").n_orders_customer.first() > 1).mean() * 100),
    }

    # ------------------------------------------------------------------ verdict table (AI hypothesis -> data test -> evidence -> verdict)
    h1 = R["H1_late_vs_ontime"]
    verdicts = []

    def mono_verdict(r):
        """Verdict for a monotone-relationship hypothesis: needs the expected sign, p<alpha AND a non-negligible effect."""
        d = abs(r["top_vs_bottom_quartile"]["cliffs_delta"])
        if r["spearman_rho"] < 0 and r["p"] < ALPHA and d >= .147:
            return "Confirmed"
        if r["spearman_rho"] < 0 and r["p"] < ALPHA:
            return "Statistically significant but negligible practical effect"
        return "Rejected / Inconclusive"

    def add(hid, hyp, test, evidence, verdict, note=""):
        verdicts.append({"id": hid, "hypothesis": hyp, "data_test": test, "statistical_evidence": evidence.replace("p=<", "p<"), "verdict": verdict, "note": note})
    add("H1", "Late delivery is a major driver of negative reviews.",
        "Mann-Whitney U + chi-square + odds ratio + bootstrap CI; logistic regression with controls",
        f"Mean score {h1['mean_score_late']:.2f} (late) vs {h1['mean_score_on_time']:.2f} (on time); dissatisfied {h1['pct_dissatisfied_late']:.1f}% vs "
        f"{h1['pct_dissatisfied_on_time']:.1f}%; Cliff's delta {h1['cliffs_delta']:.2f} ({h1['effect_size_label']}); OR {h1['odds_ratio']:.2f}; p={fp(h1['p_value'])}",
        "Confirmed (association)" if h1["p_value"] < ALPHA and abs(h1["cliffs_delta"]) >= .147 else "Inconclusive",
        "Association only. Effect is much smaller when restricted to reviews written after delivery (see H1_sensitivity): many late-order reviews were created before the recorded delivery date.")
    h2 = R["H2_sellers"]
    add("H2", "Certain sellers have systematically poor customer experience.",
        "Chi-square homogeneity across sellers (>=50 orders); Wilson CI vs overall rate",
        f"chi2={h2['chi2']:.1f}, dof={h2['dof']}, p={fp(h2['p'])}, Cramer's V={h2['cramers_v']:.3f}; {h2['n_groups_significantly_worse']} of {h2['n_groups']} sellers have CI lower bound above overall rate",
        "Confirmed" if h2["p"] < ALPHA and h2["n_groups_significantly_worse"] > 0 else "Inconclusive")
    h3 = R["H3_categories"]
    add("H3", "Some product categories have unusually high complaint rates.",
        "Chi-square homogeneity across categories (>=100 orders); Wilson CI",
        f"chi2={h3['chi2']:.1f}, dof={h3['dof']}, p={fp(h3['p'])}, V={h3['cramers_v']:.3f}; {h3['n_groups_significantly_worse']} of {h3['n_groups']} categories significantly worse",
        "Confirmed" if h3["p"] < ALPHA and h3["n_groups_significantly_worse"] > 0 else "Inconclusive")
    h4 = R["H4_freight_ratio_vs_score"]
    add("H4", "High freight costs are associated with poor customer satisfaction.",
        "Spearman correlation + top vs bottom quartile Mann-Whitney",
        f"Spearman rho={h4['spearman_rho']:.3f} (p={fp(h4['p'])}); top-vs-bottom quartile Cliff's delta {h4['top_vs_bottom_quartile']['cliffs_delta']:.3f} ({h4['top_vs_bottom_quartile']['label']})",
        mono_verdict(h4),
        "Statistical significance is not the same as practical importance; check effect size.")
    h5 = R["H5_states"]
    add("H5", "Certain geographic regions have operational problems.",
        "Chi-square homogeneity across states (>=100 orders); Wilson CI",
        f"chi2={h5['chi2']:.1f}, dof={h5['dof']}, p={fp(h5['p'])}, V={h5['cramers_v']:.3f}; {h5['n_groups_significantly_worse']} of {h5['n_groups']} states significantly worse",
        "Confirmed" if h5["p"] < ALPHA and h5["n_groups_significantly_worse"] > 0 else "Inconclusive")
    q5 = R["Q5_price_vs_score"]
    add("Q5", "Expensive orders are more likely to receive negative reviews.",
        "Spearman + top vs bottom quartile", f"rho={q5['spearman_rho']:.3f} (p={fp(q5['p'])}); Cliff's delta {q5['top_vs_bottom_quartile']['cliffs_delta']:.3f} ({q5['top_vs_bottom_quartile']['label']})",
        mono_verdict(q5))
    q6 = R["Q6_distance_vs_score"]
    add("Q6", "Longer seller-customer distance lowers satisfaction.",
        "Spearman + top vs bottom quartile", f"rho={q6['spearman_rho']:.3f} (p={fp(q6['p'])}); Cliff's delta {q6['top_vs_bottom_quartile']['cliffs_delta']:.3f} ({q6['top_vs_bottom_quartile']['label']})",
        mono_verdict(q6))
    q10 = R["Q10_repeat_purchase"]
    add("Q10", "Customers with a negative first experience are less likely to purchase again.",
        "Chi-square on first orders (purchases before 2018-03-01)",
        f"Repeat rate {q10['repeat_rate_dissatisfied_pct']:.2f}% (dissatisfied) vs {q10['repeat_rate_satisfied_pct']:.2f}% (satisfied); p={fp(q10['p'])}",
        "Confirmed" if (q10["p"] < ALPHA and q10["repeat_rate_dissatisfied_pct"] < q10["repeat_rate_satisfied_pct"]) else "Not supported (no significant difference)",
        "Repeat purchasing is rare in this dataset, so power and generalisation are limited.")
    pd.DataFrame(verdicts).to_csv(OUT / "hypothesis_verdicts.csv", index=False)
    R["verdicts"] = verdicts

    with open(OUT / "stats_results.json", "w") as f:
        json.dump(R, f, indent=2, default=str)
    return R


if __name__ == "__main__":
    R = main()
    print(json.dumps({k: R[k] for k in ["descriptive", "H1_late_vs_ontime", "H1_sensitivity_review_after_delivery_only",
                                           "delay_band_table", "spearman_delay_vs_score"]}, indent=1, default=str))
    for v in R["verdicts"]:
        print(f"\n[{v['id']}] {v['verdict']}\n   {v['statistical_evidence']}")
