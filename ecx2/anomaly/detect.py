"""Anomaly detection engine. An anomaly is an INVESTIGATION SIGNAL, not proof of fraud or fault.

Detectors (all deterministic, statistics-first):
  1 seller late-rate        : one-sided binomial test vs platform rate (BH-FDR corrected), n >= MIN_N
  2 seller dissatisfaction  : same, target review_score <= 2
  3 seller sudden change    : seller-month dissatisfaction vs that seller's earlier rate (Fisher exact, FDR)
  4 freight burden          : seller's share of orders with freight_ratio far above category median (robust z)
  5 complaint frequency     : seller's share of reviews with seller-issue/wrong/missing-product complaints vs platform
  6 multivariate            : Isolation Forest on seller feature vectors (unsupervised ranking, no p-value)
  7 platform monthly shift  : monthly late/dissatisfied rate vs trailing 3-month mean (robust z)
Run: python -m ecx2.anomaly.detect
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from statsmodels.stats.multitest import multipletests

from ecx2.common import V2_DATA, nlp_flags, orders, save_json

MIN_N, MIN_MONTH_N, MIN_PRIOR_N = 30, 15, 40
ALPHA = 0.05
CAVEAT = "Statistical signal only - not proof of misconduct or fault. Check for data issues, seasonality and small samples first."


def _binom_flags(g, col, base_rate, kind, min_n=MIN_N, label=""):
    rows = []
    for sid, x in g:
        n = len(x)
        if n < min_n:
            continue
        k = int(x[col].sum())
        p = stats.binomtest(k, n, base_rate, alternative="greater").pvalue
        rows.append({"entity": sid, "type": kind, "n": n, "events": k, "observed": k / n, "expected": base_rate, "p": p})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_adj"] = multipletests(df.p, method="fdr_bh")[1]
    lo, _ = zip(*[stats.binomtest(int(r.events), int(r.n)).proportion_ci(0.95, method="wilson") and
                  (stats.binomtest(int(r.events), int(r.n)).proportion_ci(0.95, method="wilson").low, 0) for r in df.itertuples()])
    df["wilson_low"] = lo
    return df


def run():
    d = orders().copy()
    nl = nlp_flags()
    if nl is not None:
        d = d.merge(nl[["order_id", "cat_seller_issue", "cat_wrong_product", "cat_missing_product", "cat_late_delivery"]], on="order_id", how="left")
    for c in ["cat_seller_issue", "cat_wrong_product", "cat_missing_product", "cat_late_delivery"]:
        d[c] = d.get(c, 0)
        d[c] = d[c].fillna(0).astype(int)
    d["seller_complaint"] = ((d.cat_seller_issue + d.cat_wrong_product + d.cat_missing_product) > 0).astype(int)
    d["late"] = d.is_late.astype(int); d["neg"] = d.dissatisfied.astype(int)
    base_late, base_neg, base_comp = d.late.mean(), d.neg.mean(), d.seller_complaint.mean()
    out = []
    G = d.groupby("primary_seller_id")

    # 1, 2
    for col, base, kind, why in [("late", base_late, "seller_late_rate", "late-delivery share"),
                                 ("neg", base_neg, "seller_dissatisfaction", "1-2 star review share"),
                                 ("seller_complaint", base_comp, "seller_complaint_frequency", "share of reviews citing seller-side problems (seller issue / wrong or missing product)")]:
        f = _binom_flags(G, col, base, kind)
        f = f[(f.p_adj < ALPHA) & (f.observed > f.expected * 1.25)]
        for r in f.itertuples():
            out.append({"entity_type": "seller", "entity": r.entity, "detector": kind, "period": "all", "n": r.n,
                        "measured": round(r.observed, 4), "expected": round(r.expected, 4), "p_adj": float(r.p_adj),
                        "score": float(min(100, 100 * (1 - r.p_adj) * min(1.0, (r.observed / max(r.expected, 1e-9) - 1) / 2 + .3))),
                        "why_flagged": f"{r.events}/{r.n} orders show the {why} = {r.observed:.1%} vs {r.expected:.1%} platform-wide (FDR-adjusted p={r.p_adj:.3g}; Wilson 95% lower bound {r.wilson_low:.1%}).",
                        "possible_explanation": {"seller_late_rate": "Handling or carrier delays; remote destinations; capacity limits in peak months.",
                                                 "seller_dissatisfaction": "Product-quality or description problems, delivery delays, or service issues.",
                                                 "seller_complaint_frequency": "Listing accuracy, stock errors or packing problems (hypotheses only)."}[kind],
                        "recommended_investigation": "Review this seller's recent orders and reviews, split by month and product, before any decision.",
                        "caveat": CAVEAT})
    # 3 sudden change per seller-month
    chg = []
    for sid, x in G:
        if len(x) < MIN_PRIOR_N + MIN_MONTH_N:
            continue
        for ym, m in x.groupby("purchase_ym"):
            prior = x[x.purchase_ym < ym]
            if len(m) < MIN_MONTH_N or len(prior) < MIN_PRIOR_N:
                continue
            a, b = int(m.neg.sum()), int(prior.neg.sum())
            if a / len(m) <= b / len(prior) * 1.5:
                continue
            p = stats.fisher_exact([[a, len(m) - a], [b, len(prior) - b]], alternative="greater")[1]
            chg.append({"entity": sid, "period": ym, "n": len(m), "a": a, "obs": a / len(m), "exp": b / len(prior), "p": p, "n_prior": len(prior)})
    if chg:
        c = pd.DataFrame(chg)
        c["p_adj"] = multipletests(c.p, method="fdr_bh")[1]
        for r in c[c.p_adj < ALPHA].itertuples():
            out.append({"entity_type": "seller", "entity": r.entity, "detector": "seller_sudden_negative_change", "period": r.period,
                        "n": r.n, "measured": round(r.obs, 4), "expected": round(r.exp, 4), "p_adj": float(r.p_adj),
                        "score": float(min(100, 100 * (1 - r.p_adj) * min(1.0, r.obs / max(r.exp, 1e-9) / 4 + .3))),
                        "why_flagged": f"In {r.period}, {r.a}/{r.n} orders were rated 1-2 stars ({r.obs:.1%}) vs {r.exp:.1%} in this seller's earlier {r.n_prior} reviewed orders (FDR-adjusted p={r.p_adj:.3g}).",
                        "possible_explanation": "A stock, carrier, listing or staffing change in that month; or one bad batch of products.",
                        "recommended_investigation": f"Compare {r.period} orders with earlier ones: products, carriers, delivery times, review text.",
                        "caveat": CAVEAT})
    # 4 freight burden (robust z of freight_ratio within category)
    med = d.groupby("primary_category").freight_ratio.transform("median")
    mad = d.groupby("primary_category").freight_ratio.transform(lambda s: (s - s.median()).abs().median()) * 1.4826
    d["fr_z"] = (d.freight_ratio - med) / mad.replace(0, np.nan)
    d["fr_extreme"] = (d.fr_z > 3.5).astype(int)
    base_fr = d.fr_extreme.mean()
    f = _binom_flags(d.groupby("primary_seller_id"), "fr_extreme", base_fr, "seller_freight_burden")
    if not f.empty:
        f = f[(f.p_adj < ALPHA) & (f.observed > f.expected * 1.5)]
        for r in f.itertuples():
            out.append({"entity_type": "seller", "entity": r.entity, "detector": "seller_freight_burden", "period": "all", "n": r.n,
                        "measured": round(r.observed, 4), "expected": round(r.expected, 4), "p_adj": float(r.p_adj),
                        "score": float(min(100, 100 * (1 - r.p_adj) * min(1.0, r.observed / max(r.expected, 1e-9) / 5 + .3))),
                        "why_flagged": f"{r.observed:.1%} of this seller's orders have a freight-to-price ratio far above their category median (robust z>3.5) vs {r.expected:.1%} platform-wide (FDR p={r.p_adj:.3g}).",
                        "possible_explanation": "Heavy or bulky items with low price, remote shipping, or pricing/freight settings.",
                        "recommended_investigation": "Check freight settings and whether customers see shipping cost before purchase.", "caveat": CAVEAT})
    # 6 Isolation Forest on seller feature vectors
    S = G.agg(n=("order_id", "size"), late=("late", "mean"), neg=("neg", "mean"), comp=("seller_complaint", "mean"),
              fr=("freight_ratio", "mean"), dd=("delivery_days", "mean"), hd=("seller_handling_days", "mean"), dist=("distance_km", "median"))
    S = S[S.n >= MIN_N].copy()
    feats = ["late", "neg", "comp", "fr", "dd", "hd", "dist"]
    Z = (S[feats] - S[feats].median()) / (S[feats].quantile(.75) - S[feats].quantile(.25)).replace(0, 1)
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42).fit(Z.fillna(0))
    raw = -iso.score_samples(Z.fillna(0))
    S["iso_score"] = 100 * (raw - raw.min()) / (raw.max() - raw.min())
    thr = np.quantile(S.iso_score, 0.97)
    top = S[S.iso_score >= thr].sort_values("iso_score", ascending=False)
    for sid, r in top.iterrows():
        dev = (Z.loc[sid].abs().sort_values(ascending=False).head(3)).index.tolist()
        names = {"late": "late rate", "neg": "1-2 star share", "comp": "seller-complaint share", "fr": "freight ratio", "dd": "delivery days", "hd": "handling days", "dist": "distance"}
        out.append({"entity_type": "seller", "entity": sid, "detector": "seller_multivariate_isolation_forest", "period": "all", "n": int(r.n),
                    "measured": round(float(r.iso_score), 1), "expected": None, "p_adj": None, "score": float(r.iso_score),
                    "why_flagged": "Unusual COMBINATION of seller metrics (top 3% isolation score). Most atypical on: " + ", ".join(f"{names[c]} ({S.loc[sid, c]:.3g})" for c in dev) + ".",
                    "possible_explanation": "Atypical business profile (e.g. very remote or bulky-goods seller). Unsupervised ranking: no p-value.",
                    "recommended_investigation": "Compare against sellers in the same category and region before concluding anything.", "caveat": CAVEAT})
    S.reset_index().to_csv(V2_DATA / "seller_features.csv", index=False)
    # 7 platform monthly shifts
    M = d.groupby("purchase_ym").agg(n=("order_id", "size"), late=("late", "mean"), neg=("neg", "mean"))
    M = M[M.n >= 200]
    for col, nm in [("late", "late rate"), ("neg", "1-2 star share")]:
        base = M[col].shift(1).rolling(3, min_periods=3)
        mu, sd = base.mean(), base.std().clip(lower=0.005)
        z = (M[col] - mu) / sd
        for ym in z[z > 3].index:
            out.append({"entity_type": "platform", "entity": "ALL", "detector": f"platform_monthly_{col}", "period": ym, "n": int(M.loc[ym, "n"]),
                        "measured": round(float(M.loc[ym, col]), 4), "expected": round(float(mu[ym]), 4), "p_adj": None,
                        "score": float(min(100, 40 + 10 * z[ym])),
                        "why_flagged": f"Platform {nm} in {ym} = {M.loc[ym, col]:.1%} vs trailing 3-month mean {mu[ym]:.1%} (z={z[ym]:.1f}).",
                        "possible_explanation": "Seasonal peaks, logistics disruptions or events (hypothesis only; cannot be confirmed from this dataset).",
                        "recommended_investigation": f"Break {ym} down by region, carrier stage and seller.", "caveat": CAVEAT})
    df = pd.DataFrame(out)
    m = df.p_adj.notna()   # evidence-based score: stronger FDR-adjusted evidence -> higher (log scale, capped at 100)
    df.loc[m, "score"] = np.minimum(100, 20 + 4 * (-np.log10(df.loc[m, "p_adj"].astype(float) + 1e-15)))
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "anomaly_id", [f"AN-{i+1:04d}" for i in range(len(df))])
    df.to_csv(V2_DATA / "anomalies.csv", index=False)
    summary = {"n_anomalies": len(df), "by_detector": df.detector.value_counts().to_dict(),
               "n_sellers_flagged": int(df[df.entity_type == "seller"].entity.nunique()),
               "baselines": {"late": float(base_late), "neg": float(base_neg), "seller_complaint": float(base_comp), "freight_extreme": float(base_fr)},
               "params": {"min_orders_per_seller": MIN_N, "alpha_fdr": ALPHA, "isolation_forest_top_pct": 3},
               "validation_note": "No ground-truth anomaly labels exist, so detectors are NOT scored for accuracy. Evidence is statistical (FDR-controlled tests) and the Isolation Forest is a ranking only. A planted-anomaly recovery test is in tests/test_v2_anomaly.py."}
    save_json("anomaly_summary.json", summary)
    return df, summary


if __name__ == "__main__":
    df, s = run()
    print(s)
    print(df.head(8)[["anomaly_id", "entity", "detector", "period", "measured", "expected", "score"]].to_string())
