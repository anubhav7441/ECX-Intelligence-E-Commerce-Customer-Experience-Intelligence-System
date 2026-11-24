"""Grounded data tools. Every number the agents/assistant report comes from these functions (pandas/scipy), never from an LLM.
Each tool returns a dict with 'source' (where the data came from) so answers can cite it."""
import re
import numpy as np
import pandas as pd
from scipy import stats
from ecx2.common import V2_DATA, nlp_flags, orders

CATS = ["late_delivery", "damaged_product", "wrong_product", "missing_product", "product_quality", "seller_issue", "packaging", "shipping", "payment", "other"]
PLATFORM_SRC = "data/processed/orders_with_history.parquet (95,824 delivered, reviewed orders)"


def _d():
    d = orders()
    nl = nlp_flags()
    if nl is not None and "cat_late_delivery" not in d.columns:
        d = d.merge(nl[[c for c in nl.columns if c.startswith("cat_")] + ["order_id"]], on="order_id", how="left")
    return d


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def resolve_seller(text):
    """Accept full id or an unambiguous prefix (>=6 hex chars)."""
    t = (text or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{6,32}", t):
        return None, "Seller id must be 6-32 hex characters (a full id or an unambiguous prefix)."
    ids = _d().primary_seller_id.dropna().unique()
    m = [i for i in ids if i.startswith(t)]
    if not m:
        return None, f"No seller found with id starting '{t}'."
    if len(m) > 1:
        return None, f"Prefix '{t}' matches {len(m)} sellers; give more characters."
    return m[0], None


def platform_summary():
    d = _d()
    return {"source": PLATFORM_SRC, "n_orders": int(len(d)), "late_rate": float(d.is_late.mean()), "dissatisfied_rate": float(d.dissatisfied.mean()),
            "mean_review_score": float(d.review_score.mean()), "mean_delivery_days": float(d.delivery_days.mean()),
            "mean_estimated_days": float(d.estimated_days.mean()), "n_sellers": int(d.primary_seller_id.nunique()),
            "period": [str(d.order_purchase_timestamp.min())[:10], str(d.order_purchase_timestamp.max())[:10]]}


def seller_profile(seller_id):
    sid, err = resolve_seller(seller_id)
    if err:
        return {"ok": False, "error": err}
    d = _d(); x = d[d.primary_seller_id == sid]; n = len(x); P = platform_summary()
    late_k, neg_k = int(x.is_late.sum()), int(x.dissatisfied.sum())
    monthly = x.groupby("purchase_ym").agg(n=("order_id", "size"), late=("is_late", "mean"), neg=("dissatisfied", "mean")).reset_index()
    cat = x.primary_category.value_counts().head(3).to_dict()
    return {"ok": True, "source": PLATFORM_SRC, "seller_id": sid, "n_orders": n,
            "late_rate": late_k / n, "late_ci95": wilson(late_k, n), "dissatisfied_rate": neg_k / n, "dissatisfied_ci95": wilson(neg_k, n),
            "mean_review_score": float(x.review_score.mean()), "mean_delivery_days": float(x.delivery_days.mean()),
            "mean_handling_days": float(x.seller_handling_days.mean()), "mean_carrier_days": float(x.carrier_transit_days.mean()),
            "mean_freight_ratio": float(x.freight_ratio.mean()), "seller_state": x.seller_state.mode().iat[0] if x.seller_state.notna().any() else None,
            "top_categories": cat, "monthly": monthly.round(4).to_dict("records"),
            "platform": {"late_rate": P["late_rate"], "dissatisfied_rate": P["dissatisfied_rate"], "mean_handling_days": float(d.seller_handling_days.mean()),
                         "mean_carrier_days": float(d.carrier_transit_days.mean()), "mean_delivery_days": P["mean_delivery_days"]},
            "small_sample": n < 30}


def binom_vs_platform(k, n, base):
    p = stats.binomtest(int(k), int(n), base, alternative="two-sided").pvalue
    return {"k": int(k), "n": int(n), "observed": k / n if n else None, "platform": base, "p_value": float(p), "wilson_ci95": wilson(k, n)}


def seller_complaints(seller_id):
    sid, err = resolve_seller(seller_id)
    if err:
        return {"ok": False, "error": err}
    d = _d(); x = d[d.primary_seller_id == sid]
    cols = [f"cat_{c}" for c in CATS if f"cat_{c}" in d.columns]
    if not cols:
        return {"ok": False, "error": "NLP flags not available"}
    rows = []
    for c in cols:
        if c == "cat_other":
            continue
        k, n = int(x[c].fillna(0).sum()), len(x); base = float(d[c].fillna(0).mean())
        t = binom_vs_platform(k, n, base)
        rows.append({"category": c[4:], "seller_share": k / n, "platform_share": base, "count": k, "p_value": t["p_value"]})
    rows.sort(key=lambda r: -r["seller_share"])
    return {"ok": True, "source": "data/processed/review_nlp.parquet + orders_with_history.parquet", "seller_id": sid, "n_reviews": len(x), "categories": rows[:6]}


def sample_reviews(seller_id, k=3, negative=True):
    sid, err = resolve_seller(seller_id)
    if err:
        return {"ok": False, "error": err}
    x = _d(); x = x[(x.primary_seller_id == sid) & x.review_comment_message.notna()]
    x = x[x.dissatisfied == 1] if negative else x
    x = x.sort_values("review_creation_date", ascending=False).head(k)
    return {"ok": True, "source": PLATFORM_SRC, "reviews": [{"order_id": r.order_id, "score": int(r.review_score), "text": str(r.review_comment_message)[:240]} for r in x.itertuples()]}


def seller_late_vs_dissat(seller_id):
    """Within one seller: dissatisfaction late vs on-time (Fisher exact) + category-mix adjusted expectation."""
    sid, err = resolve_seller(seller_id)
    if err:
        return {"ok": False, "error": err}
    d = _d(); x = d[d.primary_seller_id == sid]
    a = int(((x.is_late == 1) & (x.dissatisfied == 1)).sum()); b = int(((x.is_late == 1) & (x.dissatisfied == 0)).sum())
    c = int(((x.is_late == 0) & (x.dissatisfied == 1)).sum()); e = int(((x.is_late == 0) & (x.dissatisfied == 0)).sum())
    out = {"ok": True, "source": PLATFORM_SRC, "seller_id": sid, "n_late": a + b, "n_ontime": c + e,
           "dissat_if_late": a / (a + b) if a + b else None, "dissat_if_ontime": c / (c + e) if c + e else None}
    if a + b >= 5 and c + e >= 5:
        orr, p = stats.fisher_exact([[a, b], [c, e]])
        out.update({"fisher_p": float(p), "odds_ratio": float(orr)})
    # indirect standardisation by category: expected dissatisfaction given the seller's category mix
    cat_rate = d.groupby("primary_category").dissatisfied.mean()
    exp = float(x.primary_category.map(cat_rate).mean()); obs = float(x.dissatisfied.mean())
    out["dissat_observed"] = obs; out["dissat_expected_given_category_mix"] = exp
    k = int(x.dissatisfied.sum()); out["excess_p_value"] = float(stats.binomtest(k, len(x), exp, alternative="greater").pvalue) if len(x) else None
    return out


def top_sellers(metric="late_rate", n=5, min_orders=30, worst=True):
    d = _d()
    g = d.groupby("primary_seller_id").agg(n=("order_id", "size"), late_rate=("is_late", "mean"), dissatisfied_rate=("dissatisfied", "mean"),
                                           mean_review_score=("review_score", "mean"), mean_delivery_days=("delivery_days", "mean"))
    g = g[g.n >= min_orders]
    if metric not in g.columns:
        return {"ok": False, "error": f"metric must be one of {list(g.columns[1:])}"}
    g = g.sort_values(metric, ascending=not worst if metric != "mean_review_score" else worst).head(n)
    return {"ok": True, "source": PLATFORM_SRC, "metric": metric, "min_orders": min_orders,
            "rows": [{"seller_id": i, **{k: float(v) for k, v in r.items()}} for i, r in g.iterrows()]}


def anomalies(entity=None, detector=None, limit=10):
    p = V2_DATA / "anomalies.csv"
    if not p.exists():
        return {"ok": False, "error": "Anomalies not computed - run: python -m ecx2.anomaly.detect"}
    a = pd.read_csv(p)
    if entity:
        a = a[a.entity.str.startswith(entity)]
    if detector:
        a = a[a.detector == detector]
    return {"ok": True, "source": "data/processed/v2/anomalies.csv", "n": int(len(a)), "rows": a.head(limit).fillna("").to_dict("records")}


def group_late_dissat(by):
    """dissatisfaction by group for state/category (n>=100)."""
    if by not in ("customer_state", "primary_category"):
        return {"ok": False, "error": "by must be customer_state or primary_category"}
    d = _d()
    g = d.groupby(by).agg(n=("order_id", "size"), late_rate=("is_late", "mean"), dissatisfied_rate=("dissatisfied", "mean")).query("n>=100")
    return {"ok": True, "source": PLATFORM_SRC, "rows": g.sort_values("dissatisfied_rate", ascending=False).head(8).reset_index().round(4).to_dict("records")}


# ---------------- generic association test (used by the Research assistant; numbers come ONLY from here)
EXPOSURES = {
    "late delivery": ("is_late", "binary"), "multi-item order": ("multi_item", "binary"), "same-state delivery": ("same_state", "binary"),
    "high freight ratio": ("high_freight", "binary"), "long distance": ("long_distance", "binary"), "delivery days": ("delivery_days", "numeric"),
    "estimated days": ("estimated_days", "numeric"), "distance": ("distance_km", "numeric"), "freight ratio": ("freight_ratio", "numeric"),
    "price": ("items_price", "numeric"), "seller handling days": ("seller_handling_days", "numeric"),
}
OUTCOMES = {"dissatisfaction": "dissatisfied", "late delivery": "is_late", "review score": "review_score"}


def _prep():
    d = _d().copy()
    d["multi_item"] = (d.n_items > 1).astype(int)
    d["high_freight"] = (d.freight_ratio > d.freight_ratio.quantile(.75)).astype(int)
    d["long_distance"] = (d.distance_km > d.distance_km.quantile(.75)).astype(int)
    return d


def test_association(exposure, outcome, adjust=True):
    if exposure not in EXPOSURES:
        return {"ok": False, "error": f"Unsupported exposure '{exposure}'. Supported: {sorted(EXPOSURES)}"}
    if outcome not in OUTCOMES:
        return {"ok": False, "error": f"Unsupported outcome '{outcome}'. Supported: {sorted(OUTCOMES)}"}
    xcol, kind = EXPOSURES[exposure]; ycol = OUTCOMES[outcome]
    if xcol == ycol:
        return {"ok": False, "error": "Exposure and outcome are the same variable."}
    d = _prep().dropna(subset=[xcol, ycol])
    out = {"ok": True, "source": PLATFORM_SRC, "exposure": exposure, "outcome": outcome, "n": int(len(d)), "method": []}
    y = d[ycol]
    if kind == "binary" and ycol != "review_score":
        a = int(((d[xcol] == 1) & (y == 1)).sum()); b = int(((d[xcol] == 1) & (y == 0)).sum())
        c = int(((d[xcol] == 0) & (y == 1)).sum()); e = int(((d[xcol] == 0) & (y == 0)).sum())
        chi = stats.chi2_contingency([[a, b], [c, e]])
        orr = (a + .5) * (e + .5) / ((b + .5) * (c + .5)); se = np.sqrt(sum(1 / (v + .5) for v in (a, b, c, e)))
        out.update({"rate_exposed": a / (a + b), "rate_unexposed": c / (c + e), "n_exposed": a + b, "odds_ratio": float(orr),
                    "or_ci95": [float(np.exp(np.log(orr) - 1.96 * se)), float(np.exp(np.log(orr) + 1.96 * se))],
                    "p_value": float(chi[1]), "effect_size": float(np.sqrt(chi[0] / len(d)))})
        out["method"].append("chi-square test, odds ratio (Woolf CI), phi effect size")
    elif kind == "binary":
        g1, g0 = y[d[xcol] == 1], y[d[xcol] == 0]
        u = stats.mannwhitneyu(g1, g0); delta = 2 * u.statistic / (len(g1) * len(g0)) - 1
        out.update({"mean_exposed": float(g1.mean()), "mean_unexposed": float(g0.mean()), "p_value": float(u.pvalue), "effect_size": float(delta)})
        out["method"].append("Mann-Whitney U, Cliff's delta")
    else:
        r, p = stats.spearmanr(d[xcol], y)
        out.update({"spearman_rho": float(r), "p_value": float(p), "effect_size": float(r)})
        out["method"].append("Spearman rank correlation")
    es = abs(out["effect_size"])
    p = out["p_value"]
    if p >= 0.05:
        v = "Not supported (no statistically significant association)"
    elif es < 0.05:
        v = "Statistically significant but negligible effect"
    elif es < 0.2:
        v = "Supported - small effect"
    elif es < 0.4:
        v = "Supported - moderate effect"
    else:
        v = "Supported - large effect"
    out["verdict"] = v
    # adjusted logistic regression (binary outcome only)
    if adjust and ycol in ("dissatisfied", "is_late"):
        try:
            import statsmodels.api as sm
            cov = [c for c in ["distance_km", "freight_ratio", "n_items", "estimated_days", "items_price"] if c != xcol]
            if ycol == "dissatisfied" and xcol not in ("is_late", "delivery_days"):
                cov.append("is_late")
            dd = d.dropna(subset=[xcol] + cov + [ycol])
            Z = dd[[xcol] + cov].astype(float)
            for c in cov + ([xcol] if kind == "numeric" else []):
                Z[c] = (Z[c] - Z[c].mean()) / (Z[c].std() or 1)
            fit = sm.Logit(dd[ycol].astype(float), sm.add_constant(Z)).fit(disp=0, maxiter=200)
            lo, hi = fit.conf_int().loc[xcol]
            out["adjusted"] = {"odds_ratio": float(np.exp(fit.params[xcol])), "ci95": [float(np.exp(lo)), float(np.exp(hi))], "p_value": float(fit.pvalues[xcol]),
                               "controls": cov, "note": "per 1 SD for numeric exposure" if kind == "numeric" else "binary exposure"}
        except Exception as ex:  # noqa
            out["adjusted"] = {"error": f"adjusted model failed: {ex.__class__.__name__}"}
    out["caveats"] = ["Observational data: association, not causation.", "Analysed set = delivered, reviewed orders (selection bias).",
                      "Large samples make tiny effects significant - read the effect size."]
    return out


def month_profile(ym):
    if not re.fullmatch(r"\d{4}-\d{2}", str(ym or "")):
        return {"ok": False, "error": "Month must look like 2017-11."}
    d = _d(); m = d[d.purchase_ym == ym]; rest = d[d.purchase_ym != ym]
    if len(m) < 50:
        return {"ok": False, "error": f"Only {len(m)} analysed orders in {ym}; need at least 50. Available months: {sorted(d.purchase_ym.unique())[:3]} ... {sorted(d.purchase_ym.unique())[-1]}"}
    k, n = int(m.is_late.sum()), len(m)
    tab = [[k, n - k], [int(rest.is_late.sum()), len(rest) - int(rest.is_late.sum())]]
    p_late = float(stats.chi2_contingency(tab)[1])
    kn = int(m.dissatisfied.sum())
    tab2 = [[kn, n - kn], [int(rest.dissatisfied.sum()), len(rest) - int(rest.dissatisfied.sum())]]
    st = m.groupby("customer_state").agg(n=("order_id", "size"), late=("is_late", "mean")).query("n>=30").sort_values("late", ascending=False).head(3)
    return {"ok": True, "source": PLATFORM_SRC, "month": ym, "n_orders": n, "late_rate": k / n, "late_rate_other_months": float(rest.is_late.mean()),
            "late_p_value": p_late, "dissatisfied_rate": kn / n, "dissatisfied_rate_other_months": float(rest.dissatisfied.mean()),
            "dissatisfied_p_value": float(stats.chi2_contingency(tab2)[1]), "worst_states_by_late_rate": st.reset_index().round(4).to_dict("records"),
            "mean_delivery_days": float(m.delivery_days.mean()), "mean_estimated_days": float(m.estimated_days.mean())}
