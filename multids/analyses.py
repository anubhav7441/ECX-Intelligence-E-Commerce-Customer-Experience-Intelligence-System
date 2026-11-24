"""Analyses. Each takes the standardised frame (canonical role columns) and returns a result dict:
   kpis, tables, charts, findings[{type: observed|association|suggestion, text}], assumptions, notes.
EVERY number comes from computation in this file. Wording separates observed facts, statistical associations and suggestions."""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
SEED = 42


def _fig(path, w=7.5, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", lw=.6)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def F(t, text):
    return {"type": t, "text": text}


def _pfmt(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _money(x):
    return f"{x:,.2f}"


def _month(s):
    return s.dt.to_period("M").astype(str)


def _orders(std):
    return std.order_id.nunique() if "order_id" in std else len(std)


# --------------------------------------------------------------------------- 1. sales
def sales_overview(std, out):
    d = std[std.revenue.notna()].copy()
    res = {"kpis": [], "tables": {}, "charts": [], "findings": [], "assumptions": [], "notes": []}
    pos, neg = d[d.revenue > 0], d[d.revenue < 0]
    gross, returns = float(pos.revenue.sum()), float(neg.revenue.sum())
    net = gross + returns
    res["kpis"] += [{"label": "Net revenue", "value": _money(net)}, {"label": "Gross sales (positive lines)", "value": _money(gross)},
                    {"label": "Returns/credits (negative lines)", "value": _money(returns)}, {"label": "Order lines analysed", "value": f"{len(d):,}"}]
    if "order_id" in d:
        n_o = int(pos.order_id.nunique())
        res["kpis"].append({"label": "Orders with positive sales", "value": f"{n_o:,}"})
        aov = float(pos.groupby("order_id").revenue.sum().mean())
        res["kpis"].append({"label": "Average order value", "value": _money(aov)})
        res["findings"].append(F("observed", f"{n_o:,} orders had positive sales; the average order value was {_money(aov)}."))
    if "customer_id" in d:
        res["kpis"].append({"label": "Distinct customers", "value": f"{d.customer_id.nunique():,}"})
    res["findings"].append(F("observed", f"Gross sales were {_money(gross)} and returns/credits {_money(returns)} ({abs(returns) / gross:.1%} of gross), giving net revenue {_money(net)}." if gross else "No positive sales lines were found."))
    dd = d[d.date.notna()]
    if len(dd):
        m = dd.groupby(_month(dd.date)).revenue.sum().rename("net_revenue").reset_index().rename(columns={"date": "month"})
        res["tables"]["monthly_revenue"] = m.round(2).to_dict("records")
        res["kpis"].append({"label": "Period", "value": f"{dd.date.min():%Y-%m-%d} to {dd.date.max():%Y-%m-%d}"})
        if len(m) >= 3:
            fig, ax = _fig(None)
            ax.plot(m.month, m.net_revenue, color=BLUE, lw=2, marker="o", ms=3)
            ax.set_title("Net revenue by month"); ax.set_ylabel("net revenue")
            ax.set_xticks(range(0, len(m), max(1, len(m) // 10))); ax.set_xticklabels(m.month.iloc[::max(1, len(m) // 10)], rotation=45, ha="right", fontsize=7)
            _save(fig, out / "sales_monthly.png"); res["charts"].append({"file": "sales_monthly.png", "caption": "Net revenue by month (observed)."})
            best, worst = m.loc[m.net_revenue.idxmax()], m.loc[m.net_revenue.idxmin()]
            res["findings"].append(F("observed", f"Highest month: {best.month} ({_money(best.net_revenue)}); lowest month: {worst.month} ({_money(worst.net_revenue)})."))
            if len(m) >= 6:
                rho, p = stats.spearmanr(np.arange(len(m)), m.net_revenue)
                res["findings"].append(F("association", (f"Monthly revenue and time are {'positively' if rho > 0 else 'negatively'} rank-correlated (Spearman rho={rho:.2f}, {_pfmt(p)}, {len(m)} months). This describes the trend, not its cause." if p < 0.05 else f"No statistically significant monthly revenue trend was detected (Spearman rho={rho:.2f}, {_pfmt(p)}, {len(m)} months).")))
                res["assumptions"].append("Partial first/last months may bias the trend.")
        else:
            res["notes"].append("Fewer than 3 months of data; no trend chart.")
        mm = dd.assign(dow=dd.date.dt.day_name()).groupby("dow").revenue.sum()
        if len(mm) > 1:
            res["findings"].append(F("observed", f"Day of week with most revenue: {mm.idxmax()} ({mm.max() / mm.sum():.1%} of the total)."))
    else:
        res["notes"].append("No valid dates, so time trends were skipped.")
    if neg.shape[0]:
        res["findings"].append(F("suggestion", "Consider reviewing the reasons for returns/credits (hypothesis: product or fulfilment issues); the data alone cannot say why they occurred."))
    res["assumptions"] += ["Negative revenue lines are treated as returns/credits.", "Revenue is in the dataset's own currency; no currency conversion."]
    return res


# --------------------------------------------------------------------------- 2. products
def product_analysis(std, out):
    key = "product_id" if "product_id" in std else "product_name"
    d = std[std[key].notna()].copy()
    measure = "revenue" if "revenue" in d else "quantity"
    d = d[d[measure].notna()]
    g = d.groupby(key)[measure].sum().sort_values(ascending=False)
    gp = g[g > 0]
    res = {"kpis": [{"label": "Distinct products", "value": f"{g.size:,}"}], "tables": {}, "charts": [], "findings": [], "assumptions": [], "notes": []}
    top = gp.head(10)
    names = None
    if key == "product_id" and "product_name" in d:
        names = d.groupby("product_id").product_name.agg(lambda s: s.dropna().mode().iat[0] if s.notna().any() else "").to_dict()
    res["tables"]["top_products"] = [{"product": k, "name": (names or {}).get(k, ""), measure: round(float(v), 2), "share_pct": round(float(v / gp.sum() * 100), 2)} for k, v in top.items()]
    if len(gp):
        cum = gp.cumsum() / gp.sum()
        n80 = int((cum < 0.8).sum() + 1)
        res["kpis"].append({"label": f"Products making 80% of {measure}", "value": f"{n80:,} ({n80 / gp.size:.1%})"})
        res["findings"].append(F("observed", f"The top 10 products account for {top.sum() / gp.sum():.1%} of positive {measure}; {n80:,} products ({n80 / gp.size:.1%} of all) make 80% of it."))
        fig, ax = _fig(None, 7.5, 3.8)
        labs = [((names or {}).get(k) or str(k))[:28] for k in top.index][::-1]
        ax.barh(labs, top.values[::-1], color=BLUE); ax.set_title(f"Top 10 products by {measure}"); ax.grid(axis="x", color="#e1e0d9"); ax.grid(axis="y", visible=False)
        _save(fig, out / "top_products.png"); res["charts"].append({"file": "top_products.png", "caption": f"Top 10 products by {measure} (observed)."})
        if n80 / gp.size < 0.2:
            res["findings"].append(F("suggestion", "Sales are concentrated in a small share of products; consider protecting stock availability for these items (hypothesis, not tested)."))
    neg = g[g < 0]
    if len(neg):
        res["findings"].append(F("observed", f"{len(neg):,} products have a negative net {measure} (returns exceed sales)."))
    res["assumptions"].append("Products are grouped by the mapped product column; different codes for the same item are treated as different products.")
    return res


# --------------------------------------------------------------------------- 3. RFM
def customer_segmentation(std, out):
    d = std[std.customer_id.notna() & std.date.notna() & std.revenue.notna()].copy()
    miss = int(len(std) - len(d))
    d = d[d.revenue > 0]
    res = {"kpis": [], "tables": {}, "charts": [], "findings": [], "assumptions": [], "notes": []}
    if d.customer_id.nunique() < 20:
        return {**res, "notes": ["Fewer than 20 customers with valid ID, date and positive revenue; segmentation skipped."], "skipped": True}
    now = d.date.max() + pd.Timedelta(days=1)
    key = "order_id" if "order_id" in d else None
    g = d.groupby("customer_id").agg(last=("date", "max"), monetary=("revenue", "sum"), frequency=(key, "nunique") if key else ("revenue", "size"))
    g["recency"] = (now - g["last"]).dt.days
    q = lambda s, asc: pd.qcut(s.rank(method="first", ascending=asc), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    g["R"], g["F"], g["M"] = q(g.recency, False), q(g.frequency, True), q(g.monetary, True)

    def seg(r):
        if r.R >= 4 and r.F >= 4: return "Champions"
        if r.F >= 4 and r.R >= 2: return "Loyal"
        if r.R >= 4 and r.F <= 2: return "Recent / new"
        if r.R <= 2 and r.F >= 3: return "At risk"
        if r.R <= 2 and r.F <= 2: return "Lapsed / low value"
        return "Needs attention"
    g["segment"] = g.apply(seg, axis=1)
    s = g.groupby("segment").agg(customers=("monetary", "size"), revenue=("monetary", "sum"), mean_recency_days=("recency", "mean"), mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean"))
    s["customer_share_pct"] = s.customers / s.customers.sum() * 100; s["revenue_share_pct"] = s.revenue / s.revenue.sum() * 100
    s = s.sort_values("revenue", ascending=False).round(2)
    res["tables"]["rfm_segments"] = s.reset_index().to_dict("records")
    res["kpis"] += [{"label": "Customers segmented", "value": f"{len(g):,}"}, {"label": "Repeat customers (2+ orders)", "value": f"{(g.frequency >= 2).mean():.1%}"},
                    {"label": "Median days since last purchase", "value": f"{g.recency.median():.0f}"}]
    top10 = g.monetary.sort_values(ascending=False).head(max(1, len(g) // 10)).sum() / g.monetary.sum()
    res["findings"].append(F("observed", f"{len(g):,} customers were segmented; {(g.frequency >= 2).mean():.1%} placed 2+ orders. The top 10% of customers generate {top10:.1%} of revenue."))
    b = s.iloc[0]
    res["findings"].append(F("observed", f"Largest revenue segment: {s.index[0]} ({b.customer_share_pct:.1f}% of customers, {b.revenue_share_pct:.1f}% of revenue)."))
    fig, ax = _fig(None)
    ax.bar(s.index, s.revenue_share_pct, color=BLUE); ax.set_title("Revenue share by RFM segment (%)"); plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    _save(fig, out / "rfm.png"); res["charts"].append({"file": "rfm.png", "caption": "Revenue share by RFM segment (observed)."})
    if "At risk" in s.index:
        res["findings"].append(F("suggestion", f"'At risk' customers ({int(s.loc['At risk'].customers):,}) bought often but not recently; a win-back campaign could be tested (hypothesis, effect unknown)."))
    res["assumptions"] += ["RFM scores are quintile ranks within this dataset; segment rules are heuristic.", "Reference date = last date in the data + 1 day.", f"{miss:,} lines without customer ID/date/revenue were excluded; only positive-revenue lines were used."]
    g.reset_index()[["customer_id", "recency", "frequency", "monetary", "R", "F", "M", "segment"]].to_csv(out / "rfm_customers.csv", index=False)
    res["exports"] = ["rfm_customers.csv"]
    return res


# --------------------------------------------------------------------------- 4. geography
def geographic_analysis(std, out):
    d = std[std.region.notna()].copy()
    res = {"kpis": [{"label": "Regions", "value": f"{d.region.nunique():,}"}], "tables": {}, "charts": [], "findings": [], "assumptions": [], "notes": []}
    agg = {"lines": ("region", "size")}
    if "order_id" in d: agg["orders"] = ("order_id", "nunique")
    if "revenue" in d: agg["revenue"] = ("revenue", "sum")
    if "rating" in d: agg["mean_rating"] = ("rating", "mean")
    g = d.groupby("region").agg(**agg)
    sortcol = "revenue" if "revenue" in g else "orders" if "orders" in g else "lines"
    g = g.sort_values(sortcol, ascending=False)
    res["tables"]["regions"] = g.head(25).round(3).reset_index().to_dict("records")
    tot = g[sortcol].sum()
    if tot > 0:
        res["findings"].append(F("observed", f"The top region ({g.index[0]}) accounts for {g.iloc[0][sortcol] / tot:.1%} of {sortcol}; the top 3 account for {g[sortcol].head(3).sum() / tot:.1%}."))
    fig, ax = _fig(None)
    t = g[sortcol].head(10)
    ax.bar([str(i) for i in t.index], t.values, color=BLUE); ax.set_title(f"Top regions by {sortcol}"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    _save(fig, out / "geo_top.png"); res["charts"].append({"file": "geo_top.png", "caption": f"Top regions by {sortcol} (observed)."})
    if "rating" in d:
        big = g[g.lines >= 30].index
        groups = [d[(d.region == r)].rating.dropna().values for r in big]
        groups = [x for x in groups if len(x) >= 30]
        if len(groups) >= 3:
            h, p = stats.kruskal(*groups)
            n = sum(len(x) for x in groups); eps = (h - len(groups) + 1) / (n - len(groups))
            ms = g.loc[big].mean_rating
            res["findings"].append(F("association", f"Ratings differ between regions (Kruskal-Wallis {_pfmt(p)}, effect size epsilon-squared={max(eps, 0):.3f}, {len(groups)} regions with 30+ ratings); highest mean {ms.idxmax()} ({ms.max():.2f}), lowest {ms.idxmin()} ({ms.min():.2f}). A small effect size means the difference is small even if significant."))
    res["assumptions"].append("Regions with few records give unstable percentages.")
    return res


# --------------------------------------------------------------------------- 5. reviews
def review_analysis(std, out):
    from multids import textnlp
    res = {"kpis": [], "tables": {}, "charts": [], "findings": [], "assumptions": [], "notes": []}
    n = len(std)
    if "rating" in std:
        r = std.rating.dropna()
        scale = 5 if r.max() <= 5 else 10
        low, high = (r <= 2).mean() if scale == 5 else (r <= 4).mean(), (r >= 4).mean() if scale == 5 else (r >= 8).mean()
        dist = r.round().value_counts().sort_index()
        res["kpis"] += [{"label": "Ratings", "value": f"{len(r):,}"}, {"label": "Mean rating", "value": f"{r.mean():.2f}"}, {"label": "Low ratings", "value": f"{low:.1%}"}, {"label": "High ratings", "value": f"{high:.1%}"}]
        res["tables"]["rating_distribution"] = [{"rating": float(k), "count": int(v), "pct": round(float(v / len(r) * 100), 2)} for k, v in dist.items()]
        res["findings"].append(F("observed", f"Mean rating {r.mean():.2f} (median {r.median():.0f}) over {len(r):,} ratings; {low:.1%} are low and {high:.1%} high."))
        fig, ax = _fig(None)
        ax.bar([str(int(k)) for k in dist.index], dist.values, color=BLUE); ax.set_title("Rating distribution"); ax.set_xlabel("rating")
        _save(fig, out / "rating_dist.png"); res["charts"].append({"file": "rating_dist.png", "caption": "Rating distribution (observed)."})
        if "date" in std and std.date.notna().sum() > 50:
            m = std[std.date.notna() & std.rating.notna()].groupby(_month(std.date.dropna())).rating.agg(["mean", "size"])
            m = m[m["size"] >= 30]
            if len(m) >= 6:
                rho, p = stats.spearmanr(np.arange(len(m)), m["mean"])
                res["findings"].append(F("association", (f"Monthly mean rating trends {'upward' if rho > 0 else 'downward'} over time (Spearman rho={rho:.2f}, {_pfmt(p)}, {len(m)} months with 30+ ratings)." if p < 0.05 else f"No statistically significant trend in monthly mean rating was detected (Spearman rho={rho:.2f}, {_pfmt(p)}, {len(m)} months with 30+ ratings).")))
    if "review_text" in std:
        t = std[std.review_text.notna()].copy()
        t["review_text"] = t.review_text.astype(str)
        t = t[t.review_text.str.strip().str.len() >= 3]
        cap = 30000
        if len(t) > cap:
            t = t.sample(cap, random_state=SEED); res["notes"].append(f"Text analysis used a random sample of {cap:,} reviews (seed {SEED}).")
        res["kpis"].append({"label": "Reviews with text analysed", "value": f"{len(t):,}"})
        if len(t) < 30:
            res["notes"].append("Fewer than 30 reviews with text; text analysis skipped.")
        else:
            lang, counts = textnlp.detect_lang(t.review_text.tolist())
            res["findings"].append(F("observed", f"Detected review language (stop-word heuristic on 300 reviews): {lang} ({counts})."))
            if lang == "pt":
                from ml.sentiment_analysis import review_nlp as v1
                norm = t.review_text.map(v1.normalize)
                t["sent"] = norm.map(v1.overall_lexicon_sentiment).map({"positive": 1.0, "negative": -1.0}).fillna(0.0)
                t["cats"] = norm.map(lambda x: [c for c in v1.complaint_categories(x) if c != "other"])
                t["lab"] = t.sent.map(lambda s: "positive" if s > 0 else "negative" if s < 0 else "neutral")
                res["assumptions"].append("Portuguese reviews analysed with the v1 rule-based taxonomy (micro-F1 about 0.66 on 150 reviews labelled by one AI annotator).")
            else:
                t["sent"] = t.review_text.map(textnlp.lex_score)
                t["lab"] = t.sent.map(textnlp.label)
                t["cats"] = t.review_text.map(textnlp.en_categories)
                if lang != "en":
                    res["notes"].append("Language not confidently English; rule-based English results may be unreliable.")
                res["assumptions"].append("English reviews analysed with a small transparent lexicon and keyword taxonomy; not validated against human labels.")
            sd = t.lab.value_counts(normalize=True)
            res["tables"]["sentiment_share"] = [{"sentiment": k, "pct": round(float(v * 100), 2)} for k, v in sd.items()]
            res["findings"].append(F("observed", "Rule-based sentiment of review text: " + ", ".join(f"{k} {v:.1%}" for k, v in sd.items()) + "."))
            if "rating" in t and t.rating.notna().sum() > 30:
                k = t.dropna(subset=["rating"])
                rho, p = stats.spearmanr(k.sent, k.rating)
                res["findings"].append(F("association", f"Text sentiment score and rating show {'strong' if abs(rho) >= .5 else 'moderate' if abs(rho) >= .3 else 'weak'} rank agreement (Spearman rho={rho:.2f}, {_pfmt(p)}, n={len(k):,}); this is the measured agreement of the rule-based sentiment with ratings, not an accuracy claim."))
            allc = t.cats.explode().dropna()
            cc = allc.value_counts()
            if len(cc):
                tab = []
                for c, v in cc.items():
                    row = {"category": c, "reviews": int(v), "pct_of_reviews": round(float(v / len(t) * 100), 2)}
                    if "rating" in t and t.rating.notna().sum() > 30:
                        has = t.cats.map(lambda x: c in x)
                        a, b = t[has].rating.dropna(), t[~has].rating.dropna()
                        if len(a) >= 10 and len(b) >= 10:
                            u = stats.mannwhitneyu(a, b); row["mean_rating_with"] = round(float(a.mean()), 2); row["mean_rating_without"] = round(float(b.mean()), 2); row["p_value"] = float(u.pvalue)
                    tab.append(row)
                res["tables"]["complaint_categories"] = tab
                top = tab[0]
                res["findings"].append(F("observed", f"Most frequent complaint category: '{top['category']}' in {top['reviews']:,} reviews ({top['pct_of_reviews']:.1f}% of reviews with text)."))
                if "mean_rating_with" in top:
                    res["findings"].append(F("association", f"Reviews mentioning '{top['category']}' have a mean rating of {top['mean_rating_with']} vs {top['mean_rating_without']} for others ({_pfmt(top['p_value'])}); this is an association from review wording."))
                fig, ax = _fig(None)
                ax.barh([r["category"].replace("_", " ") for r in tab[:8]][::-1], [r["reviews"] for r in tab[:8]][::-1], color=ORANGE); ax.set_title("Complaint categories (reviews mentioning)"); ax.grid(axis="x", color="#e1e0d9"); ax.grid(axis="y", visible=False)
                _save(fig, out / "complaints.png"); res["charts"].append({"file": "complaints.png", "caption": "Complaint categories from review text (observed, rule-based)."})
                res["findings"].append(F("suggestion", f"Reading a sample of '{top['category']}' reviews could confirm the root cause; wording alone does not prove it."))
            if lang != "pt":
                asp = {}
                for txt in t.review_text.head(15000):
                    for k, v in textnlp.en_aspects(txt).items():
                        asp.setdefault(k, []).append(v)
                if asp:
                    at = [{"aspect": k, "mentions": len(v), "mean_sentiment": round(float(np.mean(v)), 3), "pct_negative": round(float(np.mean(np.array(v) < 0) * 100), 1)} for k, v in asp.items()]
                    at.sort(key=lambda r: r["mean_sentiment"])
                    res["tables"]["aspect_sentiment"] = at
                    res["findings"].append(F("observed", f"Aspect with the most negative average sentiment: '{at[0]['aspect']}' ({at[0]['mean_sentiment']:.2f}, {at[0]['mentions']:,} mentions); most positive: '{at[-1]['aspect']}' ({at[-1]['mean_sentiment']:.2f})."))
            if "rating" in t and t.rating.notna().sum() > 200 and lang != "pt":
                from sklearn.feature_extraction.text import TfidfVectorizer
                k = t.dropna(subset=["rating"])
                lo, hi = k[k.rating <= 2].review_text, k[k.rating >= 4].review_text
                if len(lo) >= 30 and len(hi) >= 30:
                    try:
                        v = TfidfVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2), sublinear_tf=True)
                        X = v.fit_transform(pd.concat([lo, hi]).astype(str)); names = np.array(v.get_feature_names_out())
                        diff = np.asarray(X[:len(lo)].mean(0)).ravel() - np.asarray(X[len(lo):].mean(0)).ravel()
                        res["tables"]["terms_low_rated"] = [{"term": x} for x in names[np.argsort(-diff)[:15]]]
                        res["findings"].append(F("observed", "Terms most distinctive of low-rated reviews: " + ", ".join(names[np.argsort(-diff)[:8]]) + "."))
                    except ValueError:
                        res["notes"].append("The review text has no repeated words (each word appears in fewer than 5 reviews), so term comparison was skipped.")
    if not res["kpis"]:
        res["notes"].append("No usable rating or text found.")
    res["assumptions"].append("Only customers who wrote reviews are observed (self-selection bias).")
    return res


# --------------------------------------------------------------------------- 6. delivery
def delivery_analysis(std, out):
    res = {"kpis": [], "tables": {}, "charts": [], "findings": [], "assumptions": [], "notes": []}
    dd = std.delivery_days.dropna()
    bad = int((dd < 0).sum())
    dd = dd[dd >= 0]
    if bad:
        res["notes"].append(f"{bad:,} rows with negative delivery time were excluded.")
    if len(dd) < 20:
        return {**res, "notes": res["notes"] + ["Fewer than 20 valid delivery times; analysis skipped."], "skipped": True}
    res["kpis"] += [{"label": "Orders with delivery time", "value": f"{len(dd):,}"}, {"label": "Mean delivery days", "value": f"{dd.mean():.1f}"},
                    {"label": "Median", "value": f"{dd.median():.1f}"}, {"label": "90th percentile", "value": f"{dd.quantile(.9):.1f}"}]
    res["findings"].append(F("observed", f"Mean delivery time {dd.mean():.1f} days (median {dd.median():.1f}, 90th percentile {dd.quantile(.9):.1f}, max {dd.max():.0f})."))
    fig, ax = _fig(None)
    ax.hist(dd.clip(upper=dd.quantile(.99)), bins=40, color=BLUE); ax.set_title("Delivery time (days, clipped at 99th percentile)")
    _save(fig, out / "delivery_hist.png"); res["charts"].append({"file": "delivery_hist.png", "caption": "Delivery time distribution (observed)."})
    if "delay_days" in std and std.delay_days.notna().sum() > 20:
        late = (std.delay_days.dropna() > 0)
        res["kpis"].append({"label": "Late-delivery rate", "value": f"{late.mean():.1%}"})
        res["findings"].append(F("observed", f"{late.mean():.1%} of {len(late):,} orders were delivered after the estimated date."))
        if "date" in std:
            k = std[std.delay_days.notna() & std.date.notna()]
            m = k.groupby(_month(k.date)).delay_days.agg(lambda s: (s > 0).mean()); n = k.groupby(_month(k.date)).size()
            m = m[n >= 30]
            if len(m) >= 3:
                fig, ax = _fig(None)
                ax.plot(m.index, m.values * 100, color=ORANGE, marker="o", ms=3); ax.set_title("Late-delivery rate by month (%)")
                ax.set_xticks(range(0, len(m), max(1, len(m) // 10))); ax.set_xticklabels(m.index[::max(1, len(m) // 10)], rotation=45, ha="right", fontsize=7)
                _save(fig, out / "late_monthly.png"); res["charts"].append({"file": "late_monthly.png", "caption": "Late-delivery rate by month (observed)."})
                res["findings"].append(F("observed", f"Highest monthly late rate: {m.idxmax()} ({m.max():.1%}); lowest: {m.idxmin()} ({m.min():.1%})."))
        if "rating" in std:
            k = std[std.delay_days.notna() & std.rating.notna()]
            a, b = k[k.delay_days > 0].rating, k[k.delay_days <= 0].rating
            if len(a) >= 20 and len(b) >= 20:
                u = stats.mannwhitneyu(a, b); delta = 2 * u.statistic / (len(a) * len(b)) - 1
                res["findings"].append(F("association", f"Late orders have a mean rating of {a.mean():.2f} vs {b.mean():.2f} for on-time orders ({_pfmt(u.pvalue)}, Cliff's delta={delta:.2f}). This is an association; other factors may contribute."))
                if a.mean() < b.mean():
                    res["findings"].append(F("suggestion", "Reducing late deliveries may improve ratings (hypothesis; the data cannot prove the change would have this effect)."))
        if "region" in std:
            k = std[std.delay_days.notna() & std.region.notna()].groupby("region").delay_days.agg(n="size", late=lambda s: (s > 0).mean())
            k = k[k.n >= 30].sort_values("late", ascending=False)
            if len(k):
                res["tables"]["late_rate_by_region"] = k.head(10).round(4).reset_index().to_dict("records")
                res["findings"].append(F("observed", f"Region with the highest late rate (30+ orders): {k.index[0]} ({k.late.iloc[0]:.1%}, n={int(k.n.iloc[0])})."))
    res["assumptions"].append("Only delivered orders are observed; undelivered orders are missing (survivorship).")
    return res


# --------------------------------------------------------------------------- 7. sellers
def seller_analysis(std, out):
    res = {"kpis": [], "tables": {}, "charts": [], "findings": [], "assumptions": [], "notes": []}
    d = std[std.seller_id.notna()]
    agg = {"records": ("seller_id", "size")}
    if "rating" in d: agg["mean_rating"] = ("rating", "mean")
    if "revenue" in d: agg["revenue"] = ("revenue", "sum")
    if "delivery_days" in d: agg["mean_delivery_days"] = ("delivery_days", "mean")
    g = d.groupby("seller_id").agg(**agg)
    big = g[g.records >= 30]
    res["kpis"] += [{"label": "Sellers", "value": f"{len(g):,}"}, {"label": "Sellers with 30+ records", "value": f"{len(big):,}"}]
    if len(big) < 3:
        return {**res, "notes": ["Fewer than 3 sellers have 30+ records; comparison skipped."], "skipped": True}
    if "mean_rating" in big:
        sd = d.groupby("seller_id").rating.std()
        big = big.assign(ci95=1.96 * sd.reindex(big.index) / np.sqrt(big.records))
        groups = [d[d.seller_id == s].rating.dropna().values for s in big.index[:300]]
        groups = [x for x in groups if len(x) >= 30]
        worst = big.sort_values("mean_rating").head(5); best = big.sort_values("mean_rating", ascending=False).head(5)
        res["tables"]["lowest_rated_sellers"] = worst.round(3).reset_index().to_dict("records")
        res["tables"]["highest_rated_sellers"] = best.round(3).reset_index().to_dict("records")
        if len(groups) >= 3:
            h, p = stats.kruskal(*groups)
            res["findings"].append(F("association", f"Sellers differ in rating (Kruskal-Wallis {_pfmt(p)}, {len(groups)} sellers with 30+ ratings). Lowest mean: {worst.index[0]} ({worst.mean_rating.iloc[0]:.2f} +/- {worst.ci95.iloc[0]:.2f}); highest: {best.index[0]} ({best.mean_rating.iloc[0]:.2f} +/- {best.ci95.iloc[0]:.2f})."))
        fig, ax = _fig(None)
        s = pd.concat([worst, best]).drop_duplicates()
        ax.barh([str(i)[:12] for i in s.index][::-1], s.mean_rating.values[::-1], xerr=s.ci95.values[::-1], color=BLUE); ax.set_title("Lowest and highest rated sellers (mean +/- 95% CI)")
        _save(fig, out / "sellers_rating.png"); res["charts"].append({"file": "sellers_rating.png", "caption": "Seller mean rating with 95% confidence interval (observed)."})
        res["findings"].append(F("suggestion", "Investigate the lowest-rated sellers' reviews and delivery times before any decision; differences may come from product mix or sample size."))
    if "revenue" in big:
        top = big.revenue.sort_values(ascending=False)
        res["findings"].append(F("observed", f"The top 5 sellers account for {top.head(5).sum() / g.revenue.clip(lower=0).sum():.1%} of positive revenue."))
        res["tables"]["top_sellers_by_revenue"] = big.sort_values("revenue", ascending=False).head(10).round(2).reset_index().to_dict("records")
    if "mean_delivery_days" in big:
        w = big.sort_values("mean_delivery_days", ascending=False).iloc[0]
        res["findings"].append(F("observed", f"Slowest seller (30+ records): {big.mean_delivery_days.idxmax()} with a mean of {w.mean_delivery_days:.1f} delivery days vs {d.delivery_days.mean():.1f} overall."))
    res["assumptions"].append("Rankings are descriptive; sellers with few records are excluded (minimum 30).")
    return res


# --------------------------------------------------------------------------- 8. numeric
def numeric_statistics(raw, out, exclude=()):
    from sklearn.ensemble import IsolationForest
    res = {"kpis": [], "tables": {}, "charts": [], "findings": [], "assumptions": [], "notes": []}
    num = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c]) and not pd.api.types.is_bool_dtype(raw[c]) and raw[c].nunique() > 2 and c not in exclude]
    if not num:
        return {**res, "notes": ["No suitable numeric columns."], "skipped": True}
    num = num[:30]
    desc = []
    for c in num:
        s = raw[c].dropna()
        q1, q3 = s.quantile(.25), s.quantile(.75); iqr = q3 - q1
        out_n = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()) if iqr > 0 else 0
        desc.append({"column": c, "n": int(len(s)), "missing_pct": round(float(raw[c].isna().mean() * 100), 2), "mean": round(float(s.mean()), 4), "median": round(float(s.median()), 4),
                     "std": round(float(s.std()), 4), "min": round(float(s.min()), 4), "max": round(float(s.max()), 4), "skew": round(float(stats.skew(s)), 2), "extreme_outliers_3xIQR": out_n})
    res["tables"]["numeric_summary"] = desc
    res["kpis"].append({"label": "Numeric columns analysed", "value": str(len(num))})
    sk = max(desc, key=lambda r: abs(r["skew"]))
    res["findings"].append(F("observed", f"Most skewed column: '{sk['column']}' (skewness {sk['skew']}, mean {sk['mean']} vs median {sk['median']}), so medians describe it better than means."))
    mo = max(desc, key=lambda r: r["extreme_outliers_3xIQR"])
    if mo["extreme_outliers_3xIQR"]:
        res["findings"].append(F("observed", f"'{mo['column']}' has {mo['extreme_outliers_3xIQR']:,} extreme values beyond 3x the interquartile range."))
    sample = raw[num[:12]].dropna()
    if len(sample) > 100_000:
        sample = sample.sample(100_000, random_state=SEED); res["notes"].append("Correlations/anomaly detection used a random sample of 100,000 complete rows.")
    if len(num[:12]) >= 2 and len(sample) >= 30:
        corr = sample.corr(method="spearman")
        pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) >= 0.3]
        pairs.sort(key=lambda x: -abs(x[2]))
        res["tables"]["strong_correlations"] = [{"a": a, "b": b, "spearman_rho": round(float(r), 3)} for a, b, r in pairs[:15]]
        if len(sample) < 100:
            res["notes"].append(f"Only {len(sample)} complete rows were available, so correlations are unstable; treat them as hints only.")
        if not pairs:
            res["findings"].append(F("observed", f"No pair of numeric columns has a clear relationship (all |Spearman rho| below 0.30, n={len(sample):,}). This is a genuine result, not an error."))
        if pairs:
            a, b, r = pairs[0]
            res["findings"].append(F("association", f"Strongest rank correlation: '{a}' and '{b}' (Spearman rho={r:.2f}, n={len(sample):,}). Correlation does not imply causation."))
        fig, ax = _fig(None, 5.5, 4.5)
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1); ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=7); ax.set_yticklabels(corr.columns, fontsize=7); ax.spines[:].set_visible(False); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=.046); ax.set_title("Spearman correlation")
        _save(fig, out / "correlation.png"); res["charts"].append({"file": "correlation.png", "caption": "Spearman correlation matrix (observed)."})
    if len(sample) >= 200 and sample.shape[1] >= 2:
        X = sample.copy()
        Z = (X - X.median()) / (X.quantile(.75) - X.quantile(.25)).replace(0, 1)
        iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=SEED).fit(Z)
        sc = -iso.score_samples(Z)
        k = max(1, int(len(X) * 0.01))
        idx = np.argsort(-sc)[:k]
        an = X.iloc[idx].copy(); an.insert(0, "anomaly_score", sc[idx].round(4)); an.insert(0, "row_index", X.index[idx])
        an.to_csv(out / "anomalies.csv", index=False)
        res["exports"] = ["anomalies.csv"]
        res["kpis"].append({"label": "Anomaly signals (top 1%)", "value": f"{k:,}"})
        res["tables"]["top_anomalies"] = an.head(10).round(3).to_dict("records")
        res["findings"].append(F("observed", f"Isolation Forest ranked the {k:,} most unusual rows (top 1% of {len(X):,}) across {X.shape[1]} numeric columns; they are listed in anomalies.csv."))
        res["findings"].append(F("suggestion", "Inspect the flagged rows for data-entry errors, bulk orders or genuine rare events; an anomaly is an investigation signal, not proof of an error."))
    res["assumptions"].append("Anomaly detection is unsupervised (no labels), so its accuracy cannot be measured.")
    return res


REGISTRY = {"sales_overview": sales_overview, "product_analysis": product_analysis, "customer_segmentation": customer_segmentation,
            "geographic_analysis": geographic_analysis, "review_analysis": review_analysis, "delivery_analysis": delivery_analysis, "seller_analysis": seller_analysis}
