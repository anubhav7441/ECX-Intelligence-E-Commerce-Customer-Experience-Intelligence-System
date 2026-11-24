"""Advice for a user's uploaded dataset. Works from the mapped columns only; every rule states its numbers and fires only if the needed columns exist."""
import numpy as np
import pandas as pd
from scipy import stats
from app.insights.core import card, rank, money
from multids import registry as R


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _season_is_real(rev, date, observed_ratio, n_perm=200, seed=7):
    """Permutation check: would random month labels give swings this large? Protects against advice built from noise."""
    d = pd.DataFrame({"r": rev, "d": date}).dropna()
    d = d[d.r > 0]
    months = d.d.dt.to_period("M")
    labels = months.unique()
    if len(labels) < 6 or len(d) < 60:
        return False
    rng = np.random.default_rng(seed)
    vals = d.r.values
    codes = rng.integers(0, len(labels), size=(n_perm, len(vals)))
    ratios = []
    for row in codes:
        tot = np.bincount(row, weights=vals, minlength=len(labels))
        tot = tot[tot > 0]
        ratios.append(tot.max() / tot.min() if len(tot) > 1 else 1.0)
    return observed_ratio > np.percentile(ratios, 95)


def _winback_is_real(d2, observed, n_perm=40, seed=7):
    """Do real customers go quiet more than if purchase dates were shuffled between rows?"""
    if len(d2) > 200_000:
        return True   # too large to test cheaply; the rule itself still applies
    rng = np.random.default_rng(seed)
    ref = d2.d.max(); dates = d2.d.values.copy(); counts = []
    for _ in range(n_perm):
        rng.shuffle(dates)
        x = pd.DataFrame({"c": d2.c.values, "d": dates})
        last = x.groupby("c").d.max(); freq = x.groupby("c").d.nunique()
        days = (ref - last).dt.days
        counts.append(int(((freq >= 3) & (days > max(60, days.median() * 2))).sum()))
    return observed > np.percentile(counts, 95)


def build(did):
    m = R.meta(did) or {}
    mp = {k: v for k, v in (m.get("mapping") or {}).items() if v}
    if not mp:
        return {"ready": False, "message": "Confirm the column mapping first (Dataset Center, step 3). Advice is built from the columns you identified.", "cards": []}
    df = R.load_df(did)
    if len(df) < 30:
        return {"ready": True, "message": f"Your file has only {len(df)} rows, which is too few to give reliable advice. Upload at least a few hundred rows.", "cards": []}
    cards = []

    def col(role):
        return df[mp[role]] if role in mp and mp[role] in df.columns else None

    date, rev = col("date"), col("revenue")
    qty, price = col("quantity"), col("unit_price")
    if rev is None and qty is not None and price is not None:
        rev = _num(qty) * _num(price)
    elif rev is not None:
        rev = _num(rev)
    prod = col("product_name") if col("product_name") is not None else col("product_id")
    cust, reg, rating, seller = col("customer_id"), col("region"), col("rating"), col("seller_id")
    n = len(df)
    if date is not None:
        date = pd.to_datetime(date, errors="coerce")

    # ---- returns / negative lines
    if rev is not None:
        neg = rev[rev < 0].sum(); pos = rev[rev > 0].sum()
        if pos > 0 and abs(neg) / pos >= 0.03:
            share = abs(neg) / pos
            why = "The data does not say why items were returned."
            ev = [f"Returns/credits: {money(abs(neg))} against {money(pos)} of sales ({share:.1%})"]
            if prod is not None:
                bad = rev[rev < 0].groupby(prod[rev < 0]).sum().sort_values().head(3)
                if len(bad):
                    why = "Most of the returned value comes from: " + ", ".join(f"{k} ({money(abs(v))})" for k, v in bad.items()) + ". Products that are returned often usually have a quality, size or description problem."
                    ev.append("Top returned products listed above")
            cards.append(card("returns", "Returns are taking away revenue", f"About {share:.1%} of sales value is given back as returns or credits.", why,
                              ["Contact customers who returned the top products and ask for the reason.", "Fix product descriptions, photos or size guides for those products.", "Check the supplier or packaging for the products returned most."],
                              ev, "High" if share >= 0.08 else "Medium", int((rev < 0).sum()), "Sales"))
        # ---- product concentration
    if rev is not None and prod is not None:
        g = rev[rev > 0].groupby(prod[rev > 0]).sum().sort_values(ascending=False)
        if len(g) >= 10 and g.sum() > 0:
            top10 = g.head(10).sum() / g.sum(); k80 = int((g.cumsum() / g.sum() < 0.8).sum() + 1)
            if top10 >= 0.5:
                cards.append(card("product_concentration", "Too much of your income depends on a few products",
                                  f"Your top 10 products bring in {top10:.0%} of sales ({k80} of {len(g)} products make 80%).",
                                  "When a few products carry the business, a stock-out, a price rise from the supplier or a new competitor can hurt a lot.",
                                  ["Never run out of the top products: set a re-order alert.", "Promote the next 10-20 products with bundles or discounts to spread the risk.", "Review the long tail of slow products and stop buying the ones that do not sell."],
                                  [f"Top product: {g.index[0]} ({money(g.iloc[0])})", f"{len(g)} products with sales"], "Medium", int((rev > 0).sum()), "Products"))
            elif top10 < 0.25:
                cards.append(card("product_spread", "Your sales are well spread across products", f"The top 10 products make only {top10:.0%} of sales.", "A wide spread lowers the risk from any single product.",
                                  ["Keep watching which products start to grow so you can stock them early."], [f"{len(g)} products with sales"], "Good news", int((rev > 0).sum()), "Products"))
    # ---- seasonality
    if rev is not None and date is not None:
        mm = rev[rev > 0].groupby(date[rev > 0].dt.to_period("M")).sum()
        mm = mm[mm > 0]
        if len(mm) >= 6 and mm.min() > 0 and mm.max() / mm.min() >= 2 and _season_is_real(rev, date, mm.max() / mm.min()):
            hi, lo = mm.idxmax(), mm.idxmin()
            tail = mm.iloc[-1] < mm.iloc[:-1].median() * 0.6 if len(mm) > 3 else False
            cards.append(card("seasonality", f"Sales swing a lot during the year (best {hi}, weakest {lo})",
                              f"Your best month ({hi}: {money(mm.max())}) earned {mm.max() / mm.min():.1f} times your weakest ({lo}: {money(mm.min())}).",
                              "Big swings usually come from seasons, holidays or promotions. The last month may also be incomplete, so check before drawing conclusions." if tail else "Big swings usually come from seasons, holidays or promotions.",
                              ["Order stock and plan staff ahead of your best months.", "Run a promotion or bundle in your weakest months to even things out.", "Compare with last year to confirm the pattern repeats."],
                              [f"{len(mm)} months of sales analysed"], "Medium", len(mm) * 30, "Sales", "Check that the final month in your file is complete."))
        if len(mm) >= 6:
            rho, p = stats.spearmanr(np.arange(len(mm)), mm.values)
            if p < 0.05 and abs(rho) > 0.5:
                up = rho > 0
                cards.append(card("trend", "Sales are " + ("growing steadily" if up else "slowly falling"),
                                  f"Monthly sales show a clear {'upward' if up else 'downward'} trend over {len(mm)} months.",
                                  "The trend is statistically significant (not random noise)." + ("" if up else " Typical reasons: fewer new customers, fewer repeat purchases, or price pressure."),
                                  ["Keep doing what works and note what changed recently." if up else "Find out whether the drop comes from fewer customers or smaller baskets (see customer advice).", "Set a monthly target and review it every month."],
                                  [f"Trend strength {rho:.2f}, p={p:.3f}"], "Good news" if up else "High", len(mm) * 30, "Sales"))
    # ---- customers
    if cust is not None and rev is not None:
        cr = rev[rev > 0].groupby(cust[rev > 0]).sum().sort_values(ascending=False)
        if len(cr) >= 20 and cr.sum() > 0:
            top = cr.head(max(1, int(len(cr) * 0.1))).sum() / cr.sum()
            if top >= 0.5:
                cards.append(card("customer_concentration", "A small group of customers brings most of your income",
                                  f"The top 10% of customers ({max(1, int(len(cr) * 0.1)):,} people) generate {top:.0%} of revenue.",
                                  "Losing a few of these customers would hurt more than losing many small ones.",
                                  ["Create a simple loyalty or thank-you programme for the top customers.", "Contact each top customer personally once a quarter.", "Find customers who look like them and target them with ads or offers."],
                                  [f"{len(cr):,} customers with purchases"], "Medium", len(cr), "Customers"))
        if date is not None and len(cr) >= 20:
            d2 = pd.DataFrame({"c": cust, "d": date, "r": rev}).dropna()
            d2 = d2[d2.r > 0]
            last = d2.groupby("c").d.max(); freq = d2.groupby("c").d.nunique(); ref = d2.d.max()
            days = (ref - last).dt.days
            at_risk = ((freq >= 3) & (days > max(60, days.median() * 2))).sum()
            if at_risk >= 5 and _winback_is_real(d2, at_risk):
                cards.append(card("win_back", f"{int(at_risk)} good customers have gone quiet",
                                  f"{int(at_risk)} customers who bought 3 or more times have not bought for over {int(max(60, days.median() * 2))} days.",
                                  "Regular buyers who stop are often lost to a competitor or simply forgot about you.",
                                  ["Send them a friendly 'we miss you' message with a small offer.", "Ask a few of them why they stopped buying.", "Track how many come back, and repeat if it works (this is a test, the result is not guaranteed)."],
                                  [f"{len(freq):,} customers checked"], "Medium", len(freq), "Customers", "The size of the benefit is unknown until you try."))
    # ---- regions
    if reg is not None and rev is not None:
        rr = rev[rev > 0].groupby(reg[rev > 0]).sum().sort_values(ascending=False)
        if len(rr) >= 3 and rr.sum() > 0 and rr.iloc[0] / rr.sum() >= 0.7:
            cards.append(card("region_concentration", f"Almost all sales come from one region ({rr.index[0]})",
                              f"{rr.index[0]} makes {rr.iloc[0] / rr.sum():.0%} of sales.", "Depending on one market makes the business sensitive to local changes (rules, competition, shipping costs).",
                              ["Test a small marketing budget in your next two best regions.", "Check shipping cost and time to those regions before promoting there."],
                              [f"{len(rr)} regions with sales"], "Medium", int((rev > 0).sum()), "Regions"))
    # ---- ratings
    if rating is not None:
        r = _num(rating).dropna()
        r = r[(r >= 0) & (r <= r.max())]
        if len(r) >= 50:
            low = (r <= 2).mean(); hi_ = (r >= 4).mean()
            tx = [f"{len(r):,} ratings analysed", f"Low (1-2): {low:.0%}, high (4-5): {hi_:.0%}"]
            if low >= 0.15:
                why = "Low ratings are a signal; the star number alone does not say why."
                bits = []
                if prod is not None:
                    pr = pd.DataFrame({"p": prod, "r": _num(rating)}).dropna().groupby("p").r.agg(["mean", "size"])
                    w = pr[pr["size"] >= 30].sort_values("mean").head(3)
                    if len(w):
                        why = "The lowest-rated products with at least 30 ratings are: " + ", ".join(f"{i} ({v['mean']:.1f})" for i, v in w.iterrows()) + "."; tx.append("Worst products listed above")
                cards.append(card("low_ratings", "Too many customers are giving low ratings", f"{low:.0%} of ratings are 1-2 stars.", why,
                                  ["Read the most recent low-rated reviews and write down the top 3 repeated complaints.", "Reply to unhappy customers within 48 hours and offer a fix.", "Fix or remove the worst-rated products."],
                                  tx, "High" if low >= 0.25 else "Medium", len(r), "Reviews"))
            else:
                cards.append(card("ratings_good", "Customers are mostly happy", f"{hi_:.0%} of ratings are 4-5 stars and only {low:.0%} are 1-2 stars.", "Few bad ratings means the basics (product, delivery, service) work.",
                                  ["Ask happy customers for reviews to build trust.", "Keep tracking the share of 1-2 star ratings monthly."], tx, "Good news", len(r), "Reviews"))
            if seller is not None:
                sr = pd.DataFrame({"s": seller, "r": _num(rating)}).dropna().groupby("s").r.agg(["mean", "size"])
                w = sr[sr["size"] >= 30]
                if len(w) >= 5:
                    bad = w[w["mean"] < w["mean"].mean() - 1.0].sort_values("mean")
                    if len(bad):
                        cards.append(card("weak_sellers", f"{len(bad)} sellers are rated far below the rest", f"These sellers average more than one star below the average seller ({w['mean'].mean():.1f}).",
                                          "Their buyers consistently receive a worse experience (product, delivery or service).",
                                          ["Share their ratings with them and ask for an improvement plan.", "Pause promotion of the weakest sellers until they improve."],
                                          [f"{i}: {v['mean']:.1f} average, {int(v['size'])} ratings" for i, v in bad.head(3).iterrows()], "Medium", int(bad["size"].sum()), "Sellers"))
    # ---- delivery
    dd = col("delivery_days")
    if dd is None and date is not None and col("delivery_date") is not None:
        dd = (pd.to_datetime(col("delivery_date"), errors="coerce") - date).dt.days
    if dd is not None:
        x = _num(dd).dropna(); x = x[(x >= 0) & (x < 365)]
        if len(x) >= 50:
            slow = (x > x.median() * 2).mean()
            why = "Very slow deliveries are usually caused by one slow region, carrier or seller."
            ev = [f"{len(x):,} deliveries, median {x.median():.0f} days, 90th percentile {x.quantile(.9):.0f} days"]
            if reg is not None:
                t = pd.DataFrame({"g": reg.loc[x.index], "x": x}).groupby("g").x.agg(["median", "size"]); t = t[t["size"] >= 30].sort_values("median", ascending=False)
                if len(t):
                    why = f"The slowest region is {t.index[0]} (median {t.iloc[0]['median']:.0f} days)."; ev.append(f"Slowest region: {t.index[0]}")
            if x.quantile(.9) > x.median() * 2:
                cards.append(card("slow_delivery", "A group of orders arrives much slower than normal", f"The slowest 10% of orders take {x.quantile(.9):.0f} days or more, compared with a typical {x.median():.0f}.",
                                  why, ["Find the carriers or sellers involved in the slowest orders.", "Tell customers a longer delivery time for those routes, or change carrier."], ev, "Medium", len(x), "Delivery"))
    # ---- odd values
    if rev is not None:
        v = rev.dropna()
        if len(v) >= 100:
            q1, q3 = v.quantile([.25, .75]); out_n = int(((v > q3 + 3 * (q3 - q1)) | (v < q1 - 3 * (q3 - q1))).sum())
            if out_n >= 5:
                cards.append(card("odd_values", f"{out_n:,} lines have unusually large or small amounts", "A few lines are far outside the normal range.",
                                  "These can be real bulk orders, or data-entry mistakes such as an extra zero.", ["Open the flagged lines in the anomalies file of your report and check them one by one.", "Fix wrong entries before making decisions from the totals."],
                                  [f"{len(v):,} amounts checked"], "Low", len(v), "Data quality", "An unusual value is not necessarily an error."))
    # ---- data quality
    miss = df.isna().mean(); worst = miss[miss > 0.2].sort_values(ascending=False)
    if len(worst):
        cards.append(card("missing_data", "Some columns have a lot of empty cells", f"{len(worst)} column(s) are more than 20% empty, e.g. {', '.join(map(str, worst.index[:3]))}.",
                          "Empty cells limit what can be analysed and can make results less reliable.", ["Ask whoever created the file to fill these columns, or stop relying on them.", "Re-upload the improved file to refresh the advice."],
                          [f"{c}: {v:.0%} empty" for c, v in worst.head(3).items()], "Low", n, "Data quality"))
    if not cards:
        have = [r for r in ("date", "revenue", "unit_price", "quantity", "product_name", "product_id", "customer_id", "region", "rating") if r in mp]
        if n < 30:
            msg = f"Your file has only {n} rows, which is too few to give reliable advice. Upload at least a few hundred rows."
        elif not ({"revenue", "unit_price", "quantity", "rating"} & set(have)):
            msg = "None of your mapped columns holds an amount, price, quantity or rating, so there is nothing to measure. Map at least one of these (Dataset Center, step 3)."
        else:
            msg = "No clear pattern was found. The data may simply have no strong pattern (for example, evenly spread sales), which is a valid result. Map more columns (date, product, customer, rating) to unlock more advice."
        return {"ready": True, "message": msg, "cards": []}
    return {"ready": True, "message": None, "cards": rank(cards), "rows": n}
