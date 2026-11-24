"""Advice for the Olist marketplace data (computed from the real processed files, cached)."""
from functools import lru_cache
from pathlib import Path
import pandas as pd
from app.insights.core import card, rank, pct

ROOT = Path(__file__).resolve().parents[2]
CAT_FIX = {
    "late_delivery": ("late or missing delivery", "Show honest delivery dates at checkout and send a proactive message when an order is running late."),
    "damaged_product": ("damaged products", "Improve packaging for fragile items and ask sellers to photograph items before dispatch."),
    "wrong_product": ("wrong product sent", "Add a pick-and-check step at the seller and show clear product photos and variants."),
    "missing_product": ("missing items in the parcel", "Ask sellers to confirm item counts at packing time; split multi-item orders clearly."),
    "product_quality": ("poor product quality", "Review the lowest-rated products and remove or fix listings that keep disappointing buyers."),
    "seller_issue": ("seller service problems", "Set response-time targets for sellers and warn sellers whose complaints keep rising."),
    "packaging": ("packaging problems", "Share packaging rules with sellers and test sturdier boxes on the worst categories."),
    "shipping": ("shipping problems", "Review the carriers used on the slowest routes and negotiate service levels."),
    "payment": ("payment problems", "Check failed or duplicate payments and make refund status visible to the buyer."),
}


@lru_cache(maxsize=1)
def _orders():
    cols = ["order_id", "review_score", "is_late", "dissatisfied", "delay_days", "seller_handling_days", "carrier_transit_days", "freight_ratio",
            "customer_state", "primary_category", "primary_seller_id"]
    p = ROOT / "data" / "processed" / "orders_with_history.parquet"
    d = pd.read_parquet(p, columns=[c for c in cols if c])
    return d


def _bi(name):
    return pd.read_csv(ROOT / "data" / "processed" / "bi" / f"{name}.csv")


@lru_cache(maxsize=1)
def build():
    d = _orders()
    out = []
    n = len(d)
    late, ontime = d[d.is_late == 1], d[d.is_late == 0]
    # 1. Late delivery -> bad reviews
    lr = d.is_late.mean()
    out.append(card(
        "late_vs_reviews", "Late deliveries are the biggest reason customers leave bad reviews",
        f"{pct(lr, 1)} of orders arrive after the promised date, and those customers are far unhappier: {late.dissatisfied.mean():.0%} give 1-2 stars, compared with {ontime.dissatisfied.mean():.0%} of customers whose order arrived on time.",
        f"Average rating is {late.review_score.mean():.1f} for late orders versus {ontime.review_score.mean():.1f} for on-time ones, and the gap grows the later the parcel is. Customers judge the delivery promise, not just the product.",
        ["Show a realistic delivery date at checkout (add a small safety buffer on slow routes).", "Message customers early when an order is going to be late and offer a small apology (voucher or free shipping next time).",
         "Track late orders weekly and treat 'late rate' as a headline number for the operations team."],
        [f"{n:,} delivered, reviewed orders analysed", f"Late orders: {len(late):,} ({pct(lr, 1)}); average rating {late.review_score.mean():.2f}", f"On-time orders: average rating {ontime.review_score.mean():.2f}"],
        "High", n, "Delivery", "This is an association found in the data. Late orders can also differ in other ways, so expect a large - but not exactly this - improvement if delays are cut."))
    # 2. Where does the delay come from
    mh, mc = late.seller_handling_days.mean(), late.carrier_transit_days.mean()
    mh0, mc0 = ontime.seller_handling_days.mean(), ontime.carrier_transit_days.mean()
    extra_h, extra_c = mh - mh0, mc - mc0
    if pd.notna(extra_h) and pd.notna(extra_c):
        main = "the seller (time to hand the parcel to the carrier)" if extra_h >= extra_c else "the carrier (time on the road)"
        out.append(card(
            "delay_stage", "Delays start at " + ("the seller's warehouse" if extra_h >= extra_c else "the carrier"),
            f"Late orders take longer at both steps, but the bigger jump is at {main}.",
            f"Sellers hold late orders for {mh:.1f} days before dispatch versus {mh0:.1f} days for on-time orders (+{extra_h:.1f}); carriers take {mc:.1f} days versus {mc0:.1f} (+{extra_c:.1f}).",
            ["Ask sellers to dispatch within 2 days and show seller dispatch time on their profile." if extra_h >= extra_c else "Review the carriers on the slowest routes and agree delivery-time targets with them.",
             "Send sellers an automatic reminder when an order has not been dispatched after 2 days.", "Reward fast-dispatch sellers with better search ranking."],
            [f"Seller handling: {mh:.1f} days (late) vs {mh0:.1f} (on time)", f"Carrier transit: {mc:.1f} days (late) vs {mc0:.1f} (on time)"],
            "High" if extra_h >= extra_c else "Medium", len(late), "Delivery"))
    # 3. Worst regions
    st = _bi("states")
    hot = st[st.hotspot == True].sort_values("dissatisfied_rate", ascending=False)  # noqa: E712
    if len(hot):
        names = ", ".join(hot.customer_state.head(5))
        top = hot.iloc[0]
        out.append(card(
            "hot_states", f"Customers in {names} are unhappier than elsewhere",
            f"{len(hot)} states have clearly more bad reviews than the national average. The worst is {top.customer_state}, where {top.dissatisfied_rate:.0%} of orders get 1-2 stars and {top.late_rate:.0%} arrive late.",
            f"These states also have longer deliveries (about {hot.avg_delivery_days.mean():.0f} days on average) and a higher late rate ({hot.late_rate.mean():.0%} on average), so distance and carrier coverage are the likely reasons.",
            [f"Add extra delivery days to the promise for {names}.", "Work with a regional carrier or a local warehouse for these states.", "Review seller-to-customer routes that cross the country and encourage sellers closer to these customers."],
            [f"{len(hot)} flagged states (statistically significant gap)", *(f"{r.customer_state}: {r.dissatisfied_rate:.0%} unhappy, {r.late_rate:.0%} late, {int(r.orders):,} orders" for r in hot.head(4).itertuples())],
            "High", int(hot.orders.sum()), "Regions"))
    # 4. Category problems
    cat = _bi("categories")
    pr = cat[cat.segment == "PRIORITY PROBLEM"].sort_values("orders", ascending=False)
    if len(pr):
        t = pr.head(4)
        out.append(card(
            "priority_categories", "Some product categories need attention first",
            f"{len(pr)} categories sell well but disappoint customers more than average. Biggest: " + ", ".join(t.primary_category.str.replace("_", " ")) + ".",
            "These categories combine a high bad-review rate with a meaningful number of orders, so fixing them helps the most customers. Common reasons in the review text are late delivery, product quality and wrong or damaged items.",
            ["Read the recent 1-2 star reviews for each of these categories and list the top 3 repeated complaints.", "Remove or fix the lowest-rated listings and sellers inside these categories.", "Check packaging and delivery time for these categories separately."],
            [f"{r.primary_category.replace('_', ' ')}: {r.dissatisfied_rate:.0%} unhappy, {int(r.orders):,} orders, average rating {r.avg_score:.2f}" for r in t.itertuples()],
            "Medium", int(t.orders.sum()), "Products"))
    # 5. Risky sellers
    se = _bi("sellers")
    rs = se[(se.segment == "PRIORITY PROBLEM") & (se.orders >= 100)].sort_values("orders", ascending=False)
    if len(rs):
        out.append(card(
            "risky_sellers", f"{len(rs)} high-volume sellers create many unhappy customers",
            f"These sellers each have at least 100 orders and a clearly worse record than their peers; together they handled {int(rs.orders.sum()):,} orders.",
            f"On average they deliver late {rs.late_rate.mean():.0%} of the time and {rs.dissatisfied_rate.mean():.0%} of their buyers leave 1-2 stars.",
            ["Contact these sellers with their own numbers and a clear improvement target.", "Give them 60-90 days to improve before limiting how visible their products are.", "Share what the best sellers do (fast dispatch, good packaging)."],
            [f"{len(rs)} sellers with 100+ orders flagged", f"Average late rate {rs.late_rate.mean():.0%}, average unhappy rate {rs.dissatisfied_rate.mean():.0%}"],
            "Medium", int(rs.orders.sum()), "Sellers"))
    # 6. What customers complain about
    nlp_p = ROOT / "data" / "processed" / "review_nlp.parquet"
    if nlp_p.exists():
        nl = pd.read_parquet(nlp_p)
        cats = [c for c in nl.columns if c.startswith("cat_") and c[4:] in CAT_FIX]
        neg = nl[nl.sent_true.eq("negative")] if "sent_true" in nl and nl.sent_true.notna().any() else nl[nl[cats].sum(axis=1) > 0]
        if len(neg):
            share = (neg[cats].mean()).sort_values(ascending=False)
            t = [(c[4:], v) for c, v in share.items() if v > 0][:3]
            if t:
                top = CAT_FIX[t[0][0]]
                out.append(card(
                    "complaints", f"Most complaints are about {top[0]}",
                    "When customers write a negative review, these are the topics they mention most: " + "; ".join(f"{CAT_FIX[k][0]} ({v:.0%})" for k, v in t) + ".",
                    "Review text is grouped by keyword rules (Portuguese). A review can mention more than one topic, so shares add up to more than 100%.",
                    [CAT_FIX[k][1] for k, _ in t],
                    [f"{len(neg):,} negative reviews with text analysed"], "Medium", len(neg), "Customer voice",
                    "Keyword matching is simple and can miss unusual wording."))
    # 7. Peak months
    m = _bi("monthly")
    if "late_pct" in m:
        w = m.sort_values("late_pct", ascending=False).iloc[0]
        if w.late_pct >= 2 * max(m.late_pct.median(), 0.5):
            out.append(card(
                "peak_month", f"Deliveries went badly in {w.purchase_month}",
                f"In {w.purchase_month} {w.late_pct:.0f}% of orders were late, against a typical month of {m.late_pct.median():.1f}%.",
                (f"Order volume that month ({int(w.orders):,}) was well above a typical month ({int(m.orders.median()):,}), so carriers and sellers were probably stretched." if w.orders >= 1.3 * m.orders.median()
                 else f"Order volume that month ({int(w.orders):,}) was not unusual (typical month {int(m.orders.median()):,}), so volume alone does not explain it - something changed with carriers or sellers."),
                ["Ask carriers and top sellers what went wrong in that month and write down the answer.", "Set an alert so that a late rate above 10% in any week is investigated immediately.", "Add a temporary delivery buffer when the late rate starts rising."],
                [f"{w.purchase_month}: {int(w.orders):,} orders, {w.late_pct:.1f}% late", f"Typical month: {m.late_pct.median():.1f}% late"], "Medium", int(w.orders), "Delivery",
                "This points to timing, not a proven single cause."))
    # 8. Good news: freight is not the problem
    fr = d[["freight_ratio", "dissatisfied"]].dropna()
    if len(fr) > 1000:
        hi = fr[fr.freight_ratio >= fr.freight_ratio.quantile(.8)].dissatisfied.mean(); lo = fr[fr.freight_ratio <= fr.freight_ratio.quantile(.2)].dissatisfied.mean()
        out.append(card(
            "freight_ok", "Shipping cost is not what drives bad reviews",
            "It is tempting to blame expensive shipping, but the gap is small: customers with the highest shipping cost (relative to price) are only slightly less satisfied than those with the lowest.",
            f"Unhappy rate is {hi:.0%} for the highest-freight fifth of orders versus {lo:.0%} for the lowest fifth. Delivery lateness explains far more.",
            ["Do not spend effort cutting shipping prices just to improve reviews.", "Put the effort into delivery speed and honest delivery dates instead."],
            [f"{len(fr):,} orders compared"], "Good news", len(fr), "Delivery") if abs(hi - lo) < 0.05 else None)
    return rank([c for c in out if c])


def summary():
    d = _orders()
    return {"orders": int(len(d)), "late_rate": float(d.is_late.mean()), "mean_review": float(d.review_score.mean()), "dissatisfied_rate": float(d.dissatisfied.mean())}
