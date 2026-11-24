"""Capability engine: an analysis runs ONLY when its required column roles are mapped. Otherwise a clear reason is returned.
Requirement syntax: 'a|b+c' means  a  OR  (b AND c).  Every requirement in the list must be satisfied."""
from multids.mapping import ROLES

ANALYSES = {
    "sales_overview": {"title": "Sales & revenue overview", "group": "Transactions", "requires": ["date", "revenue|quantity+unit_price"],
                       "needs_text": "a date column and either a revenue column or quantity + unit price"},
    "product_analysis": {"title": "Product analysis", "group": "Transactions", "requires": ["product_id|product_name", "revenue|quantity+unit_price|quantity"],
                         "needs_text": "a product column and revenue (or quantity)"},
    "customer_segmentation": {"title": "Customer segmentation (RFM)", "group": "Customers", "requires": ["customer_id", "date", "revenue|quantity+unit_price"],
                              "needs_text": "customer ID, date and revenue"},
    "geographic_analysis": {"title": "Geographic comparison", "group": "Geography", "requires": ["region", "revenue|quantity+unit_price|order_id|rating"],
                            "needs_text": "a country/state/region column and a measure (revenue, orders or rating)"},
    "review_analysis": {"title": "Review & sentiment analysis", "group": "Reviews", "requires": ["rating|review_text"],
                        "needs_text": "a rating column and/or a review text column"},
    "delivery_analysis": {"title": "Delivery performance", "group": "Delivery", "requires": ["delivery_days|date+delivery_date"],
                          "needs_text": "delivery days, or order date + delivered date"},
    "seller_analysis": {"title": "Seller comparison", "group": "Sellers", "requires": ["seller_id", "rating|revenue|quantity+unit_price|delivery_days|date+delivery_date"],
                        "needs_text": "a seller column and a performance measure (rating, revenue or delivery time)"},
    "numeric_statistics": {"title": "Statistics & anomaly detection", "group": "Numeric", "requires": [],
                           "needs_text": "at least one numeric column (no mapping needed)", "needs_numeric": True},
}


def _roles_ok(spec, roles):
    return any(all(p in roles for p in alt.split("+")) for alt in spec.split("|"))


def _nice(spec):
    alts = []
    for alt in spec.split("|"):
        alts.append(" + ".join(ROLES[p][0] for p in alt.split("+")))
    return " or ".join(alts)


def evaluate(mapping, n_numeric_cols=0):
    roles = {r for r, c in (mapping or {}).items() if c}
    out = []
    for aid, a in ANALYSES.items():
        missing = [s for s in a["requires"] if not _roles_ok(s, roles)]
        if a.get("needs_numeric") and n_numeric_cols < 1:
            missing.append("a numeric column")
        ok = not missing
        reason = None if ok else "Cannot run: this analysis needs " + a["needs_text"] + ". Not mapped: " + "; ".join(_nice(m) if "|" in m or m in ROLES or "+" in m and all(p in ROLES for p in m.split("+")) else m for m in missing) + "."
        out.append({"id": aid, "title": a["title"], "group": a["group"], "available": ok, "reason": reason})
    return out


def can_run(aid, mapping, n_numeric_cols=0):
    for c in evaluate(mapping, n_numeric_cols):
        if c["id"] == aid:
            return c
    return {"id": aid, "available": False, "reason": f"Unknown analysis '{aid}'."}


def model_compatibility(mapping, kinds=None):
    """Never auto-apply the Olist-trained model. Explain exactly why it is not applicable."""
    need = ["Olist-specific delivery features (delivery_days, delay_days, estimated_days)", "freight value and freight ratio", "seller and customer state", "Olist product category names (Portuguese)", "historical seller/category review statistics built from Olist reviews"]
    roles = {r for r, c in (mapping or {}).items() if c}
    problems = []
    for r, lbl in [("delivery_days", "delivery days"), ("delay_days", "delay vs estimate"), ("rating", "review score target")]:
        if r not in roles:
            problems.append(f"No mapped column for {lbl}.")
    problems.append("The Olist model was trained on Olist's own feature definitions and 2016-2018 Brazilian marketplace data; other datasets use different units, categories and target definitions, so predictions would not be valid.")
    return {"compatible": False, "olist_model_applied": False, "reasons": problems, "required_by_olist_model": need,
            "next_step": "A new model could be trained on THIS dataset only (separate model file, Olist model untouched) after you confirm a target definition. Retraining is not run automatically."}
