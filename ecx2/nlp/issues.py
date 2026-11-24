"""Review Intelligence v2: wraps the v1 rule-based Portuguese analyser and adds main issue, evidence keywords,
root-cause HYPOTHESIS, language check and a heuristic confidence. Nothing in v1 is changed.

Confidence is a transparent heuristic (evidence strength + agreement between rule sentiment and the trained model),
NOT a calibrated probability.
"""
from ml.sentiment_analysis import review_nlp as v1

PRIORITY = ["damaged_product", "wrong_product", "missing_product", "late_delivery", "seller_issue", "product_quality",
            "shipping", "packaging", "payment", "other"]
ROOT_CAUSE = {
    "late_delivery": "Possible fulfilment or carrier delay (handling time, carrier transit or an unrealistic delivery estimate).",
    "damaged_product": "Possible packaging weakness or rough handling in transit.",
    "wrong_product": "Possible picking/listing error (wrong variant, colour or size shipped or advertised).",
    "missing_product": "Possible incomplete shipment (multi-item order split or items omitted at packing).",
    "product_quality": "Possible quality or description-mismatch problem with the product itself.",
    "seller_issue": "Possible seller communication/service problem (no response, refunds, unmet promises).",
    "shipping": "Possible tracking or delivery-handover problem (address, missed delivery, tracking not updated).",
    "packaging": "Possible inadequate or untidy packaging.",
    "payment": "Possible billing, refund or payment-processing problem.",
    "other": "No specific operational cause detected in the text.",
}
_PT = set("o a os as de do da dos das que não nao com para por um uma muito produto entrega chegou recebi comprei foi mas em no na meu minha".split())
_EN = set("the and is was not but this that with for very product delivery arrived received bought order it of to".split())


def detect_language(norm):
    w = norm.split()
    if len(w) < 2:
        return "unknown"
    pt, en = sum(x in _PT for x in w), sum(x in _EN for x in w)
    if pt == 0 and en == 0:
        return "unknown"
    return "pt" if pt >= en else "en"


def _boundary_matches(t, rx):
    """v1 patterns have no leading word boundary (e.g. 'ma qualidade' matches inside 'otima qualidade').
    v2 keeps only matches that start at a word start. v1 itself is left unchanged for reproducibility."""
    return [m.group(0).strip() for m in rx.finditer(t) if m.start() == 0 or t[m.start() - 1] == " "]


def v2_categories(norm):
    t = v1._PRAISE_PHRASES.sub(" ", norm)
    found = [c for c in v1.COMPLAINT_CATEGORIES[:-1] if _boundary_matches(t, v1._COMPILED[c])]
    return found or ["other"]


def matched_terms(norm, cats):
    t = v1._PRAISE_PHRASES.sub(" ", norm)
    return {c: sorted(set(m for m in _boundary_matches(t, v1._COMPILED[c]) if m))[:6] for c in cats if c in v1._COMPILED}


def analyse_v2(text, model=None):
    if not isinstance(text, str) or not text.strip():
        return {"ok": False, "error": "Empty review text."}
    r = v1.analyse(text)
    norm = r["norm_text"]
    cats = v2_categories(norm)
    r["severity"] = v1.severity(norm, cats)
    lang = detect_language(norm)
    terms = matched_terms(norm, cats)
    main = next((c for c in PRIORITY if c in cats), "other")
    # evidence strength for the main issue = number of distinct trigger terms
    ev = len(terms.get(main, []))
    conf = 0.35 + 0.15 * min(ev, 3)
    notes = []
    if main == "other":
        conf = 0.3
    if len(norm.split()) < 4:
        conf -= 0.1; notes.append("very short text")
    if lang == "en":
        conf -= 0.2; notes.append("English text: taxonomy is Portuguese-only, results unreliable")
    elif lang == "unknown":
        conf -= 0.1; notes.append("language could not be determined")
    ms = None
    if model is not None:
        probs = model.predict_proba([norm])[0]
        ms = dict(zip(model.classes_, [round(float(p), 3) for p in probs]))
        lab = max(ms, key=ms.get)
        agree = (lab == "negative") == (r["lexicon_sentiment"] == "negative")
        conf += 0.1 if agree else -0.1
        if not agree:
            notes.append("rule sentiment and trained model disagree")
    conf = round(max(0.05, min(0.95, conf)), 2)
    return {"ok": True, "language": lang, "sentiment_rule": r["lexicon_sentiment"], "sentiment_model": ms,
            "complaint_categories": cats, "main_issue": main, "keywords": terms.get(main, []), "all_keywords": terms,
            "aspects": {k: v for k, v in r["aspects"].items() if v}, "severity": r["severity"],
            "possible_root_cause": ROOT_CAUSE[main], "confidence": conf,
            "confidence_note": "heuristic score from evidence strength and model agreement; not a calibrated probability",
            "caveats": notes + ["Root cause is a hypothesis from wording only."]}
