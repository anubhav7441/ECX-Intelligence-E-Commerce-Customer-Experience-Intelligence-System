"""Small, transparent rule-based review NLP for English (new) and Portuguese (reuses v1 rules). Lexicon sentiment is an
approximation; the analysis reports how well it agrees with the ratings instead of claiming accuracy."""
import re
import numpy as np

POS = set("good great excellent love loved perfect amazing awesome works well nice happy best recommend recommended easy fast quiet sturdy solid reliable worth satisfied fantastic wonderful perfectly durable comfortable".split())
NEG = set("bad poor terrible awful worst broken broke defective useless waste disappointed disappointing horrible cheap flimsy leaks leaked stopped failed fail failure noisy loud cracked damaged wrong missing returned refund return refunded junk garbage unhappy never problem problems issue issues difficult hard".split())
NEGATORS = {"not", "no", "never", "dont", "don't", "doesnt", "doesn't", "didnt", "didn't", "wont", "won't", "cant", "can't", "isnt", "isn't", "wasnt", "wasn't", "hardly"}
EN_TAX = {
    "delivery_shipping": r"\b(late|arrived late|delay\w*|shipping|shipped|never arrived|didn'?t arrive|took (forever|weeks)|slow delivery|lost in)\b",
    "damaged_defective": r"\b(damaged|broken|broke|cracked|defect\w*|dented|arrived (broken|damaged))\b",
    "product_quality": r"\b(poor quality|low quality|cheap(ly)? made|flimsy|fell apart|stopped working|doesn'?t work|didn'?t work|not working|does not work|junk|garbage|malfunction\w*|leak\w*)\b",
    "wrong_or_missing_item": r"\b(wrong (item|size|part|model|color)|missing (part|piece|item|screws?)|not as described|different from|incompatible|doesn'?t fit|does not fit|wrong one)\b",
    "packaging": r"\b(packag\w*|box was|arrived open|poorly packed)\b",
    "price_value": r"\b(overpriced|too expensive|not worth|waste of money|rip ?off|pricey)\b",
    "customer_service": r"\b(customer service|no response|never responded|rude|support|seller (was|is) (rude|unresponsive))\b",
    "refund_return": r"\b(refund\w*|return(ed|ing)?|money back|exchange|replacement)\b",
}
EN_ASPECTS = {
    "quality": r"\b(quality|build|material|made|sturdy|flimsy|durable|durability)\b",
    "price_value": r"\b(price|value|worth|cheap|expensive|cost|money)\b",
    "delivery": r"\b(delivery|shipping|shipped|arrived|package|packaging|box)\b",
    "performance": r"\b(works?|working|performance|power|noise|noisy|quiet|fast|slow|efficien\w*)\b",
    "installation_ease": r"\b(install\w*|setup|set up|easy to|instructions|assembl\w*|fit|fits)\b",
    "service": r"\b(customer service|support|seller|refund|warranty|response)\b",
}
_TOK = re.compile(r"[a-z']+")
_C = {k: re.compile(v) for k, v in EN_TAX.items()}
_A = {k: re.compile(v) for k, v in EN_ASPECTS.items()}


def lex_score(text):
    """Returns a score in [-1, 1]: (pos - neg)/(pos + neg) with simple negation (flip within 3 tokens)."""
    toks = _TOK.findall(text.lower())
    p = n = 0
    for i, t in enumerate(toks):
        neg = any(x in NEGATORS for x in toks[max(0, i - 3):i])
        if t in POS:
            p, n = (p, n + 1) if neg else (p + 1, n)
        elif t in NEG:
            p, n = (p + 1, n) if neg else (p, n + 1)
    return 0.0 if p + n == 0 else (p - n) / (p + n)


def label(score):
    return "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"


def en_categories(text):
    t = text.lower()
    return [k for k, rx in _C.items() if rx.search(t)]


def en_aspects(text):
    """aspect -> sentiment of the sentences that mention it"""
    out = {}
    for sent in re.split(r"[.!?;\n]+", text.lower()):
        if not sent.strip():
            continue
        for k, rx in _A.items():
            if rx.search(sent):
                s = lex_score(sent)
                if s != 0:
                    out.setdefault(k, []).append(s)
    return {k: float(np.mean(v)) for k, v in out.items()}


def detect_lang(texts):
    from ecx2.nlp.issues import detect_language, v1
    c = {"pt": 0, "en": 0, "unknown": 0}
    for t in texts[:300]:
        c[detect_language(v1.normalize(t))] += 1
    return max(c, key=c.get), c
