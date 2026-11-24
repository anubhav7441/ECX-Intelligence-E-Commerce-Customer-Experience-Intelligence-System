"""Grounded business research assistant.

Pipeline: question -> intent router (rules) -> tools (pandas/scipy) -> deterministic answer with citations.
  * Works with NO LLM. If an optional local LLM is configured (ECX_LLM_URL, e.g. Ollama), it may only REPHRASE the already
    computed answer; the rephrasing is rejected if it introduces numbers that are not in the tool facts.
  * Unsupported/unknown questions are refused with a list of what can be answered. Statistics are never produced by an LLM.
"""
import json
import os
import re
from ecx2 import store, tools
from ecx2.agents.system import Supervisor

SUPPORTED = ["platform KPIs ('what is the late rate?')", "worst/best sellers ('which sellers have the highest late rate?')",
             "a seller profile ('tell me about seller 1ca7077d')", "a seller investigation ('investigate seller 1ca7077d')",
             "association tests ('does late delivery affect dissatisfaction?', 'is distance related to late delivery?')",
             "state / category comparison ('which states have the most dissatisfied customers?')",
             "monthly events ('what happened in 2017-11?')", "anomalies ('show anomalies')"]
_SELLER = re.compile(r"\b([0-9a-f]{6,32})\b")
_MONTH = re.compile(r"\b(20\d{2}-\d{2})\b")
SYN_X = {"late delivery": ["late", "delay", "delayed", "on time", "on-time", "lateness"], "multi-item order": ["multi-item", "multiple items", "more than one item", "multi item"],
         "same-state delivery": ["same state", "same-state"], "high freight ratio": ["freight ratio", "shipping cost ratio"],
         "long distance": ["long distance", "far away"], "delivery days": ["delivery time", "delivery days", "transit time", "days to deliver"],
         "estimated days": ["estimated days", "delivery estimate", "promised"], "distance": ["distance"], "price": ["price", "expensive"],
         "seller handling days": ["handling", "seller dispatch", "dispatch time"], "freight ratio": ["freight"]}
SYN_Y = {"dissatisfaction": ["dissatisf", "unhappy", "negative review", "bad review", "1-2 star", "poor review"], "late delivery": ["late delivery", "late deliveries", "delayed"], "review score": ["review score", "rating", "stars", "score"]}


def _has(q, words):
    return any(w in q for w in words)


def route(q):
    ql = q.lower()
    if _has(ql, ["investigate", "why is seller", "why might seller", "root cause"]) and (_SELLER.search(ql) or _MONTH.search(ql)):
        return "investigate"
    if _MONTH.search(ql) and _has(ql, ["what happened", "month", "in 20", "surge", "spike"]):
        return "month"
    if _has(ql, ["anomal", "unusual", "flagged", "suspicious", "outlier"]):
        return "anomalies"
    m = _SELLER.search(ql)
    if m and "seller" in ql:
        return "seller"
    if _has(ql, ["which seller", "worst seller", "best seller", "top seller", "sellers with", "sellers have"]):
        return "top_sellers"
    if _has(ql, ["which state", "which categor", "states have", "categories have", "by state", "by category", "worst state", "worst categor"]):
        return "groups"
    if _has(ql, ["affect", "relat", "associat", "correlat", "impact", "influence", "cause", "drive", "lead to", "effect of", "does ", "do "]) and \
            _pick(ql, SYN_X) and _pick(ql, SYN_Y):
        return "association"
    if _has(ql, ["late rate", "kpi", "average review", "how many orders", "overall", "platform", "dissatisfied rate", "mean review", "on average"]):
        return "kpi"
    return None


def _pick(ql, syn, exclude=None):
    best = None
    for k, ws in syn.items():
        if k == exclude:
            continue
        for w in ws + [k]:
            i = ql.find(w)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, k)
    return best[1] if best else None


def _p(v):
    return "p<1e-300" if v < 1e-300 else f"p={v:.3g}"


def _fmt(x, p=1):
    return f"{x * 100:.{p}f}%"


def answer(q, db=None, llm=None):
    q = (q or "").strip()
    if not q:
        return {"ok": False, "error": "Please type a question."}
    if len(q) > 500:
        return {"ok": False, "error": "Question is longer than 500 characters."}
    r = route(q)
    ql = q.lower()
    cites, calls, text, refused, extra = [], [], "", False, {}
    try:
        if r == "kpi":
            k = tools.platform_summary(); calls.append("platform_summary")
            text = (f"Across {k['n_orders']:,} delivered, reviewed orders ({k['period'][0]} to {k['period'][1]}): late rate {_fmt(k['late_rate'])}, dissatisfied (1-2 stars) {_fmt(k['dissatisfied_rate'])}, "
                    f"mean review score {k['mean_review_score']:.2f}, mean delivery {k['mean_delivery_days']:.1f} days vs {k['mean_estimated_days']:.1f} estimated.")
            cites.append({"source": k["source"], "ref": "platform_summary"})
        elif r == "top_sellers":
            metric = "dissatisfied_rate" if _has(ql, ["dissatisf", "unhappy", "review", "negative"]) else "late_rate"
            if _has(ql, ["delivery days", "slow"]):
                metric = "mean_delivery_days"
            best = _has(ql, ["best", "lowest", "fewest"])
            t = tools.top_sellers(metric, 5, worst=not best); calls.append("top_sellers")
            if not t["ok"]:
                return _refuse(q, t["error"], db)
            text = f"{'Best' if best else 'Worst'} 5 sellers by {metric.replace('_', ' ')} (min {t['min_orders']} orders): " + "; ".join(
                f"{x['seller_id'][:8]} {_fmt(x[metric]) if 'rate' in metric else f'{x[metric]:.1f}'} (n={int(x['n'])})" for x in t["rows"]) + ". Ranking is descriptive; check sample size and confidence intervals before drawing conclusions."
            cites.append({"source": t["source"], "ref": "top_sellers"})
        elif r == "seller":
            sid = _SELLER.search(ql).group(1); p = tools.seller_profile(sid); calls.append("seller_profile")
            if not p["ok"]:
                return _refuse(q, p["error"], db)
            text = (f"Seller {p['seller_id'][:8]}: {p['n_orders']} orders, late {_fmt(p['late_rate'])} (95% CI {_fmt(p['late_ci95'][0])}-{_fmt(p['late_ci95'][1])}) vs {_fmt(p['platform']['late_rate'])} platform; "
                    f"dissatisfied {_fmt(p['dissatisfied_rate'])} vs {_fmt(p['platform']['dissatisfied_rate'])}; mean review {p['mean_review_score']:.2f}." + (" Small sample (<30 orders): interpret with caution." if p["small_sample"] else ""))
            cites.append({"source": p["source"], "ref": f"seller {p['seller_id'][:8]}"})
        elif r == "investigate":
            tgt = ("seller", _SELLER.search(ql).group(1)) if "seller" in ql or not _MONTH.search(ql) else ("platform_month", _MONTH.search(ql).group(1))
            out = Supervisor(db).investigate(*tgt); calls.append("Supervisor.investigate")
            if not out["ok"]:
                return _refuse(q, out["error"], db)
            rep = out["report"]; extra["investigation"] = rep.model_dump()
            text = rep.summary + " Evidence: " + ", ".join(sorted({f.evidence_id for a in rep.reports for f in a.findings if f.evidence_id})[:6]) + "."
            cites += [{"source": "Evidence Ledger", "ref": f"run {rep.run_id}"}]
        elif r == "month":
            m = tools.month_profile(_MONTH.search(ql).group(1)); calls.append("month_profile")
            if not m["ok"]:
                return _refuse(q, m["error"], db)
            text = (f"{m['month']}: late rate {_fmt(m['late_rate'])} vs {_fmt(m['late_rate_other_months'])} in other months (p={m['late_p_value']:.3g}); dissatisfied {_fmt(m['dissatisfied_rate'])} vs {_fmt(m['dissatisfied_rate_other_months'])}. "
                    "The dataset has no information on why (events, strikes, stock), so causes are hypotheses only.")
            cites.append({"source": m["source"], "ref": f"month {m['month']}"})
        elif r == "anomalies":
            a = tools.anomalies(None, None, 5); calls.append("anomalies")
            if not a["ok"]:
                return _refuse(q, a["error"], db)
            text = f"{a['n']} anomaly signals are recorded. Top 5: " + " | ".join(f"{x['anomaly_id']} {x['entity'][:8]} {x['detector']} ({x['period']}): {x['why_flagged'][:110]}" for x in a["rows"]) + " Anomalies are investigation signals, not proof of fault."
            cites.append({"source": a["source"], "ref": "anomalies.csv"})
        elif r == "groups":
            by = "primary_category" if "categor" in ql else "customer_state"
            g = tools.group_late_dissat(by); calls.append("group_late_dissat")
            text = f"Highest dissatisfaction by {'category' if by == 'primary_category' else 'state'} (n>=100): " + "; ".join(f"{x[by]} {_fmt(x['dissatisfied_rate'])} (late {_fmt(x['late_rate'])}, n={int(x['n'])})" for x in g["rows"][:5]) + "."
            cites.append({"source": g["source"], "ref": "group_late_dissat"})
        elif r == "association":
            y = _pick(ql, {k: v for k, v in SYN_Y.items() if k != "late delivery"}) or "late delivery"
            x = _pick(ql, {k: v for k, v in SYN_X.items() if k != y})
            t = tools.test_association(x, y); calls.append("test_association")
            if not t["ok"]:
                return _refuse(q, t["error"], db)
            body = (f"{_fmt(t['rate_exposed'])} vs {_fmt(t['rate_unexposed'])} (odds ratio {t['odds_ratio']:.2f}, 95% CI {t['or_ci95'][0]:.2f}-{t['or_ci95'][1]:.2f})" if "odds_ratio" in t else
                    f"means {t['mean_exposed']:.2f} vs {t['mean_unexposed']:.2f} (Cliff's delta {t['effect_size']:.2f})" if "mean_exposed" in t else f"Spearman rho {t['spearman_rho']:.3f}")
            adj = t.get("adjusted", {})
            text = (f"Hypothesis: {x} is associated with {y}. n={t['n']:,}. {body}; {_p(t['p_value'])}; effect size {t['effect_size']:.2f}. Verdict: {t['verdict']}."
                    + (f" After adjusting for {', '.join(adj['controls'])}: odds ratio {adj['odds_ratio']:.2f} (95% CI {adj['ci95'][0]:.2f}-{adj['ci95'][1]:.2f})." if "odds_ratio" in adj else "")
                    + " This is an association in observational data, not proof of causation.")
            cites.append({"source": t["source"], "ref": f"test_association({x}, {y})"}); extra["test"] = t
        else:
            return _refuse(q, "I can only answer questions that my project data and statistical tools can support. I will not guess.", db, route_name="unsupported")
    except Exception as ex:   # noqa
        return _refuse(q, f"A tool failed ({ex.__class__.__name__}); no answer was produced rather than risking an unsupported one.", db, route_name=r or "error")
    final = text
    used_llm = False
    if llm is None:
        llm = _llm_from_env()
    if llm is not None:
        try:
            rew = llm(f"Rewrite the answer below in 2-3 plain sentences. Use ONLY the facts and numbers given; do not add any number.\n\nQUESTION: {q}\nANSWER FACTS: {text}")
            if rew and _numbers_subset(rew, text):
                final, used_llm = rew.strip(), True
        except Exception:   # noqa  - LLM unavailable: fall back silently to the deterministic answer
            pass
    store.log_assistant(q, r, True, False, final, db=db)
    return {"ok": True, "route": r, "answer": final, "deterministic_answer": text, "used_llm": used_llm, "citations": cites,
            "tool_calls": calls, "grounded": True, "refused": False, **extra}


def _numbers_subset(new, base):
    nums = lambda s: set(re.findall(r"\d+(?:\.\d+)?", s))
    return nums(new) <= nums(base)


def _refuse(q, reason, db, route_name="refused"):
    msg = reason + " Supported question types: " + "; ".join(SUPPORTED) + "."
    store.log_assistant(q, route_name, False, True, msg, db=db)
    return {"ok": True, "route": route_name, "answer": msg, "citations": [], "tool_calls": [], "grounded": False, "refused": True, "used_llm": False}


def _llm_from_env():
    url = os.environ.get("ECX_LLM_URL")      # e.g. http://localhost:11434  (Ollama). UNTESTED against a real server in the build environment.
    if not url or not url.lower().startswith(("http://", "https://")):   # only web addresses; never file: or other schemes
        return None
    model = os.environ.get("ECX_LLM_MODEL", "llama3.2")

    def call(prompt):
        import urllib.request
        req = urllib.request.Request(url.rstrip("/") + "/api/generate", data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:  # nosec B310
            return json.loads(r.read())["response"]
    return call
