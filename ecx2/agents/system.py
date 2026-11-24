"""Multi-agent investigation system: a deterministic Supervisor delegates to five specialised agents.

Design choice (documented): a small custom tool-calling architecture instead of LangGraph/CrewAI - fewer dependencies, fully
testable offline, every step logged. Agents never compute numbers themselves; they call tools in ecx2.tools (pandas/scipy).
An LLM is NOT required anywhere in this pipeline.
"""
import time
import uuid
from ecx2 import store, tools
from ecx2.agents.schemas import AgentReport, Finding, InvestigationReport, ProposedAction, ToolCall

LIMITS = ["Observational historical data (Olist 2016-2018); associations are not causes.",
          "Only delivered, reviewed orders are analysed; anomalies/hypotheses are investigation signals, not proof of fault.",
          "Root causes are hypotheses; the system does not execute any real-world action."]


def conf(n, p):
    if p is None:
        return "Low"
    return "High" if n >= 100 and p < 1e-3 else "Medium" if n >= 30 and p < 0.05 else "Low"


class Base:
    name = "Agent"

    def __init__(self, db=None, run_id=None):
        self.db, self.run_id = db, run_id
        self.rep = None

    def call(self, fn, *a, **k):
        t = time.time()
        tc = ToolCall(name=fn.__name__, args={"args": [str(x)[:40] for x in a], **{kk: str(v)[:40] for kk, v in k.items()}})
        try:
            r = fn(*a, **k)
            if isinstance(r, dict) and r.get("ok") is False:
                tc.ok, tc.error = False, r.get("error")
                self.rep.errors.append(f"{fn.__name__}: {r.get('error')}")
            elif isinstance(r, dict) and r.get("source") and r["source"] not in self.rep.data_sources:
                self.rep.data_sources.append(r["source"])
            return r
        except Exception as ex:   # an agent failure must never crash the investigation
            tc.ok, tc.error = False, f"{ex.__class__.__name__}: {ex}"
            self.rep.errors.append(f"{fn.__name__} raised {tc.error}")
            return {"ok": False, "error": tc.error}
        finally:
            tc.seconds = round(time.time() - t, 3)
            self.rep.tool_calls.append(tc)

    def add(self, statement, entity_type, entity_id, metric=None, value=None, stat=None, confidence="Medium", limitation="", rec=""):
        f = Finding(statement=statement, metric=metric, value=value, stat_result=stat, confidence=confidence, limitation=limitation)
        f.evidence_id = store.add_evidence(statement, self.name, entity_type, entity_id, metric, value, stat, confidence, limitation, rec, self.run_id, db=self.db)
        self.rep.findings.append(f)
        return f

    def start(self, task):
        self.rep = AgentReport(agent=self.name, task=task)
        return self.rep

    def finish(self):
        if self.rep.errors:
            self.rep.status = "failed" if not self.rep.findings else "partial"
        return self.rep


class SellerAgent(Base):
    name = "Seller Investigation Agent"

    def run(self, sid):
        self.start(f"Profile seller {sid[:8]} and list statistical anomaly flags")
        p = self.call(tools.seller_profile, sid)
        if not p.get("ok"):
            return self.finish()
        sid = p["seller_id"]; P = p["platform"]
        self.add(f"Seller {sid[:8]} has {p['n_orders']} analysed orders; mean review {p['mean_review_score']:.2f}; mean freight ratio {p['mean_freight_ratio']:.2f}.",
                 "seller", sid, "n_orders", p["n_orders"], None, "High" if p["n_orders"] >= 100 else "Medium" if p["n_orders"] >= 30 else "Low",
                 "Small sample - interpret with caution." if p["small_sample"] else "")
        an = self.call(tools.anomalies, sid, None, 10)
        if an.get("ok"):
            for r in an["rows"][:4]:
                self.add(f"[{r['anomaly_id']}] {r['why_flagged']}", "seller", sid, r["detector"], r["measured"], {"p_adj": r["p_adj"] or None, "score": r["score"]},
                         "Medium", r["caveat"], r["recommended_investigation"])
            if not an["rows"]:
                self.rep.uncertainty.append("No anomaly detector flagged this seller (or it has fewer than 30 orders).")
        if p["small_sample"]:
            self.rep.uncertainty.append("Fewer than 30 orders: statistical tests have little power.")
        self.rep.recommended_next_step = "Delivery Risk and Customer Review agents examine where the problems come from."
        return self.finish()


class DeliveryRiskAgent(Base):
    name = "Delivery Risk Agent"

    def run(self, sid):
        self.start(f"Assess delivery performance of seller {sid[:8]} against the platform")
        p = self.call(tools.seller_profile, sid)
        if not p.get("ok"):
            return self.finish()
        sid = p["seller_id"]; P = p["platform"]; n = p["n_orders"]
        k = round(p["late_rate"] * n)
        t = tools.binom_vs_platform(k, n, P["late_rate"])
        self.add(f"Late-delivery rate {p['late_rate']:.1%} ({k}/{n}) vs {P['late_rate']:.1%} platform-wide (binomial p={t['p_value']:.3g}).",
                 "seller", sid, "late_rate", p["late_rate"], t, conf(n, t["p_value"]), "Late = delivered after the estimated date (date level).",
                 "Review the seller's dispatch process." if p["late_rate"] > P["late_rate"] else "")
        self.add(f"Handling {p['mean_handling_days']:.1f} d (platform {P['mean_handling_days']:.1f}), carrier transit {p['mean_carrier_days']:.1f} d (platform {P['mean_carrier_days']:.1f}).",
                 "seller", sid, "stage_days", {"handling": p["mean_handling_days"], "carrier": p["mean_carrier_days"]}, None, "Medium",
                 "Stage means ignore product mix and distance.")
        slow = "seller handling" if (p["mean_handling_days"] - P["mean_handling_days"]) > (p["mean_carrier_days"] - P["mean_carrier_days"]) else "carrier transit"
        self.rep.uncertainty.append(f"Hypothesis: the larger gap to the platform is in {slow} - descriptive only; distance and product mix not controlled.")
        try:
            import pandas as pd
            from ecx2.common import V2_DATA
            rs = pd.read_parquet(V2_DATA / "risk_scores_test.parquet")
            x = rs[rs.primary_seller_id == sid]
            if len(x) >= 10:
                self.add(f"Early-risk model (checkout stage) mean score {x.risk_A0.mean():.3f} vs {rs.risk_A0.mean():.3f} for all test-period orders ({len(x)} orders).",
                         "seller", sid, "mean_risk_A0", float(x.risk_A0.mean()), None, "Low", "Checkout-stage model is weak (test ROC-AUC ~0.65); a score is a ranking signal, not a probability.")
            else:
                self.rep.uncertainty.append("Too few test-period orders for a model-based risk view of this seller.")
        except Exception as ex:   # noqa
            self.rep.errors.append(f"risk scores unavailable: {ex.__class__.__name__}")
        self.rep.recommended_next_step = "Research Validation Agent tests whether delays explain the seller's dissatisfaction."
        return self.finish()


class ReviewAgent(Base):
    name = "Customer Review Agent"

    def run(self, sid):
        self.start(f"Summarise what customers of seller {sid[:8]} complain about")
        c = self.call(tools.seller_complaints, sid)
        if not c.get("ok"):
            return self.finish()
        sid = c["seller_id"]
        for r in c["categories"][:3]:
            if r["count"] >= 3 and r["seller_share"] > r["platform_share"] * 1.3:
                self.add(f"{r['seller_share']:.1%} of this seller's reviews mention '{r['category']}' ({r['count']} reviews) vs {r['platform_share']:.1%} platform-wide (p={r['p_value']:.3g}).",
                         "seller", sid, f"complaint_{r['category']}", r["seller_share"], {"p": r["p_value"]}, conf(c["n_reviews"], r["p_value"]),
                         "Rule-based Portuguese taxonomy; micro-F1 ~0.66 on a 150-review sample.", f"Read recent '{r['category']}' reviews.")
        if not self.rep.findings:
            self.rep.uncertainty.append("No complaint category is clearly above the platform share.")
        s = self.call(tools.sample_reviews, sid, 3)
        if s.get("ok") and s["reviews"]:
            self.add("Recent negative reviews (verbatim, for human reading): " + " | ".join(f"[{r['order_id'][:8]}] {r['text'][:110]}" for r in s["reviews"]),
                     "seller", sid, "sample_reviews", len(s["reviews"]), None, "Low", "Illustrative sample, not representative.")
        self.rep.recommended_next_step = "Compare these complaint themes with the delivery findings."
        return self.finish()


class ResearchValidationAgent(Base):
    name = "Research Validation Agent"

    def run(self, sid):
        self.start(f"Statistically test the hypotheses about seller {sid[:8]}")
        v = self.call(tools.seller_late_vs_dissat, sid)
        self.verdicts = []
        if not v.get("ok"):
            return self.finish()
        sid = v["seller_id"]
        # H1: dissatisfaction above expectation given category mix?
        n = v["n_late"] + v["n_ontime"]
        p = v.get("excess_p_value")
        h1 = ("Supported" if p is not None and p < 0.01 and v["dissat_observed"] > v["dissat_expected_given_category_mix"] * 1.3 else
              "Not supported" if p is not None and p >= 0.05 else "Inconclusive")
        self.add(f"H1 'Dissatisfaction is above what the seller's category mix predicts': observed {v['dissat_observed']:.1%} vs expected {v['dissat_expected_given_category_mix']:.1%} (p={p:.3g}) -> {h1}.",
                 "seller", sid, "H1_excess_dissatisfaction", v["dissat_observed"], {"expected": v["dissat_expected_given_category_mix"], "p": p, "verdict": h1}, conf(n, p), "Category-mix standardisation only.")
        self.verdicts.append({"hypothesis": "H1 excess dissatisfaction vs category mix", "verdict": h1, "p": p})
        # H2: late deliveries drive it
        if v.get("fisher_p") is not None:
            late_share = None
            a = round(v["dissat_if_late"] * v["n_late"]); c = round(v["dissat_if_ontime"] * v["n_ontime"])
            late_share = a / (a + c) if a + c else None
            h2 = "Supported (partly)" if v["fisher_p"] < 0.05 and late_share is not None and late_share >= 0.25 else "Weak: late orders are only a minor share of the dissatisfaction" if v["fisher_p"] < 0.05 else "Not supported"
            self.add(f"H2 'Late deliveries drive this seller's dissatisfaction': {v['dissat_if_late']:.0%} dissatisfied when late vs {v['dissat_if_ontime']:.0%} on time (Fisher p={v['fisher_p']:.3g}); late orders account for {late_share:.0%} of its dissatisfied orders -> {h2}.",
                     "seller", sid, "H2_late_drives_dissat", v["dissat_if_late"], {"fisher_p": v["fisher_p"], "odds_ratio": v["odds_ratio"], "late_share_of_dissat": late_share, "verdict": h2},
                     conf(n, v["fisher_p"]), "Association within one seller; small cells possible.")
            self.verdicts.append({"hypothesis": "H2 late deliveries drive dissatisfaction", "verdict": h2, "p": v["fisher_p"], "late_share_of_dissatisfaction": late_share})
        else:
            self.rep.uncertainty.append("H2 not testable: fewer than 5 late or on-time orders.")
            self.verdicts.append({"hypothesis": "H2 late deliveries drive dissatisfaction", "verdict": "Insufficient data", "p": None})
        self.rep.recommended_next_step = "Action Planning Agent proposes actions for human approval."
        return self.finish()


class ActionPlanningAgent(Base):
    name = "Action Planning Agent"

    def run(self, sid, reports, verdicts):
        self.start(f"Propose reviewable actions for seller {sid[:8]} (none are executed)")
        self.proposed = []
        ev = lambda agent_prefix, key=None: [f.evidence_id for r in reports if r.agent.startswith(agent_prefix) for f in r.findings
                                              if key is None or (f.metric or "").startswith(key)]
        vd = {v["hypothesis"][:2]: v for v in verdicts}
        seller_ev = ev("Seller", None)[:1]

        def prop(title, prio, role, reason, ids):
            ids = [i for i in ids if i]
            if not ids:
                return
            aid = store.propose_action(title, reason, prio, reason, ids, role, source=self.name, db=self.db)
            self.proposed.append(ProposedAction(action_id=aid, title=title, priority=prio, owner_role=role, reason=reason, evidence_ids=ids))
            self.add(f"Proposed {aid}: {title} [{prio}] - awaiting human approval.", "action", aid, "proposed_action", prio, None, "Medium", "Proposal only.", "")
        if vd.get("H1", {}).get("verdict") == "Supported":
            prop(f"Review seller {sid[:8]} listings and recent negative reviews", "High", "Seller Manager",
                 "Dissatisfaction is significantly above what the seller's category mix predicts.", ev("Research", "H1") + seller_ev)
        h2 = vd.get("H2", {}).get("verdict", "")
        late_ev = ev("Delivery", "late_rate")
        if h2.startswith("Supported"):
            prop(f"Examine dispatch/carrier delays for seller {sid[:8]}", "High", "Logistics Manager", "Late orders show much higher dissatisfaction for this seller.", ev("Research", "H2") + late_ev)
        elif any(f.metric == "late_rate" and (f.stat_result or {}).get("p_value", 1) < 0.01 and (f.value or 0) > 0.1 for r in reports for f in r.findings):
            prop(f"Check delivery estimates and dispatch speed for seller {sid[:8]}", "Medium", "Logistics Manager", "Late-delivery rate is significantly above the platform rate.", late_ev)
        comp = ev("Customer", "complaint_")
        if comp:
            prop(f"Customer-support review of '{comp and next(f.metric for r in reports for f in r.findings if f.evidence_id == comp[0])[10:]}' complaints for seller {sid[:8]}", "Medium", "Customer Support Lead",
                 "A complaint theme is more frequent than on the platform.", comp[:2])
        if not self.proposed:
            self.rep.uncertainty.append("No hypothesis was supported strongly enough to propose an action; monitor only.")
        self.rep.recommended_next_step = "A human reviewer approves, rejects or edits each proposal in the Action Center."
        return self.finish()


class MonthAgent(Base):
    name = "Delivery Risk Agent"

    def run(self, ym):
        self.start(f"Assess platform delivery and satisfaction in {ym}")
        m = self.call(tools.month_profile, ym)
        if not m.get("ok"):
            return self.finish()
        self.add(f"In {ym}: late rate {m['late_rate']:.1%} vs {m['late_rate_other_months']:.1%} in other months (p={m['late_p_value']:.3g}); dissatisfied {m['dissatisfied_rate']:.1%} vs {m['dissatisfied_rate_other_months']:.1%} (p={m['dissatisfied_p_value']:.3g}).",
                 "platform_month", ym, "late_rate", m["late_rate"], {"p_value": m["late_p_value"]}, conf(m["n_orders"], m["late_p_value"]), "Calendar months only; no external event data.",
                 "Break the month down by region and carrier stage.")
        if m["worst_states_by_late_rate"]:
            w = m["worst_states_by_late_rate"]
            self.add("Highest late rates by customer state: " + ", ".join(f"{r['customer_state']} {r['late']:.0%} (n={r['n']})" for r in w), "platform_month", ym, "state_late", w, None, "Medium", "States with >=30 orders only.")
        self.rep.uncertainty.append("Cause of any surge cannot be determined from this dataset (no holiday/strike/stock data) - hypotheses only.")
        self.rep.recommended_next_step = "Human review of regional logistics for that month."
        return self.finish()


class Supervisor:
    """Plans the investigation, calls agents in order, validates, then summarises. Deterministic."""

    def __init__(self, db=None):
        self.db = db

    def investigate(self, target_type, target_id, question=None):
        run_id = "RUN-" + uuid.uuid4().hex[:8]
        reports, verdicts, actions = [], [], []
        if target_type == "seller":
            sid, err = tools.resolve_seller(target_id)
            if err:
                return {"ok": False, "error": err}
            q = question or f"Why might seller {sid[:8]} be associated with poor customer experience?"
            plan = ["Seller Investigation Agent: profile + anomaly flags", "Delivery Risk Agent: delivery performance vs platform",
                    "Customer Review Agent: complaint themes", "Research Validation Agent: statistical tests of hypotheses",
                    "Action Planning Agent: proposals for human approval"]
            for A in (SellerAgent, DeliveryRiskAgent, ReviewAgent):
                reports.append(A(self.db, run_id).run(sid))
            rv = ResearchValidationAgent(self.db, run_id); reports.append(rv.run(sid)); verdicts = rv.verdicts
            ap = ActionPlanningAgent(self.db, run_id); reports.append(ap.run(sid, reports[:-1], verdicts)); actions = ap.proposed
            tid = sid
        elif target_type == "platform_month":
            q = question or f"What happened to delivery and satisfaction in {target_id}?"
            plan = ["Delivery Risk Agent: month vs other months", "Customer Review Agent: (seller-level only - skipped)", "Action Planning Agent: proposals for human approval"]
            ma = MonthAgent(self.db, run_id); reports.append(ma.run(target_id)); tid = target_id
            if not any(f.metric == "late_rate" for f in reports[0].findings):
                return {"ok": False, "error": "; ".join(reports[0].errors) or "Month not analysable."}
            f = next(f for f in reports[0].findings if f.metric == "late_rate")
            if (f.stat_result or {}).get("p_value", 1) < 0.01 and f.value > 0.1:
                ap = ActionPlanningAgent(self.db, run_id); ap.start("Propose actions for platform month")
                aid = store.propose_action(f"Review regional logistics for {target_id}", "Late rate was significantly elevated in this month.", "Medium", "Late rate significantly above other months.", [f.evidence_id], "Logistics Manager", source=ap.name, db=self.db)
                actions.append(ProposedAction(action_id=aid, title=f"Review regional logistics for {target_id}", priority="Medium", owner_role="Logistics Manager", reason="Late rate significantly above other months.", evidence_ids=[f.evidence_id]))
        else:
            return {"ok": False, "error": "target_type must be 'seller' or 'platform_month'."}
        confs = [f.confidence for r in reports for f in r.findings if f.confidence]
        oc = "High" if confs and confs.count("High") >= len(confs) / 2 else "Low" if not confs or confs.count("Low") > len(confs) / 2 else "Medium"
        failed = [r.agent for r in reports if r.status == "failed"]
        parts = [f"{v['hypothesis']}: {v['verdict']}" for v in verdicts]
        summary = (f"Investigation of {target_type} {tid[:8] if target_type == 'seller' else tid}. " +
                   (("Validated hypotheses - " + "; ".join(parts) + ". ") if parts else "") +
                   f"{len(actions)} action(s) proposed for human review. " + (f"Agents that failed: {', '.join(failed)}. " if failed else "") +
                   "All findings are statistical associations from historical data.")
        rep = InvestigationReport(run_id=run_id, target_type=target_type, target_id=tid, question=q, plan=plan, reports=reports, verdicts=verdicts,
                                  actions=actions, summary=summary, overall_confidence=oc, limitations=LIMITS)
        return {"ok": True, "report": rep}
