"""Playbooks: turn a root-cause finding into concrete, reviewable steps.

Playbooks are written rules (not model output). Each one lists steps, a suggested owner, the effort and WHY it applies.
Impact is stated only as "events in the window linked to this driver" (a real count from the data). No percentage improvement is promised,
because that would need an experiment. A proposed action goes through the existing human approval flow."""
from ecx2 import store as astore
from ecx3 import store

PLAYBOOKS = {
    ("late_rate", "seller"): dict(
        title="Review this seller's delivery performance", owner="Seller Manager", effort="Medium",
        steps=["Check the seller's recent orders: handling time, shipping-limit date and missed promises.",
               "Contact the seller, share the numbers and ask for a corrective plan.",
               "Check stock and order-processing capacity (a sudden volume rise is a common cause).",
               "Agree a follow-up date and re-check the late rate after the monitoring window."]),
    ("late_rate", "handling"): dict(
        title="Reduce seller handling time", owner="Seller Manager", effort="Medium",
        steps=["List the sellers with the longest handling times in this segment.",
               "Send a reminder of the handling-time promise and the shipping-limit date.",
               "Consider a handling-time limit or alert for slow sellers.",
               "Re-check handling time and late rate after two weeks."]),
    ("late_rate", "transit"): dict(
        title="Review the carrier on slow routes", owner="Logistics Manager", effort="High",
        steps=["Compare carrier transit time on the affected routes with the previous period.",
               "Raise the issue with the carrier and ask for a cause.",
               "Consider moving volume on these routes to another carrier if the delay persists."]),
    ("late_rate", "state"): dict(
        title="Check delivery to this region", owner="Logistics Manager", effort="Medium",
        steps=["Check whether one carrier or one seller group serves most late orders in this region.",
               "Compare the promised delivery date with actual transit time for the region.",
               "If the promise is too tight, widen the estimated delivery date for this region."]),
    ("late_rate", "seller_state"): dict(
        title="Check shipments leaving this origin region", owner="Logistics Manager", effort="Medium",
        steps=["Check pickup and hand-off delays for sellers in this region.",
               "Look for one seller or one carrier hub that accounts for most delays.",
               "Share the findings with the sellers and the carrier."]),
    ("late_rate", "distance"): dict(
        title="Review the delivery promise for this distance band", owner="Operations Manager", effort="Medium",
        steps=["Compare promised vs actual delivery days for this distance band.",
               "If most late orders are promised too tightly, widen the estimate for this band.",
               "Re-check after the change."]),
    ("late_rate", "category"): dict(
        title="Review fulfilment for this product category", owner="Category Manager", effort="Medium",
        steps=["Check which sellers and carriers handle this category's late orders.",
               "Check for bulky or fragile items that need a different shipping method.",
               "Agree a fix with the category's top sellers."]),
    ("dissatisfaction_rate", "seller"): dict(
        title="Review this seller's customer feedback", owner="Seller Manager", effort="Medium",
        steps=["Read the recent low-rated reviews for the seller and group them by reason.",
               "Separate delivery problems from product problems.",
               "Contact the seller with the findings and agree what will change."]),
    ("dissatisfaction_rate", "category"): dict(
        title="Review product quality and listings in this category", owner="Category Manager", effort="Medium",
        steps=["Read low-rated reviews and group them (wrong item, damaged, not as described).",
               "Check listing quality: photos and description length.",
               "Flag repeated problem products for review."]),
    ("dissatisfaction_rate", "state"): dict(
        title="Check service quality for this region", owner="Customer Support Lead", effort="Medium",
        steps=["Check if the unhappy orders in this region are also late.",
               "Read a sample of complaints for common reasons.",
               "Share findings with logistics or support as needed."]),
}
GENERIC = dict(title="Investigate the change and assign an owner", owner="Operations Manager", effort="Low",
               steps=["Confirm the change is real (check for data problems and small samples).",
                      "Break it down by region, category and seller to find where it comes from.",
                      "Assign an owner and a follow-up date."])
DRIFT = dict(title="Check whether the models need retraining", owner="Data Analyst", effort="Medium",
             steps=["Confirm the shift is real: check whether it is seasonal or caused by a data or reporting change.",
                    "Compare the models' recent predictions with what actually happened, once the reviews have arrived.",
                    "If the models are clearly less accurate, retrain on newer data and compare before replacing the saved models.",
                    "Record the decision and re-run the drift check next period."])
PRIORITY = {"critical": "Critical", "warning": "Medium", "info": "Low"}


def recommend(alert, analysis):
    """Ranked list of playbook suggestions for an alert and its root-cause analysis."""
    if alert.get("kind") == "data_drift":
        return [{**DRIFT, "playbook": "data_drift/retrain_check", "why": alert["title"], "linked_events": 0.0, "steps": list(DRIFT["steps"])}]
    items, seen = [], set()

    def add(key, why, events):
        if key in seen:
            return
        pb = PLAYBOOKS.get(key)
        if pb is None:
            return
        seen.add(key)
        items.append({**pb, "playbook": f"{key[0]}/{key[1]}", "why": why, "linked_events": round(float(events), 1), "steps": list(pb["steps"])})

    m = alert["metric"]
    if alert["entity_type"] == "seller":
        add((m, "seller"), "The alert is about one seller.", analysis["total_excess_events"])
    stage = analysis.get("stage_check") or {}
    slow = stage.get("slower_stages") or []
    if any("handling" in s.lower() for s in slow):
        add((m, "handling"), "Seller handling time is significantly slower than in the baseline.", analysis["total_excess_events"])
    if any("transit" in s.lower() for s in slow):
        add((m, "transit"), "Carrier transit time is significantly slower than in the baseline.", analysis["total_excess_events"])
    for dr in analysis["drivers"]:
        why = (f"{dr['dimension_label']} '{dr['level']}' holds {dr['share_of_excess'] * 100:.0f}% of the extra events "
               f"(rate {dr['rate_current'] * 100:.1f}% vs {dr['rate_rest_current'] * 100:.1f}% elsewhere).")
        add((m, dr["dimension"]), why, dr["excess_events"])
    if not items:
        items.append({**GENERIC, "playbook": "generic", "why": "No single driver stands out in the data.", "linked_events": round(analysis["total_excess_events"], 1), "steps": list(GENERIC["steps"])})
    return items[:5]


def propose_from_alert(alert, analysis, playbook_index=0, actor="monitor", db=None, ops_db=None):
    """Record the evidence and propose an action (status 'proposed': a named person must approve it). Links it to the alert."""
    if analysis is None:                  # data-drift alerts have no root-cause breakdown
        analysis = {"drivers": [], "stage_check": None, "confidence": "Medium", "top3_share": 0.0, "top_share": 0.0, "total_excess_events": 0.0,
                    "limitations": ["Input distributions changed; this does not by itself show the models are wrong."]}
    recs = recommend(alert, analysis)
    if not (0 <= playbook_index < len(recs)):
        raise ValueError("playbook_index out of range")
    pb = recs[playbook_index]
    stats_txt = {"p_adjusted": alert.get("p_value"), "effect_points": alert.get("effect"), "confidence": analysis["confidence"], "top3_share": analysis["top3_share"]}
    ev = astore.add_evidence(alert["title"], "ecx3.monitor", alert["entity_type"], alert["entity_id"], alert["metric"],
                             model_output={"drivers": analysis["drivers"][:3], "stage_check": analysis.get("stage_check")}, stat_result=stats_txt,
                             confidence=analysis["confidence"], limitation=analysis["limitations"][0],
                             recommendation=pb["title"], run_id=alert["alert_id"], db=db)
    owner = pb["owner"] if pb["owner"] in astore.ROLES else "Operations Manager"
    aid = astore.propose_action(f"{pb['title']} ({alert['entity_type']} {alert['entity_id']})"[:200],
                                "\n".join(f"{i + 1}. {s}" for i, s in enumerate(pb["steps"])), PRIORITY.get(alert["severity"], "Medium"),
                                pb["why"], [ev], owner, source="ecx3.playbook", db=db)
    store.link_action(alert["alert_id"], aid, ops_db)
    return aid
