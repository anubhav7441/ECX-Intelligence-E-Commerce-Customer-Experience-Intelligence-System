"""Operations layer routes: alerts, root cause, playbooks, outcome tracking, API keys and the key-protected /api/v1 API.

Login-protected pages and /api/ops/* use the normal session guard. /api/v1/* uses API keys (Authorization: Bearer ecx_...).
Keys are created by an administrator, shown once, stored only as a hash, rate-limited per key and can be revoked."""
import re
from datetime import date
from flask import Blueprint, current_app, g, jsonify, render_template, request
from ecx2 import store as astore
from ecx2.common import clean
from ecx3 import monitor, outcomes, playbooks, rca, store
from app.auth import security as sec

ops = Blueprint("ops", __name__)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def odb():
    return current_app.config.get("OPS_DB")


def adb_():
    return current_app.config.get("APP_DB")


def err(msg, code=400):
    return jsonify(error=msg), code


def body():
    b = request.get_json(silent=True)
    return b if isinstance(b, dict) else {}


def actor():
    u = g.get("user")
    return (u["email"] if u else "local-user")


def _orders():
    from ecx2.common import orders
    return orders()


def _good_date(s):
    if not isinstance(s, str) or not DATE_RE.match(s):
        return False
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


def _analysis(alert, w=28, b=84):
    """Root-cause breakdown, or None for alerts that have none (data drift)."""
    if alert.get("kind") == "data_drift":
        return None
    return rca.analyse(_orders(), alert, w, b)


# ------------------------------------------------------------------ pages
@ops.route("/ops")
def center():
    return render_template("ops.html", active="ops", counts=store.alert_counts(odb()), jobs=store.list_jobs(5, odb()),
                           alerts=store.list_alerts(limit=100, db=odb()), note_counts=store.note_counts(odb()))


@ops.route("/ops/alert/<alert_id>")
def alert_page(alert_id):
    a = store.get_alert(alert_id, odb())
    if a is None:
        return render_template("error.html", code=404, message="Alert not found."), 404
    try:
        an = _analysis(a)
        recs = playbooks.recommend(a, an)
    except (ValueError, FileNotFoundError) as e:
        an, recs = None, []
    act = astore.get_action(a["action_id"], db=adb_()) if a.get("action_id") else None
    return render_template("ops_alert.html", active="ops", a=a, an=an, recs=recs, act=act, notes=store.list_notes(alert_id, odb()), outs=store.list_outcomes(a.get("action_id"), 10, odb()) if a.get("action_id") else [])


@ops.route("/ops/outcomes")
def outcomes_page():
    return render_template("ops_outcomes.html", active="ops", outs=store.list_outcomes(limit=50, db=odb()),
                           actions=[x for x in astore.list_actions(db=adb_()) if x["status"] in ("approved", "in_progress", "completed")])


@ops.route("/admin/keys")
def keys_page():
    return render_template("ops_keys.html", active="admin", keys=store.list_api_keys(odb()))


# ------------------------------------------------------------------ session API
@ops.post("/api/ops/run")
def api_run():
    b = body()
    a = b.get("as_of")
    if a is not None and not _good_date(a):
        return err("as_of must be a date like 2018-06-30")
    try:
        return jsonify(clean(monitor.run_monitor(db=odb(), d=_orders(), as_of=a)))
    except FileNotFoundError as e:
        return err(str(e), 503)


@ops.get("/api/ops/alerts")
def api_alerts():
    s, v = request.args.get("state"), request.args.get("severity")
    if s and s not in store.ALERT_STATES or v and v not in store.SEVERITIES:
        return err("unknown state or severity filter")
    return jsonify(alerts=store.list_alerts(s, v, db=odb()), counts=store.alert_counts(odb()))


@ops.get("/api/ops/alerts/<alert_id>")
def api_alert(alert_id):
    a = store.get_alert(alert_id, odb())
    if a is None:
        return err("alert not found", 404)
    try:
        an = _analysis(a)
        return jsonify(clean({"alert": a, "analysis": an, "recommendations": playbooks.recommend(a, an)}))
    except ValueError as e:
        return jsonify(alert=a, analysis=None, recommendations=[], note=str(e))


@ops.post("/api/ops/alerts/<alert_id>/state")
def api_alert_state(alert_id):
    b = body()
    try:
        return jsonify(store.set_alert_state(alert_id, b.get("state"), actor(), b.get("note", ""), odb()))
    except KeyError:
        return err("alert not found", 404)
    except ValueError as e:
        return err(str(e))


@ops.get("/api/ops/alerts/<alert_id>/notes")
def api_notes(alert_id):
    if store.get_alert(alert_id, odb()) is None:
        return err("alert not found", 404)
    return jsonify(notes=store.list_notes(alert_id, odb()))


@ops.post("/api/ops/alerts/<alert_id>/notes")
def api_add_note(alert_id):
    try:
        return jsonify(store.add_note(alert_id, actor(), body().get("body", ""), odb())), 201
    except KeyError:
        return err("alert not found", 404)
    except ValueError as e:
        return err(str(e))


@ops.post("/api/ops/alerts/<alert_id>/propose")
def api_propose(alert_id):
    a = store.get_alert(alert_id, odb())
    if a is None:
        return err("alert not found", 404)
    if a.get("action_id"):
        return err(f"an action ({a['action_id']}) already exists for this alert", 409)
    try:
        aid = playbooks.propose_from_alert(a, _analysis(a), int(body().get("playbook_index", 0)), db=adb_(), ops_db=odb())
    except (ValueError, TypeError) as e:
        return err(str(e))
    return jsonify(astore.get_action(aid, db=adb_())), 201


@ops.post("/api/ops/outcomes")
def api_outcome():
    b = body()
    for k in ("metric", "entity_type", "entity_id", "action_date"):
        if not b.get(k):
            return err(f"{k} is required")
    if not _good_date(b["action_date"]):
        return err("action_date must be a date like 2018-06-30")
    if b["entity_type"] not in ("platform", "state", "category", "seller"):
        return err("entity_type must be platform, state, category or seller")
    try:
        w = int(b.get("window_days", 28))
        oid, r = outcomes.record(_orders(), b.get("action_id"), b.get("alert_id"), b["metric"], b["entity_type"], b["entity_id"], b["action_date"], w, actor(), odb())
    except (ValueError, TypeError) as e:
        return err(str(e))
    return jsonify(clean({"outcome_id": oid, "result": r})), 201


@ops.get("/api/ops/outcomes")
def api_outcomes():
    return jsonify(outcomes=clean(store.list_outcomes(request.args.get("action_id"), 100, odb())))


@ops.post("/api/ops/digest")
def api_digest():
    """Build and save the digest files, return the summary (and a text version to read in the page)."""
    from ecx3 import digest
    a = body().get("as_of")
    if a is not None and not _good_date(a):
        return err("as_of must be a date like 2018-06-30")
    g = digest.build(_orders(), odb(), adb_(), a)
    paths = digest.write(g, current_app.config.get("DIGEST_DIR"))
    return jsonify(clean({"digest": g, "text": digest.to_text(g), "files": {k: v.split("/")[-1] for k, v in paths.items()}}))


# ------------------------------------------------------------------ admin: API keys (the global guard enforces admin for /admin and /api/admin)
@ops.get("/api/admin/keys")
def api_keys():
    return jsonify(keys=store.list_api_keys(odb()))


@ops.post("/api/admin/keys")
def api_key_create():
    b = body()
    try:
        kid, key = store.create_api_key(b.get("name", ""), b.get("scope", "read"), actor(), odb())
    except ValueError as e:
        return err(str(e))
    from app.auth import db as adb
    adb.audit("api_key_created", f"{kid} scope={b.get('scope', 'read')}", g.get("user"), request.remote_addr)
    return jsonify(key_id=kid, key=key, note="Copy this key now. It is shown only once and cannot be recovered."), 201


@ops.post("/api/admin/keys/<key_id>/revoke")
def api_key_revoke(key_id):
    try:
        store.revoke_api_key(key_id, odb())
    except KeyError:
        return err("key not found", 404)
    from app.auth import db as adb
    adb.audit("api_key_revoked", key_id, g.get("user"), request.remote_addr)
    return jsonify(ok=True)


# ------------------------------------------------------------------ /api/v1 (API keys)
def _key_auth(need):
    h = request.headers.get("Authorization", "")
    if not h.lower().startswith("bearer "):
        return None, (err("Missing API key. Send 'Authorization: Bearer <key>'.", 401))
    k = store.verify_api_key(h[7:].strip(), odb())
    if k is None:
        return None, err("Invalid or revoked API key.", 401)
    if need == "write" and k["scope"] != "write":
        return None, err("This key has read-only access.", 403)
    if sec.rate_limited(f"k{k['key_id']}", current_app.config["API_RATE_LIMIT_PER_MINUTE"]):
        return None, err("Rate limit reached for this key. Please wait a minute.", 429)
    return k, None


@ops.get("/api/v1/health")
def v1_health():
    k, e = _key_auth("read")
    return e if e else jsonify(status="ok", key=k["key_id"], scope=k["scope"])


@ops.get("/api/v1/alerts")
def v1_alerts():
    k, e = _key_auth("read")
    if e:
        return e
    s, v = request.args.get("state"), request.args.get("severity")
    if s and s not in store.ALERT_STATES or v and v not in store.SEVERITIES:
        return err("unknown state or severity filter")
    return jsonify(alerts=store.list_alerts(s, v, db=odb()), counts=store.alert_counts(odb()))


@ops.get("/api/v1/alerts/<alert_id>")
def v1_alert(alert_id):
    k, e = _key_auth("read")
    if e:
        return e
    a = store.get_alert(alert_id, odb())
    return jsonify(alert=a) if a else err("alert not found", 404)


@ops.post("/api/v1/alerts/<alert_id>/acknowledge")
def v1_ack(alert_id):
    k, e = _key_auth("write")
    if e:
        return e
    b = body()
    try:
        return jsonify(store.set_alert_state(alert_id, "acknowledged", f"api:{k['key_id']}", b.get("note", ""), odb()))
    except KeyError:
        return err("alert not found", 404)


@ops.get("/api/v1/actions")
def v1_actions():
    k, e = _key_auth("read")
    if e:
        return e
    s = request.args.get("status")
    if s and s not in astore.STATUSES:
        return err("unknown status")
    return jsonify(actions=astore.list_actions(s, db=adb_()))


@ops.get("/api/v1/outcomes")
def v1_outcomes():
    k, e = _key_auth("read")
    if e:
        return e
    return jsonify(outcomes=clean(store.list_outcomes(request.args.get("action_id"), 100, odb())))
