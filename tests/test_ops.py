"""Operations layer: alerts, root cause, playbooks, outcome measurement, API keys and the /api/v1 API."""
import numpy as np
import pandas as pd
import pytest
from app import create_app
from config import AuthTestConfig, TestConfig
from app.auth import security as sec, db as adb
from ecx2 import store as astore
from ecx3 import monitor, outcomes, playbooks, rca, store

PW = "Str0ngPass-123"


def synth(seed=0, n_days=140, per_day=40, spike=None, effect_after=None):
    """Synthetic orders. Base late rate 6%, unhappy 10%. `spike` = (state, from_day, late_rate) raises one state's late rate late in the period.
    `effect_after` = (state, day, new_rate) models a fix that lowers that state's late rate from `day`."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2018-01-01")
    rows = []
    states = ["SP", "RJ", "MG", "BA", "RS"]
    for day in range(n_days):
        for _ in range(per_day):
            st = states[rng.integers(0, 5)]
            late = 0.06
            if spike and st == spike[0] and day >= spike[1]:
                late = spike[2]
            if effect_after and st == effect_after[0] and day >= effect_after[1]:
                late = effect_after[2]
            is_late = int(rng.random() < late)
            rows.append({"order_purchase_timestamp": start + pd.Timedelta(days=day, hours=int(rng.integers(0, 24))), "customer_state": st,
                         "primary_category": f"cat{rng.integers(0, 4)}", "primary_seller_id": f"s{rng.integers(0, 8)}", "seller_state": "SP",
                         "is_late": is_late, "dissatisfied": int(rng.random() < (0.5 if is_late else 0.08)), "distance_km": float(rng.integers(20, 2500)),
                         "items_price": float(rng.integers(20, 400)), "total_weight_g": float(rng.integers(100, 6000)), "freight_ratio": float(rng.random()),
                         "seller_handling_days": float(rng.integers(0, 6)), "carrier_transit_days": float(rng.integers(2, 15))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ store
def test_alert_dedupe_and_state_rules(tmp_path):
    db = tmp_path / "o.db"
    a = {"dedupe_key": "k1", "kind": "rate_increase", "metric": "late_rate", "entity_type": "state", "entity_id": "SP", "severity": "warning", "title": "t"}
    i1, new1 = store.upsert_alert(a, db)
    i2, new2 = store.upsert_alert(dict(a, severity="critical"), db)
    assert i1 == i2 and new1 and not new2
    assert store.get_alert(i1, db)["severity"] == "critical"
    with pytest.raises(ValueError):
        store.set_alert_state(i1, "resolved", "Asha", "", db)                # note required to resolve
    with pytest.raises(ValueError):
        store.set_alert_state(i1, "acknowledged", " ", "", db)               # a named person is required
    assert store.set_alert_state(i1, "resolved", "Asha", "fixed", db)["state"] == "resolved"
    i3, new3 = store.upsert_alert(a, db)                                      # a resolved problem can come back as a new alert
    assert new3 and i3 != i1


def test_api_key_is_hashed_and_revocable(tmp_path):
    db = tmp_path / "o.db"
    kid, key = store.create_api_key("bi tool", "read", "admin@x.com", db)
    import sqlite3
    raw = sqlite3.connect(db).execute("SELECT key_hash, prefix FROM api_keys").fetchone()
    assert key not in raw[0] and key.startswith("ecx_") and len(raw[0]) == 64          # only a SHA-256 hash is stored
    assert store.verify_api_key(key, db)["key_id"] == kid
    assert store.verify_api_key(key + "x", db) is None and store.verify_api_key("", db) is None and store.verify_api_key("nokey", db) is None
    store.revoke_api_key(kid, db)
    assert store.verify_api_key(key, db) is None
    with pytest.raises(ValueError):
        store.create_api_key("x", "admin", "a", db)
    with pytest.raises(ValueError):
        store.create_api_key("  ", "read", "a", db)
    assert all("key_hash" not in k for k in store.list_api_keys(db))


# ------------------------------------------------------------------ monitor
def test_monitor_finds_planted_spike_and_stays_quiet_on_stable_data():
    d = synth(spike=("SP", 100, 0.30))
    al = monitor.detect(d, as_of="2018-05-20")
    assert any(a["entity_type"] == "state" and a["entity_id"] == "SP" and a["metric"] == "late_rate" for a in al)
    assert not any(a["entity_type"] == "state" and a["entity_id"] == "RJ" and a["metric"] == "late_rate" for a in al)
    sp = [a for a in al if a["entity_id"] == "SP" and a["metric"] == "late_rate" and a["entity_type"] == "state"][0]
    assert sp["current_value"] > sp["baseline_value"] + 0.05 and sp["p_value"] < 0.05
    quiet = [a for a in monitor.detect(synth(seed=3), as_of="2018-05-20") if a["metric"] == "late_rate"]
    assert len(quiet) <= 1                                                       # no meaningful change planted (a false alarm is rare, not zero)


def test_monitor_ignores_small_samples():
    d = synth(per_day=2, spike=("SP", 100, 0.9))                                  # far too few orders per segment
    assert not [a for a in monitor.detect(d, as_of="2018-05-20") if a["entity_type"] in ("state", "category", "seller")]


def test_run_monitor_records_job_and_dedupes(tmp_path):
    db = tmp_path / "o.db"
    d = synth(spike=("SP", 100, 0.30))
    s1 = monitor.run_monitor(db=db, d=d, as_of="2018-05-20")
    s2 = monitor.run_monitor(db=db, d=d, as_of="2018-05-20")
    assert s1["created"] >= 1 and s2["created"] == 0 and s2["updated"] == s1["found"]
    assert store.list_jobs(5, db)[0]["status"] == "ok"
    assert len(store.list_alerts(db=db)) == s1["found"]


# ------------------------------------------------------------------ root cause and playbooks
def test_rca_names_the_planted_driver(tmp_path):
    d = synth(spike=("SP", 100, 0.30))
    al = [a for a in monitor.detect(d, as_of="2018-05-20") if a["entity_type"] == "platform" and a["metric"] == "late_rate"]
    assert al, "platform late rate should rise when one large state spikes"
    an = rca.analyse(d, al[0])
    top = an["top_driver"]
    assert top["dimension"] == "state" and top["level"] == "SP"
    assert 0.5 < top["share_of_excess"] <= 1.0 and an["limitations"]
    recs = playbooks.recommend(al[0], an)
    assert recs and all(r["steps"] and r["owner"] for r in recs)


def test_rca_rejects_unknown_metric():
    with pytest.raises(ValueError):
        rca.analyse(synth(n_days=30), {"metric": "nope", "entity_type": "platform", "entity_id": "all"})


def test_propose_from_alert_needs_human_approval(tmp_path):
    ops, app_db = tmp_path / "o.db", tmp_path / "a.db"
    d = synth(spike=("SP", 100, 0.30))
    monitor.run_monitor(db=ops, d=d, as_of="2018-05-20")
    al = store.list_alerts(db=ops)[0]
    an = rca.analyse(d, al)
    aid = playbooks.propose_from_alert(al, an, 0, db=app_db, ops_db=ops)
    act = astore.get_action(aid, db=app_db)
    assert act["status"] == "proposed" and act["evidence_ids"]                    # never auto-approved, always backed by evidence
    assert store.get_alert(al["alert_id"], ops)["action_id"] == aid
    with pytest.raises(ValueError):
        playbooks.propose_from_alert(al, an, 99, db=app_db, ops_db=ops)


# ------------------------------------------------------------------ outcomes
def test_outcome_detects_a_real_improvement_and_no_change():
    d = synth(spike=("SP", 0, 0.30), effect_after=("SP", 90, 0.06), per_day=60)
    r = outcomes.measure(d, "late_rate", "state", "SP", "2018-04-01", 28)          # day 90 = 2018-04-01
    assert r["verdict"] == "improved" and r["ci95"][1] < 0 and r["method"] == "difference_in_differences"
    rising = outcomes.measure(synth(per_day=60, spike=("RJ", 0, 0.06), effect_after=("SP", 90, 0.06)), "late_rate", "state", "SP", "2018-04-01", 28)
    assert rising["verdict"] in ("improved", "worsened", "inconclusive") and "caution" in rising
    flat = outcomes.measure(synth(per_day=60), "late_rate", "state", "SP", "2018-04-01", 28)
    assert flat["verdict"] in ("inconclusive",)


def test_outcome_needs_enough_data_and_valid_inputs():
    d = synth(n_days=60)
    assert outcomes.measure(d, "late_rate", "state", "SP", "2018-02-28", 28)["verdict"] == "insufficient_data"   # after-window mostly beyond the data
    assert outcomes.measure(d, "late_rate", "state", "NOPE", "2018-02-01", 14)["verdict"] == "insufficient_data"
    with pytest.raises(ValueError):
        outcomes.measure(d, "late_rate", "state", "SP", "2018-02-01", 3)
    with pytest.raises(ValueError):
        outcomes.measure(d, "bogus", "state", "SP", "2018-02-01", 28)


# ------------------------------------------------------------------ routes (real Olist data)
@pytest.fixture()
def client():
    return create_app(TestConfig).test_client()


def test_ops_pages_and_api_flow(client):
    assert client.get("/ops").status_code == 200 and client.get("/ops/outcomes").status_code == 200 and client.get("/admin/keys").status_code == 200
    assert client.post("/api/ops/run", json={"as_of": "not-a-date"}).status_code == 400
    r = client.post("/api/ops/run", json={})
    assert r.status_code == 200 and r.get_json()["found"] >= 0
    al = client.get("/api/ops/alerts").get_json()["alerts"]
    assert al, "the real Olist data produces alerts for the latest window"
    drift_al = [a for a in al if a["kind"] == "data_drift"]
    rate_al = [a for a in al if a["kind"] == "rate_increase"]
    assert rate_al and drift_al, "real data gives rate alerts and a drift alert for the latest window"
    did = drift_al[0]["alert_id"]
    assert client.get(f"/ops/alert/{did}").status_code == 200
    dj = client.get(f"/api/ops/alerts/{did}").get_json()
    assert dj["analysis"] is None and dj["recommendations"][0]["playbook"] == "data_drift/retrain_check"
    assert client.post(f"/api/ops/alerts/{did}/propose", json={}).status_code == 201     # drift alerts can also become a reviewed action
    aid = rate_al[0]["alert_id"]
    assert client.get(f"/ops/alert/{aid}").status_code == 200
    assert client.get("/ops/alert/ALT-99999").status_code == 404
    j = client.get(f"/api/ops/alerts/{aid}").get_json()
    assert j["analysis"]["limitations"] and j["recommendations"]
    assert client.post(f"/api/ops/alerts/{aid}/propose", json={"playbook_index": 0}).status_code == 201
    assert client.post(f"/api/ops/alerts/{aid}/propose", json={}).status_code == 409
    assert client.post(f"/api/ops/alerts/{aid}/state", json={"state": "resolved"}).status_code == 400
    assert client.post(f"/api/ops/alerts/{aid}/state", json={"state": "acknowledged"}).get_json()["state"] == "acknowledged"
    assert client.get("/api/ops/alerts?severity=bad").status_code == 400
    ro = client.post("/api/ops/outcomes", json={"metric": "late_rate", "entity_type": "state", "entity_id": "SP", "action_date": "2018-03-01"})
    assert ro.status_code == 201 and ro.get_json()["result"]["verdict"] in ("improved", "worsened", "inconclusive")
    assert client.post("/api/ops/outcomes", json={"metric": "late_rate", "entity_type": "bad", "entity_id": "x", "action_date": "2018-03-01"}).status_code == 400
    assert client.post("/api/ops/outcomes", json={"metric": "late_rate", "entity_type": "state", "entity_id": "SP", "action_date": "01/03/2018"}).status_code == 400
    assert client.get("/api/ops/outcomes").get_json()["outcomes"]


@pytest.fixture()
def auth_app(tmp_path):
    class C(AuthTestConfig):
        AUTH_DB = str(tmp_path / "auth.db")
        OPS_DB = str(tmp_path / "ops.db")
        APP_DB = str(tmp_path / "app.db")
        RATE_LIMIT_PER_MINUTE = 1000
        API_RATE_LIMIT_PER_MINUTE = 5
    sec.reset_state()
    a = create_app(C)
    with a.app_context():
        adb.conn().__enter__()
        adb.create_user("admin@example.com", "Admin", sec.hash_password(PW), "admin")
        adb.create_user("user@example.com", "User", sec.hash_password(PW), "user")
    return a


def _login(app, email):
    c = app.test_client()
    c.get("/login")
    with c.session_transaction() as s:
        t = s["csrf"]
    assert c.post("/login", data={"email": email, "password": PW, "csrf_token": t}).status_code in (200, 302)
    with c.session_transaction() as s:
        return c, {"X-CSRF-Token": s["csrf"]}


def test_ops_needs_login_and_keys_are_admin_only(auth_app):
    anon = auth_app.test_client()
    assert anon.get("/api/ops/alerts").status_code == 401
    assert anon.get("/ops", follow_redirects=False).status_code in (301, 302, 401)
    u, uh = _login(auth_app, "user@example.com")
    assert u.get("/api/ops/alerts").status_code == 200
    assert u.get("/api/admin/keys").status_code == 403 and u.post("/api/admin/keys", json={"name": "x"}, headers=uh).status_code == 403
    assert u.get("/admin/keys").status_code == 403
    a, ah = _login(auth_app, "admin@example.com")
    assert a.post("/api/admin/keys", json={"name": "x"}).status_code == 400               # CSRF still enforced on browser calls


def test_api_v1_key_flow_scopes_and_rate_limit(auth_app):
    a, ah = _login(auth_app, "admin@example.com")
    r = a.post("/api/admin/keys", json={"name": "bi", "scope": "read"}, headers=ah)
    assert r.status_code == 201
    rk = r.get_json()["key"]; rid = r.get_json()["key_id"]
    wk = a.post("/api/admin/keys", json={"name": "bot", "scope": "write"}, headers=ah).get_json()["key"]
    assert a.post("/api/admin/keys", json={"name": "x", "scope": "root"}, headers=ah).status_code == 400
    assert a.post("/api/ops/run", json={}, headers=ah).status_code == 200
    anon = auth_app.test_client()
    assert anon.get("/api/v1/alerts").status_code == 401                                   # no key
    assert anon.get("/api/v1/alerts", headers={"Authorization": "Bearer ecx_wrong"}).status_code == 401
    assert anon.get("/api/v1/alerts", headers={"Authorization": "Basic abc"}).status_code == 401
    H = lambda k: {"Authorization": f"Bearer {k}"}
    got = anon.get("/api/v1/alerts", headers=H(rk))                                        # a bearer key works without a login or CSRF token
    assert got.status_code == 200 and got.get_json()["alerts"]
    aid = got.get_json()["alerts"][0]["alert_id"]
    assert anon.post(f"/api/v1/alerts/{aid}/acknowledge", headers=H(rk), json={}).status_code == 403      # read-only key cannot write
    ack = anon.post(f"/api/v1/alerts/{aid}/acknowledge", headers=H(wk), json={"note": "seen"})
    assert ack.status_code == 200 and ack.get_json()["state"] == "acknowledged" and ack.get_json()["ack_by"].startswith("api:KEY-")
    assert anon.get("/api/v1/actions", headers=H(rk)).status_code == 200 and anon.get("/api/v1/outcomes", headers=H(rk)).status_code == 200
    assert anon.get("/api/v1/actions?status=zzz", headers=H(rk)).status_code == 400
    assert anon.get("/api/v1/alerts/ALT-99999", headers=H(rk)).status_code == 404
    codes = [anon.get("/api/v1/health", headers=H(rk)).status_code for _ in range(8)]
    assert 429 in codes                                                                    # per-key rate limit (5 per minute in this test)
    sec.reset_state()
    assert a.post(f"/api/admin/keys/{rid}/revoke", headers=ah).status_code == 200
    assert anon.get("/api/v1/health", headers=H(rk)).status_code == 401                    # revoked keys stop working at once
    assert a.post("/api/admin/keys/KEY-99999/revoke", headers=ah).status_code == 404


# ------------------------------------------------------------------ digest
def test_digest_files_text_and_email_rules(tmp_path, monkeypatch):
    from ecx3 import digest
    d = synth(spike=("SP", 100, 0.30))
    ops, app_db = tmp_path / "o.db", tmp_path / "a.db"
    monitor.run_monitor(db=ops, d=d, as_of="2018-05-20")
    g = digest.build(d, ops, app_db, as_of="2018-05-20")
    assert g["as_of"] == "2018-05-20" and g["kpis"] and g["counts"]["open"] >= 1
    paths = digest.write(g, tmp_path / "out")
    assert all(__import__("pathlib").Path(p).exists() for p in paths.values())
    page = open(paths["html"], encoding="utf-8").read()
    assert "ECX-Intelligence digest" in page and "<script" not in page
    g["open_alerts"][0]["title"] = "<script>alert(1)</script>"                       # text from the data is escaped in the HTML
    assert "<script>alert(1)" not in digest.to_html(g)
    for v in digest.SMTP_VARS:
        monkeypatch.delenv(v, raising=False)
    assert not digest.email_configured()
    with pytest.raises(RuntimeError):
        digest.send_email(g)                                                          # e-mail stays off without configuration


def test_digest_route_and_cli(client, tmp_path):
    r = client.post("/api/ops/digest", json={})
    assert r.status_code == 200 and r.get_json()["text"].startswith("ECX-Intelligence digest")
    assert client.post("/api/ops/digest", json={"as_of": "x"}).status_code == 400
    from ecx3 import digest
    digest.main(["--as-of", "2018-06-30", "--out", str(tmp_path / "cli")])
    assert (tmp_path / "cli" / "digest_2018-06-30.html").exists()


# ------------------------------------------------------------------ drift and readiness
def test_drift_psi_flags_a_shift_and_not_a_stable_feature():
    from ecx3 import drift
    rng = np.random.default_rng(1)
    base = rng.normal(100, 20, 3000)
    assert drift.psi(base, rng.normal(100, 20, 3000)) < 0.1                                # same distribution: stable
    assert drift.psi(base, rng.normal(160, 20, 3000)) > 0.25                               # strongly shifted: large
    assert drift.psi(base[:50], base) is None                                              # too little data: no verdict
    d = synth(n_days=140, per_day=40)
    d.loc[pd.to_datetime(d.order_purchase_timestamp) >= "2018-04-23", "distance_km"] += 3000    # shift only the latest window
    al = drift.drift_alerts(d, as_of="2018-05-20")
    assert any(a["entity_id"] == "distance_km" and a["severity"] == "warning" for a in al)
    assert not any(a["entity_id"] == "items_price" for a in al)


def test_readyz_reports_checks(client):
    r = client.get("/readyz")
    j = r.get_json()
    assert r.status_code in (200, 503) and set(j["checks"]) >= {"auth_db", "ops_db", "app_db", "orders_data", "models"}
    assert all(isinstance(v, bool) for v in j["checks"].values())
    assert "/" not in str(j)                                                               # no file paths leak


def test_readyz_is_public_when_login_is_required(auth_app):
    assert auth_app.test_client().get("/readyz").status_code in (200, 503)


def test_alert_page_does_not_claim_shares_add_up(tmp_path):
    """Driver shares come from different dimensions and overlap, so the page must not show a combined share above 100%."""
    from ecx3 import rca
    import inspect
    assert "top_share" in inspect.getsource(rca)
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "app" / "templates" / "ops_alert.html").read_text(encoding="utf-8")
    assert "Top three drivers hold" not in html and "do not add up" in html


# ------------------------------------------------------------------ alert notes
def test_notes_are_validated_ordered_and_append_only(tmp_path):
    db = tmp_path / "o.db"
    a = {"dedupe_key": "k", "kind": "rate_increase", "metric": "late_rate", "entity_type": "state", "entity_id": "SP", "severity": "warning", "title": "t"}
    aid, _ = store.upsert_alert(a, db)
    with pytest.raises(ValueError):
        store.add_note(aid, "Asha", "   ", db)                       # empty note
    with pytest.raises(ValueError):
        store.add_note(aid, " ", "text", db)                          # a named person is required
    with pytest.raises(ValueError):
        store.add_note(aid, "Asha", "x" * (store.NOTE_MAX + 1), db)   # too long
    with pytest.raises(KeyError):
        store.add_note("ALT-99999", "Asha", "hi", db)
    n1 = store.add_note(aid, "Asha", "  Called the carrier.  ", db)
    store.add_note(aid, "Ravi", "Carrier confirmed a backlog.", db)
    assert n1["body"] == "Called the carrier." and n1["author"] == "Asha"
    notes = store.list_notes(aid, db)
    assert [n["author"] for n in notes] == ["Asha", "Ravi"] and store.note_counts(db) == {aid: 2}
    assert not hasattr(store, "delete_note") and not hasattr(store, "update_note")   # notes form an audit trail


def test_notes_api_and_page_escape_html(client):
    client.post("/api/ops/run", json={})
    al = client.get("/api/ops/alerts").get_json()["alerts"]
    aid = al[0]["alert_id"]
    assert client.get(f"/api/ops/alerts/{aid}/notes").get_json()["notes"] == []
    assert client.post(f"/api/ops/alerts/{aid}/notes", json={"body": ""}).status_code == 400
    assert client.post("/api/ops/alerts/ALT-99999/notes", json={"body": "x"}).status_code == 404
    assert client.get("/api/ops/alerts/ALT-99999/notes").status_code == 404
    r = client.post(f"/api/ops/alerts/{aid}/notes", json={"body": "<script>alert(1)</script> checked"})
    assert r.status_code == 201 and r.get_json()["author"]
    page = client.get(f"/ops/alert/{aid}").get_data(as_text=True)
    assert "&lt;script&gt;alert(1)&lt;/script&gt; checked" in page and "<script>alert(1)</script>" not in page
    assert "(1 note)" in client.get("/ops").get_data(as_text=True)


def test_notes_need_login_and_csrf(auth_app):
    anon = auth_app.test_client()
    assert anon.post("/api/ops/alerts/ALT-00001/notes", json={"body": "x"}).status_code in (400, 401, 403)   # rejected before any note is stored
    c, h = _login(auth_app, "user@example.com")
    assert c.post("/api/ops/alerts/ALT-00001/notes", json={"body": "x"}).status_code in (400, 403)   # CSRF enforced
    assert c.post("/api/ops/alerts/ALT-00001/notes", json={"body": "x"}, headers=h).status_code == 404   # signed in, alert absent
