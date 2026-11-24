"""Digest: a short, self-contained report of what needs attention (HTML + JSON + plain text files), plus an optional e-mail.

Run from the command line (also what you schedule with Windows Task Scheduler or cron):
    python -m ecx3.digest --run-monitor            # run the monitor, then write today's digest
    python -m ecx3.digest --as-of 2018-06-30       # replay the digest as it would have looked on a past date
E-mail is OFF unless ALL of these environment variables are set (nothing is ever stored in code or in the database):
    ECX_SMTP_HOST, ECX_SMTP_PORT, ECX_SMTP_USER, ECX_SMTP_PASSWORD, ECX_SMTP_FROM, ECX_DIGEST_TO   (add --email to send)"""
import argparse
import html
import json
import os
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from ecx2 import store as astore
from ecx3 import monitor, store

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = Path(os.environ.get("ECX_DIGEST_DIR", ROOT / "reports" / "digests"))
SMTP_VARS = ["ECX_SMTP_HOST", "ECX_SMTP_PORT", "ECX_SMTP_USER", "ECX_SMTP_PASSWORD", "ECX_SMTP_FROM", "ECX_DIGEST_TO"]


def build(d, ops_db=None, app_db=None, as_of=None, max_alerts=10):
    """Collect the digest content from the data and the stores."""
    cur, base, (cs, end, bs) = monitor.windows(d, as_of=as_of)
    asof = (__import__("pandas").Timestamp(end) - __import__("pandas").Timedelta(days=1)).date().isoformat()
    kpis = []
    for key, m in monitor.METRICS.items():
        if len(cur) and len(base):
            kpis.append({"metric": key, "label": m["label"], "current": float(cur[m["col"]].mean()), "baseline": float(base[m["col"]].mean()),
                         "n_current": int(len(cur)), "n_baseline": int(len(base))})
    alerts = [a for a in store.list_alerts(state="open", db=ops_db)][:max_alerts]
    acts = astore.list_actions(db=app_db)
    waiting = [a for a in acts if a["status"] == "proposed"]
    running = [a for a in acts if a["status"] in ("approved", "in_progress")]
    outs = store.list_outcomes(limit=5, db=ops_db)
    return {"as_of": asof, "generated_at": datetime.now().isoformat(timespec="seconds"), "window": f"{cs} to {asof} vs {bs} to {cs}", "kpis": kpis,
            "counts": store.alert_counts(ops_db), "open_alerts": alerts,
            "actions_waiting_for_approval": [{"id": a["action_id"], "title": a["title"], "priority": a["priority"]} for a in waiting[:10]],
            "actions_in_progress": [{"id": a["action_id"], "title": a["title"], "status": a["status"]} for a in running[:10]],
            "recent_outcomes": [{"id": o["outcome_id"], "metric": o["metric"], "entity": f"{o['entity_type']} {o['entity_id']}", "verdict": o["verdict"]} for o in outs],
            "note": "Signals to investigate, not proof of causes. Verify before acting."}


def _pct(x):
    return f"{x * 100:.1f}%"


def to_text(g):
    L = [f"ECX-Intelligence digest - data up to {g['as_of']}", f"Windows: {g['window']}", ""]
    for k in g["kpis"]:
        L.append(f"- {k['label']}: {_pct(k['current'])} (previous period {_pct(k['baseline'])})")
    L += ["", f"Open alerts: {g['counts']['open']} (critical {g['counts']['open_critical']}, warning {g['counts']['open_warning']})"]
    for a in g["open_alerts"]:
        L.append(f"  [{a['severity']}] {a['alert_id']}: {a['title']}")
    L += ["", f"Actions waiting for approval: {len(g['actions_waiting_for_approval'])}"]
    L += [f"  {a['id']} ({a['priority']}): {a['title']}" for a in g["actions_waiting_for_approval"]]
    L += ["", f"Actions in progress: {len(g['actions_in_progress'])}"] + [f"  {a['id']} ({a['status']}): {a['title']}" for a in g["actions_in_progress"]]
    L += ["", "Recent outcome checks:"] + [f"  {o['id']}: {o['entity']} {o['metric']} -> {o['verdict'].replace('_', ' ')}" for o in g["recent_outcomes"]]
    L += ["", g["note"]]
    return "\n".join(L)


def to_html(g):
    e = html.escape
    rows = "".join(f"<tr><td>{e(a['severity'])}</td><td>{e(a['alert_id'])}</td><td>{e(a['title'])}</td></tr>" for a in g["open_alerts"]) or "<tr><td colspan=3>No open alerts.</td></tr>"
    k = "".join(f"<li>{e(x['label'])}: <b>{_pct(x['current'])}</b> (previous period {_pct(x['baseline'])})</li>" for x in g["kpis"])
    w = "".join(f"<li>{e(a['id'])} ({e(a['priority'])}): {e(a['title'])}</li>" for a in g["actions_waiting_for_approval"]) or "<li>None</li>"
    p = "".join(f"<li>{e(a['id'])} ({e(a['status'])}): {e(a['title'])}</li>" for a in g["actions_in_progress"]) or "<li>None</li>"
    o = "".join(f"<li>{e(x['id'])}: {e(x['entity'])} {e(x['metric'])} &rarr; {e(x['verdict'].replace('_', ' '))}</li>" for x in g["recent_outcomes"]) or "<li>None</li>"
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>ECX-Intelligence digest {e(g['as_of'])}</title>"
            "<style>body{font-family:Arial,sans-serif;max-width:760px;margin:24px auto;color:#1b2140}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:6px;text-align:left}"
            "h1{font-size:1.4rem}h2{font-size:1.05rem;margin-top:1.4rem}.n{color:#555;font-size:.85rem}</style></head><body>"
            f"<h1>ECX-Intelligence digest</h1><p class='n'>Data up to {e(g['as_of'])}. Windows: {e(g['window'])}.</p>"
            f"<h2>Key rates</h2><ul>{k}</ul><h2>Open alerts ({g['counts']['open']}: {g['counts']['open_critical']} critical, {g['counts']['open_warning']} warning)</h2>"
            f"<table><tr><th>Severity</th><th>ID</th><th>Alert</th></tr>{rows}</table>"
            f"<h2>Actions waiting for approval</h2><ul>{w}</ul><h2>Actions in progress</h2><ul>{p}</ul><h2>Recent outcome checks</h2><ul>{o}</ul>"
            f"<p class='n'>{e(g['note'])}</p></body></html>")


def write(g, out_dir=None):
    out = Path(out_dir or DEFAULT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"digest_{g['as_of']}"
    paths = {"html": out / f"{stem}.html", "json": out / f"{stem}.json", "txt": out / f"{stem}.txt"}
    paths["html"].write_text(to_html(g), encoding="utf-8")
    paths["json"].write_text(json.dumps(g, indent=2, default=str), encoding="utf-8")
    paths["txt"].write_text(to_text(g), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def email_configured(env=None):
    env = os.environ if env is None else env
    return all(env.get(v) for v in SMTP_VARS)


def send_email(g, env=None):
    """Send the digest by e-mail. Raises RuntimeError if e-mail is not configured. Credentials come only from environment variables."""
    env = os.environ if env is None else env
    if not email_configured(env):
        raise RuntimeError("E-mail is not configured. Set: " + ", ".join(SMTP_VARS))
    msg = EmailMessage()
    msg["Subject"] = f"ECX-Intelligence digest {g['as_of']}: {g['counts']['open']} open alert(s)"
    msg["From"], msg["To"] = env["ECX_SMTP_FROM"], env["ECX_DIGEST_TO"]
    msg.set_content(to_text(g)); msg.add_alternative(to_html(g), subtype="html")
    with smtplib.SMTP(env["ECX_SMTP_HOST"], int(env["ECX_SMTP_PORT"]), timeout=30) as s:
        s.starttls(context=ssl.create_default_context())
        s.login(env["ECX_SMTP_USER"], env["ECX_SMTP_PASSWORD"])
        s.send_message(msg)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write the ECX-Intelligence digest.")
    ap.add_argument("--run-monitor", action="store_true", help="run the monitor first")
    ap.add_argument("--as-of", help="replay as at a date (YYYY-MM-DD)")
    ap.add_argument("--email", action="store_true", help="also e-mail it (needs the SMTP environment variables)")
    ap.add_argument("--out", help="output folder")
    a = ap.parse_args(argv)
    from ecx2.common import orders
    d = orders()
    if a.run_monitor:
        print("monitor:", monitor.run_monitor(d=d, as_of=a.as_of))
    g = build(d, as_of=a.as_of)
    paths = write(g, a.out)
    print("digest written:", paths["html"])
    if a.email:
        send_email(g)
        print("digest e-mailed")


if __name__ == "__main__":
    main()
