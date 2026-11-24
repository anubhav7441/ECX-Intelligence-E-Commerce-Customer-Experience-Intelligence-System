"""Multi-Dataset Analysis routes (additive blueprint). Each dataset is isolated by its own dataset id."""
import re
import threading
from flask import Blueprint, current_app, g, jsonify, render_template, request, send_from_directory
from app.auth import db as adb
from multids import capabilities, engine, ingest, registry as R, report
from multids.profile import LoadError

md = Blueprint("md", __name__)
UPLOAD_LIMIT = 90 * 1024 * 1024
SAFE_FILE = re.compile(r"^[\w.\-]+\.(csv|png|pdf)$")


def err(msg, code=400):
    return jsonify(error=msg if isinstance(msg, str) else "; ".join(msg)), code


def _check(did):
    if not R.valid_id(did):
        return err("Invalid dataset id.", 404)
    try:
        m = R.meta(did)
    except (FileNotFoundError, ValueError):
        m = None
    if not m:
        return err("Dataset not found.", 404)
    if did == R.OLIST_ID and not m.get("available"):
        return err("Olist processed data is not installed (extract the data ZIP into data/processed).", 404)
    if current_app.config["AUTH_REQUIRED"] and did != R.OLIST_ID:         # ownership: users only see their own datasets (admins see all)
        u = g.get("user")
        if not u or (u["role"] != "admin" and m.get("owner_id") != u["id"]):
            return err("Dataset not found.", 404)                        # 404, not 403, so ids of other users' datasets are not revealed
    return None


_RUNNING = set()
_LOCK = threading.Lock()


def _uid():
    u = g.get("user")
    return u["id"] if u else 0


@md.route("/multidata")
def page():
    return render_template("multidata.html", active="multidata")


@md.get("/api/md/datasets")
def datasets():
    rows = R.list_all()
    if current_app.config["AUTH_REQUIRED"]:
        u = g.get("user")
        rows = [d for d in rows if d["dataset_id"] == R.OLIST_ID or (u and (u["role"] == "admin" or d.get("owner_id") == u["id"]))]
    return jsonify(datasets=rows)


@md.post("/api/md/upload")
def upload():
    request.max_content_length = UPLOAD_LIMIT          # per-route limit; v1 upload keeps its own 20 MB limit
    f = request.files.get("file")
    if f is None or not f.filename:
        return err("Attach a file in the 'file' field (.csv, .xlsx or .jsonl).")
    try:
        r = ingest.ingest(f.filename, f.read(), request.form.get("name"), request.form.get("adapter", "auto"),
                          request.form.get("category"), request.form.get("sheet") or None, request.form.get("source") or None)
    except LoadError as e:
        return err(str(e))
    except ValueError as e:
        return err(str(e))
    if current_app.config["AUTH_REQUIRED"] and g.get("user"):
        m = R.meta(r["dataset_id"]); m["owner_id"] = g.user["id"]; R.save_meta(r["dataset_id"], m)
        r["meta"] = m
    adb.audit("dataset_upload", f"{r['dataset_id']} {f.filename[:60]}", g.get("user"), request.remote_addr)
    return jsonify(r), 201


@md.get("/api/md/<did>")
def detail(did):
    bad = _check(did)
    if bad:
        return bad
    m = R.meta(did)
    try:
        prof = engine.get_profile(did)
    except FileNotFoundError as e:
        return err(str(e), 404)
    sug = engine.suggest_mapping(did)
    res = R.results(did) or {}
    return jsonify(meta=m, profile=prof, suggestion=sug, capabilities=engine.capability_list(did), history=R.history(did),
                   results={k: {"status": v.get("status"), "title": v.get("title"), "reason": v.get("reason"), "kpis": v.get("kpis"), "findings": v.get("findings"),
                               "charts": v.get("charts"), "notes": v.get("notes"), "assumptions": v.get("assumptions")} for k, v in res.items()},
                   report_ready=(R.report_dir(did) / "report.pdf").exists())


@md.post("/api/md/<did>/mapping")
def mapping(did):
    bad = _check(did)
    if bad:
        return bad
    if did == R.OLIST_ID:
        return err("The Olist mapping is fixed (read-only).")
    b = request.get_json(silent=True)
    if not isinstance(b, dict) or not isinstance(b.get("mapping"), dict):
        return err("Send {\"mapping\": {role: column}}.")
    r = engine.confirm_mapping(did, b["mapping"])
    return (jsonify(r), 200) if r["ok"] else (jsonify(r), 400)


@md.post("/api/md/<did>/run")
def run(did):
    bad = _check(did)
    if bad:
        return bad
    b = request.get_json(silent=True)
    ids = b.get("analyses") if isinstance(b, dict) else None
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        return err("Send {\"analyses\": [ids]}.")
    unknown = [i for i in ids if i not in capabilities.ANALYSES]
    if unknown:
        return err(f"Unknown analyses: {unknown}")
    key = (did, tuple(sorted(ids)))
    with _LOCK:                                                           # block duplicate simultaneous runs of the same job
        if key in _RUNNING:
            return err("This analysis is already running. Please wait for it to finish.", 409)
        _RUNNING.add(key)
    try:
        r = engine.run(did, ids)
    finally:
        with _LOCK:
            _RUNNING.discard(key)
    if r["ok"]:
        _log_run(did, "analysis", ids, r)
    return (jsonify(r), 200) if r["ok"] else err(r["error"])


@md.post("/api/md/<did>/report")
def make_report(did):
    bad = _check(did)
    if bad:
        return bad
    r = report.build(did)
    if r["ok"]:
        _log_run(did, "report", [], r)
    return (jsonify(r), 200) if r["ok"] else err(r["error"])


@md.get("/api/md/<did>/visuals")
def visuals_view(did):
    bad = _check(did)
    if bad:
        return bad
    from multids import visuals
    try:
        return jsonify(visuals.build(did))
    except FileNotFoundError as e:
        return err(str(e), 404)
    except Exception:
        current_app.logger.exception("visuals failed")
        return err("Could not draw the charts for this file.", 500)


@md.post("/api/md/<did>/download-report")
def download_report(did):
    """One click: run the analyses if none ran yet, build the PDF, and return its download link."""
    bad = _check(did)
    if bad:
        return bad
    m = R.meta(did) or {}
    if not m.get("mapping_confirmed"):
        return err("Open this file and confirm the column matching first, then download the report.")
    if not (R.results(did) or {}):
        ids = [c["id"] for c in engine.capability_list(did) if c.get("available")]
        if not ids:
            return err("This file does not have the columns needed for any analysis, so there is no report to make.")
        r = engine.run(did, ids)
        if not r["ok"]:
            return err(r["error"])
        _log_run(did, "analysis", ids, r)
    r = report.build(did)
    if not r["ok"]:
        return err(r["error"])
    _log_run(did, "report", [], r)
    return jsonify(ok=True, url=f"/api/md/{did}/files/report.pdf")


@md.get("/api/md/<did>/files")
def files(did):
    bad = _check(did)
    if bad:
        return bad
    d = R.report_dir(did)
    return jsonify(files=sorted(p.name for p in d.iterdir() if p.is_file() and SAFE_FILE.match(p.name)))


@md.get("/api/md/<did>/files/<name>")
def get_file(did, name):
    bad = _check(did)
    if bad:
        return bad
    if not SAFE_FILE.match(name):
        return err("Not found.", 404)
    d = R.report_dir(did)
    if not (d / name).is_file():
        return err("Not found.", 404)
    return send_from_directory(d, name, as_attachment=name.endswith((".csv", ".pdf")), download_name=f"{did}_{name}" if name.endswith((".csv", ".pdf")) else None)


@md.get("/api/md/<did>/model-check")
def model_check(did):
    bad = _check(did)
    if bad:
        return bad
    return jsonify(capabilities.model_compatibility(R.meta(did).get("mapping")))


@md.delete("/api/md/<did>")
def delete(did):
    bad = _check(did)
    if bad:
        return bad
    try:
        R.delete(did)
    except PermissionError as e:
        return err(str(e), 403)
    return jsonify(deleted=did)


def _log_run(did, kind, ids, r):
    """Store a small, real summary of what happened so the History page can link back to the saved results (no recompute)."""
    try:
        m = R.meta(did)
        res = R.results(did) or {}
        summ = {"analyses": {k: (v or {}).get("status") for k, v in res.items() if not ids or k in ids}}
        adb.record_run(_uid(), did, m.get("name", did), kind, ids, "ok", summ)
    except Exception:
        current_app.logger.exception("could not record run history")
