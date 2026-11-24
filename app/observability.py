"""Request tracing, access logs and metrics.

* Every response carries an X-Request-ID header (a safe incoming value is reused, otherwise a new one is made) so one request
  can be followed through the logs.
* One access-log line per request: method, route pattern, status, milliseconds, request id, user id.
  Query strings, form data and tokens are NEVER logged.
* Set ECX_LOG_JSON=1 to write the logs as JSON lines (easy to ingest by a log tool).
* /metrics (Prometheus text format) is OFF unless ECX_METRICS_TOKEN is set; then it needs "Authorization: Bearer <token>".
  Metrics are kept in memory, per process (with several workers each one reports its own counters).
"""
import hmac
import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict

from flask import Response, abort, g, request

_RID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_lock = threading.Lock()
_count = defaultdict(int)        # (method, route, status) -> requests
_dur_sum = defaultdict(float)    # route -> seconds
_dur_n = defaultdict(int)        # route -> requests
_START = time.time()


def reset_metrics():
    with _lock:
        _count.clear(); _dur_sum.clear(); _dur_n.clear()


class JsonFormatter(logging.Formatter):
    def format(self, record):
        d = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for k in ("request_id", "method", "route", "status", "ms", "user"):
            if hasattr(record, k):
                d[k] = getattr(record, k)
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, ensure_ascii=False)


def _route_label():
    r = request.url_rule
    return r.rule if r is not None else "unmatched"


def _prom_escape(v):
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def init_observability(app):
    @app.before_request
    def _start():
        g._t0 = time.perf_counter()
        rid = request.headers.get("X-Request-ID", "")
        g.request_id = rid if _RID.match(rid) else uuid.uuid4().hex[:16]

    @app.after_request
    def _finish(resp):
        t0 = g.get("_t0")
        if t0 is None:
            return resp
        ms = (time.perf_counter() - t0) * 1000
        route = _route_label()
        resp.headers["X-Request-ID"] = g.get("request_id", "")
        if not request.path.startswith("/static/"):
            with _lock:
                _count[(request.method, route, resp.status_code)] += 1
                _dur_sum[route] += ms / 1000
                _dur_n[route] += 1
            u = g.get("user")
            app.logger.info("%s %s -> %s in %.0f ms", request.method, route, resp.status_code, ms,
                            extra={"request_id": g.get("request_id"), "method": request.method, "route": route,
                                   "status": resp.status_code, "ms": round(ms), "user": (u["id"] if u else None)})
        return resp

    @app.get("/metrics")
    def metrics():
        token = app.config.get("METRICS_TOKEN")
        if not token:
            abort(404)
        sent = request.headers.get("Authorization", "")
        if not sent.startswith("Bearer ") or not hmac.compare_digest(sent[7:].encode(), str(token).encode()):
            return Response("unauthorized\n", status=401, mimetype="text/plain")
        lines = ["# HELP ecx_requests_total HTTP requests by method, route pattern and status.", "# TYPE ecx_requests_total counter"]
        with _lock:
            for (m, r, s), n in sorted(_count.items()):
                lines.append(f'ecx_requests_total{{method="{m}",route="{_prom_escape(r)}",status="{s}"}} {n}')
            lines += ["# HELP ecx_request_seconds_sum Total time spent answering requests, by route pattern.", "# TYPE ecx_request_seconds_sum counter"]
            for r, v in sorted(_dur_sum.items()):
                lines.append(f'ecx_request_seconds_sum{{route="{_prom_escape(r)}"}} {v:.6f}')
            lines += ["# HELP ecx_request_seconds_count Requests timed, by route pattern.", "# TYPE ecx_request_seconds_count counter"]
            for r, n in sorted(_dur_n.items()):
                lines.append(f'ecx_request_seconds_count{{route="{_prom_escape(r)}"}} {n}')
        lines += ["# HELP ecx_uptime_seconds Seconds since this process started.", "# TYPE ecx_uptime_seconds gauge", f"ecx_uptime_seconds {time.time() - _START:.0f}"]
        return Response("\n".join(lines) + "\n", mimetype="text/plain; version=0.0.4")
