"""Request tracing, access logs and the optional /metrics endpoint."""
import json
import logging

import pytest

from app import create_app
from app.observability import JsonFormatter, reset_metrics
from config import TestConfig


class MetricsConfig(TestConfig):
    METRICS_TOKEN = "test-metrics-token-123"


@pytest.fixture()
def mclient():
    reset_metrics()
    return create_app(MetricsConfig).test_client()


def test_request_id_is_added_and_safe_incoming_id_is_reused(client):
    r = client.get("/healthz")
    assert len(r.headers["X-Request-ID"]) >= 8
    r2 = client.get("/healthz", headers={"X-Request-ID": "trace-abc-12345"})
    assert r2.headers["X-Request-ID"] == "trace-abc-12345"


def test_unsafe_incoming_request_id_is_replaced(client):
    r = client.get("/healthz", headers={"X-Request-ID": "short"})
    assert r.headers["X-Request-ID"] != "short"        # too short: replaced
    r = client.get("/healthz", headers={"X-Request-ID": "has spaces and <script>"})
    assert "<" not in r.headers["X-Request-ID"] and " " not in r.headers["X-Request-ID"]


def test_metrics_disabled_by_default(client):
    assert client.get("/metrics").status_code == 404


def test_metrics_requires_the_token(mclient):
    assert mclient.get("/metrics").status_code == 401
    assert mclient.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert mclient.get("/metrics", headers={"Authorization": "Basic test-metrics-token-123"}).status_code == 401


def test_metrics_counts_requests_by_route_pattern_not_raw_path(mclient):
    mclient.get("/healthz"); mclient.get("/healthz"); mclient.get("/does-not-exist-12345")
    r = mclient.get("/metrics", headers={"Authorization": "Bearer test-metrics-token-123"})
    assert r.status_code == 200 and r.mimetype == "text/plain"
    body = r.get_data(as_text=True)
    assert 'ecx_requests_total{method="GET",route="/healthz",status="200"} 2' in body
    assert 'route="unmatched",status="404"' in body and "does-not-exist-12345" not in body   # no unbounded labels, no raw paths
    assert "ecx_uptime_seconds" in body and "ecx_request_seconds_sum" in body


def test_access_log_has_route_and_status_but_no_query_string(client, caplog):
    with caplog.at_level(logging.INFO):
        client.get("/healthz?token=SECRET123")
    lines = [r for r in caplog.records if getattr(r, "route", None) == "/healthz"]
    assert lines and lines[-1].status == 200 and lines[-1].request_id
    assert "SECRET123" not in " ".join(r.getMessage() for r in caplog.records)


def test_json_formatter_writes_valid_json():
    rec = logging.LogRecord("n", logging.INFO, "f", 1, "hello %s", ("world",), None)
    rec.request_id = "abc12345"; rec.status = 200
    d = json.loads(JsonFormatter().format(rec))
    assert d["msg"] == "hello world" and d["request_id"] == "abc12345" and d["status"] == 200 and d["level"] == "INFO"
