from ecx3 import scheduler


def test_interval_default_and_custom():
    assert scheduler.interval_seconds({}) == 86400
    assert scheduler.interval_seconds({"ECX_MONITOR_INTERVAL_HOURS": "6"}) == 21600
    assert scheduler.interval_seconds({"ECX_MONITOR_INTERVAL_HOURS": "0.5"}) == 1800


def test_interval_bad_values_fall_back():
    for bad in ("abc", "0", "-3", ""):
        assert scheduler.interval_seconds({"ECX_MONITOR_INTERVAL_HOURS": bad}) == 86400


def test_cycle_runs_monitor_writes_digest_and_skips_email_when_off(monkeypatch):
    monkeypatch.setattr("ecx2.common.orders", lambda: "DATA")
    calls = []
    out = scheduler.run_cycle(
        monitor_fn=lambda d, as_of: calls.append(("monitor", d)) or {"new": 1},
        build_fn=lambda d, as_of: calls.append(("build", d)) or {"x": 1},
        write_fn=lambda g: calls.append(("write", g)) or {"html": "a.html"},
        email_fn=lambda g: calls.append(("email", g)),
        email_ok_fn=lambda: False)
    assert [c[0] for c in calls] == ["monitor", "build", "write"]
    assert out["emailed"] is False and out["digest"]["html"] == "a.html"


def test_cycle_emails_only_when_configured(monkeypatch):
    monkeypatch.setattr("ecx2.common.orders", lambda: "DATA")
    sent = []
    out = scheduler.run_cycle(
        monitor_fn=lambda d, as_of: {}, build_fn=lambda d, as_of: {"g": 1}, write_fn=lambda g: {"html": "a"},
        email_fn=lambda g: sent.append(g), email_ok_fn=lambda: True)
    assert sent == [{"g": 1}] and out["emailed"] is True


def test_loop_survives_a_failing_cycle_and_sleeps_between():
    runs, sleeps = [], []

    def cycle():
        runs.append(1)
        if len(runs) == 1:
            raise RuntimeError("boom")

    n = scheduler.loop(every=5, cycle=cycle, sleep=sleeps.append, max_cycles=3)
    assert n == 3 and len(runs) == 3
    assert sleeps == [5, 5]   # no sleep after the last cycle
