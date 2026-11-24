"""Built-in scheduler: run the monitor and write the digest every N hours, without Windows Task Scheduler or cron.

    python -m ecx3.scheduler --once            # one cycle, then exit
    python -m ecx3.scheduler                   # keep running; one cycle every ECX_MONITOR_INTERVAL_HOURS (default 24)

A cycle = run the monitor, write the digest files, and e-mail the digest only if the SMTP environment variables are all set.
A failed cycle is logged and the loop carries on with the next one, so one bad run never stops the schedule."""
import argparse
import logging
import os
import time

log = logging.getLogger("ecx.scheduler")


def interval_seconds(env=None):
    """Hours between cycles from ECX_MONITOR_INTERVAL_HOURS (default 24). Bad or non-positive values fall back to 24."""
    env = os.environ if env is None else env
    try:
        h = float(env.get("ECX_MONITOR_INTERVAL_HOURS", "24"))
    except ValueError:
        h = 24.0
    if h <= 0:
        h = 24.0
    return int(h * 3600)


def run_cycle(as_of=None, monitor_fn=None, build_fn=None, write_fn=None, email_fn=None, email_ok_fn=None):
    """One monitor + digest cycle. The functions are injectable so the cycle can be tested without data."""
    from ecx2.common import orders
    from ecx3 import digest, monitor
    monitor_fn = monitor_fn or monitor.run_monitor
    build_fn = build_fn or digest.build
    write_fn = write_fn or digest.write
    email_fn = email_fn or digest.send_email
    email_ok_fn = email_ok_fn or digest.email_configured
    d = orders()
    result = monitor_fn(d=d, as_of=as_of)
    g = build_fn(d, as_of=as_of)
    paths = write_fn(g)
    emailed = False
    if email_ok_fn():
        email_fn(g)
        emailed = True
    log.info("cycle ok: digest=%s emailed=%s", paths.get("html"), emailed)
    return {"monitor": result, "digest": paths, "emailed": emailed}


def loop(every=None, cycle=None, sleep=time.sleep, max_cycles=None):
    """Run cycles forever (or max_cycles times, for tests). A failing cycle is logged and does not stop the loop."""
    every = every if every is not None else interval_seconds()
    cycle = cycle or run_cycle
    n = 0
    while max_cycles is None or n < max_cycles:
        try:
            cycle()
        except Exception:  # noqa: BLE001 - keep the schedule alive; the error is logged with its traceback
            log.exception("cycle failed; will retry in %s s", every)
        n += 1
        if max_cycles is not None and n >= max_cycles:
            break
        sleep(every)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the ECX-Intelligence monitor on a schedule.")
    ap.add_argument("--once", action="store_true", help="run one cycle and exit")
    ap.add_argument("--as-of", help="replay as at a date (YYYY-MM-DD)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.once:
        run_cycle(as_of=a.as_of)
        return
    log.info("scheduler started, interval %s s", interval_seconds())
    loop(cycle=lambda: run_cycle(as_of=a.as_of))


if __name__ == "__main__":
    main()
