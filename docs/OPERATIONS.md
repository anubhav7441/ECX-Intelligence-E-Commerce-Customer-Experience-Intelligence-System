# ECX-Intelligence Operations Layer

The original system analyses data and explains it. The operations layer closes the loop:

**Detect -> Diagnose -> Recommend -> Act -> Measure**

| Step | What it does | Where |
|---|---|---|
| Detect | Compares the last 28 days with the 84 days before for the platform and each state, category and seller. Raises an alert only when the rise is statistically meaningful (Fisher exact test, Benjamini-Hochberg correction) and large enough to matter. Also checks input data drift (PSI). | `ecx3/monitor.py`, `ecx3/drift.py` |
| Diagnose | Splits the extra problem events by region, category, seller, distance, price, weight, freight share and seller handling time, and checks which delivery stage got slower. | `ecx3/rca.py` |
| Recommend | Maps the finding to a written playbook (steps, owner, effort). Impact is stated only as a real count of linked events, never as a promised percentage. | `ecx3/playbooks.py` |
| Act | Turns a playbook into a *proposed* action with evidence in the existing Action Center. A named person must approve it. | `ecx2/store.py` (unchanged) |
| Measure | Before/after comparison against all other orders (difference-in-differences) with a 95% interval and a verdict of improved, worsened, inconclusive or insufficient data. | `ecx3/outcomes.py` |

Supporting pieces: API keys and the `/api/v1` API (`ecx3/store.py`, `app/routes/ops.py`), the digest report (`ecx3/digest.py`), the readiness check `/readyz`, Docker files and the CI workflow.

## Pages
- `/ops` Operations Center: alerts, run the monitor, create a digest.
- `/ops/alert/<id>`: root cause, playbooks, propose an action, acknowledge or resolve.
- `/ops/outcomes`: measure whether an action worked (with chart).
- `/admin/keys` (admin only): create and revoke API keys.

## Run the monitor and digest from the command line
```
python -m ecx3.digest --run-monitor            # monitor, then write today's digest to reports/digests/
python -m ecx3.digest --as-of 2018-06-30       # replay as it would have looked on a past date
```
Schedule it with Windows Task Scheduler (daily or weekly) running the first command in the project folder.
E-mail is OFF unless every SMTP variable in `.env.example` is set in your private environment, then add `--email`.

## API (for other systems)
Create a key at `/admin/keys` (shown once, stored only as a hash). Then:
```
curl -H "Authorization: Bearer <key>" http://127.0.0.1:5000/api/v1/alerts?state=open
```
Endpoints: `GET /api/v1/health`, `/alerts`, `/alerts/<id>`, `/actions`, `/outcomes`; `POST /api/v1/alerts/<id>/acknowledge` (needs a write key).
Keys are rate limited (`ECX_API_RATE_LIMIT`, default 120 per minute) and can be revoked.

## Running behind a real web server
```
pip install -r requirements-prod.txt
waitress-serve --listen=127.0.0.1:8000 wsgi:app          # Windows
gunicorn --bind 0.0.0.0:8000 --workers 2 wsgi:app        # Linux / Docker
docker compose up --build                                  # needs a .env with SECRET_KEY, see docker-compose.yml
```
`/healthz` says the process is up. `/readyz` says the databases, data and models are available.

## Honest limits
- Alerts and root causes are statistical signals and associations, not proven causes.
- The Olist data ends in August 2018. The monitor uses the newest order date as "today", so it works as a replay. Recent windows only contain orders already delivered and reviewed, which can create artificial shifts.
- Outcome measurement is not a randomised experiment. It assumes the entity and the control would have moved in parallel.
- Drift checking looks at inputs only. Checking prediction quality needs the real outcomes, which arrive late.
- SQLite and a small server suit a prototype or a small team. A large deployment would move to PostgreSQL, add HTTPS in front, and have its security reviewed.
- The Docker files were written but not built or run in the environment where they were produced. Run `docker compose up --build` once and fix anything your machine reports.

## Quality gates (run before you submit or deploy)
| Check | Command | What it proves |
|---|---|---|
| Lint | `ruff check .` | no unused imports, undefined names or common bug patterns (rules in `pyproject.toml`) |
| Security scan | `bandit -q -r app ecx2 ecx3 multids recsys -ll` | no medium or high findings in our own code |
| Package scan | `pip-audit -r requirements.txt` | no known vulnerabilities in the pinned packages (checked on the date of the run) |
| Tests and coverage | `python -m pytest -q --cov --cov-report=term-missing` | all tests pass; coverage shows which lines no test touches |

On Windows run all four with `.\scripts\check.ps1`. The same checks run on every push through `.github/workflows/ci.yml` (it also builds the Docker image).

## Monitoring a running server
- Every response has an `X-Request-ID` header. Send your own (8-64 letters, digits, `.`, `_`, `-`) to follow one request through the logs.
- `logs/ecx.log` has one access line per request: method, route pattern, status, time in ms, request id and user id. Query strings and form data are never logged. Set `ECX_LOG_JSON=1` for JSON lines.
- `/metrics` (Prometheus text) is OFF by default. Set `ECX_METRICS_TOKEN` to a long random value to enable it, then call it with `Authorization: Bearer <token>`. Counters are per worker process and reset on restart.
- `/healthz` = process is up. `/readyz` = databases, data and models are available.

## Automatic monitoring (scheduler)

No Windows Task Scheduler or cron is needed. The built-in scheduler runs the monitor, writes the digest files and, only if all the SMTP variables are set, e-mails the digest.

- One cycle now: `python -m ecx3.scheduler --once`
- Keep running (one cycle every `ECX_MONITOR_INTERVAL_HOURS`, default 24): `python -m ecx3.scheduler`
- In Docker, alongside the website: `docker compose --profile scheduler up --build`

### Optional: Windows Task Scheduler
If you prefer the operating system to start each run, use the helper script from the project folder:

- Create a daily job at 09:00: `.\scripts\schedule_windows.ps1` (or `-Time 18:30`)
- Run it now to test: `schtasks /Run /TN "ECX Monitor"`; output goes to `logs\scheduler.log`
- Remove it: `.\scripts\schedule_windows.ps1 -Remove`

The job calls `python -m ecx3.scheduler --once`. The computer must be on (and you signed in) at the scheduled time.

A failed cycle is logged with its error and the next cycle still runs. On the bundled Olist data the monitor replays the newest data it has (August 2018), so repeated runs give the same alerts.


## Notes on alerts
Each alert page has a Notes box. Anyone signed in can add a comment (up to 1000 characters); the name comes from the login. Notes are kept in `database/ecx_ops.db` (table `alert_notes`), are shown oldest first, and cannot be edited or deleted, so they work as an audit trail. The alert list shows how many notes each alert has. API: `GET` and `POST /api/ops/alerts/<id>/notes`.
