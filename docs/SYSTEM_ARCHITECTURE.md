# System Architecture

## Flow
`Landing (/)` -> `Register / Login` -> `My Workspace` -> `Analytics (Olist dashboard, AI modules, Dataset Center)`; a restricted **guest demo** (`/demo`) hangs off the landing page.

## Layers
| Layer | Where | Notes |
|---|---|---|
| Web (Flask app factory) | `app/__init__.py` | blueprints: `pages`, `api`, `v2`, `md`, `auth`, `demo`, `admin` |
| Access control | `app/auth/__init__.py` | one global `before_request` guard: default-deny, explicit public list, CSRF, per-user rate limit, security headers |
| Accounts | `app/auth/` + SQLite `database/ecx_auth.db` | tables `users`, `audit_log`, `settings`, `analysis_runs` |
| Guest demo | `app/demo/`, `data/demo/demo_sample.parquet` | 500-row real sample; limits read from config / admin settings |
| Advice engine | `app/insights/` | rule-based cards (Olist: from processed files; uploads: from mapped columns); API `/api/insights/*`, page `/insights`, also used in the PDF report |
| Admin | `app/admin/` | JSON API + one page, admin-only |
| Analytics v1 | `app/routes`, `app/services`, `ml/`, `data/processed` | unchanged Olist research pipeline |
| 2.0 modules | `ecx2/`, `app/routes/v2.py` | risk, voice, anomalies, agents, assistant, simulator, ledger |
| Multi-dataset | `multids/`, `app/routes/multids.py` | one isolated folder per dataset under `data/datasets/<id>/` |

## Access rules
| Who | Can open |
|---|---|
| Guest (no login) | landing, login, register, `/demo`, `/api/demo/*`, static files |
| Registered user | everything above + all analysis pages and APIs, own datasets, own history |
| Administrator | + `/admin`, `/api/admin/*`, any dataset |

Unauthenticated page requests redirect to `/login`; API requests get JSON `401`. Another user's dataset returns `404` (its existence is not revealed).

## Ownership
Uploaded datasets store `owner_id` in their `meta.json`; lists and every `/api/md/<id>/...` route check it. The Olist dataset and the evidence/action ledger are the **shared research data of the project** (not per-user); the Olist dataset is read-only.

## History
Every analysis run or report build writes a row to `analysis_runs` (who, which dataset, when, status). The History page links to the **saved** results/PDF; it never recomputes.

## Performance and safety notes
- Duplicate simultaneous runs of the same analysis on a dataset are rejected (`409`).
- Per-user rate limit on expensive endpoints (`RATE_LIMIT_PER_MINUTE`, default 40).
- Upload size limits: 20 MB (v1 upload), 90 MB (multi-dataset upload). Files are parsed in memory, then stored as Parquet.
- Logs: rotating file `logs/ecx.log` (1 MB x 3), viewable by admins.

## Known limits
Single-process Flask development server; in-memory rate-limit and login-lockout counters reset on restart and are per process; SQLite; no email verification or password reset by email; no HTTPS unless you place the app behind a proxy. These are acceptable for a local project, not for public hosting.
