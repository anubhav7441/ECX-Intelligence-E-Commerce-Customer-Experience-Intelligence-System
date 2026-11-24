# Testing Report (SaaS transformation)

All results below were produced in the development sandbox (Linux, Python 3.13, headless Chromium via Playwright) on the final code. **Nothing was tested on Windows or with a real internet-facing deployment.**

## Automated tests (pytest)
- Total: **168 passed** (14 new recommendation tests in `tests/test_recsys.py` covering customer with history, limited history, new customer, popular and thin products, similar products, bought-together recomputed from raw baskets, metric hand-checks, leakage-free split, evaluation file, API and page, login protection; 119 existing v1/v2/multi-dataset tests + 35 new in `tests/test_auth_flow.py`, including advice cards, ownership of advice, advice inside the PDF report and bundled-font checks).
- The 2 existing page-list tests were updated from `/` to `/dashboard` because the dashboard moved; no other existing test was changed.
- Existing tests run with authentication OFF (`TestConfig`) so the original behaviour is checked unchanged; the new tests run with authentication ON (`AuthTestConfig`).

## The 12 required journeys
| # | Journey | How verified | Result |
|---|---|---|---|
| 1 | Guest opens landing page | pytest + browser | pass |
| 2 | Guest uses demo (500 rows, limited predict/investigate) | pytest + browser | pass |
| 3 | Guest blocked from restricted pages/APIs | pytest + browser (redirect to login, 401 for APIs) | pass |
| 4 | Register (validation, duplicates, hashing) | pytest + browser | pass |
| 5 | Login (wrong password, lockout, redirect safety, disabled user) | pytest + browser | pass |
| 6 | Upload dataset (validation, ownership) | pytest + browser | pass |
| 7 | Analysis on own dataset | pytest (synthetic Online-Retail-like sample, auto mapping) | pass |
| 8 | PDF report build and download (other user gets 404) | pytest | pass |
| 9 | History records per user; open saved results | pytest + browser (page loads) | pass |
| 10 | Logout | pytest + browser | pass |
| 11 | Admin login | pytest + browser | pass |
| 12 | Admin management (roles, disable, last-admin guard, demo config, audit, logs, datasets) | pytest + browser (all tabs) | pass |

Browser run: 17 checks, 17 passed, **0 JavaScript console errors**, no horizontal scroll at 390 px width on the landing page.

## Security checks covered by tests
Password hashed (not plain), generic login error, 5-failure lockout, CSRF required on POST, open-redirect blocked, cookie `HttpOnly` + `SameSite=Lax`, security headers, user-to-user dataset isolation (list, detail, delete, report, file download), per-user rate limit, admin-only routes blocked for users and guests, password hashes never returned by the admin API, demo limits enforced on the server (cookie clearing does not reset them), demo limits not applied to registered users.

## Existing Olist functionality
All 119 earlier tests (v1 pages/APIs, ML, NLP, agents, assistant, simulator, ledger, multi-dataset engine and PDF) still pass. Olist research analysis is not limited by the demo caps.

## Not tested / known limitations
- Windows, other browsers (only Chromium), tablets/phones beyond one 390 px check.
- Load/performance with many users or very large (>90 MB) files; the Flask development server is single-process.
- Real UCI Online Retail II and Amazon Reviews files (synthetic samples only), a real local LLM.
- Penetration testing / third-party security review; HTTPS deployment; email flows (none exist).
- Accessibility was designed for (labels, focus outlines, keyboard-usable controls, reduced-motion) but no screen-reader audit was done.
**Conclusion:** works as a local research prototype. It is **not** claimed to be production-ready.

## Redesign and plain-English advice (added later)
- New look (Inter font bundled locally, light dashboard, dark landing page, split login/register, simplified menu, 4-step My Data flow). Checked in a headless browser: all journeys pass, 0 JavaScript errors, no horizontal scroll at 390 px on landing and login pages.
- Advice engine (`app/insights/`): every card has Problem / Why / What to do, evidence numbers and a confidence label. Olist cards are computed from the real processed files; uploaded-data cards use only the columns the user mapped. Verified on the Olist data and on the two synthetic sample files. **Not verified on real Online Retail II or Amazon files.**
- The "Why" text describes what the data shows (associations); it is not proof of cause.
- Not tested: users with no experience (no usability study was run), screen readers, Windows.

## Windows model-loading fix (portable XGBoost copies)
On one Windows PC the original `models/best_*.joblib` files could not be opened (`XGBoostError: input stream corrupted`) although their checksums matched. The cause is the way XGBoost pickles are read on that machine. The original files were NOT changed or retrained. `models/portable/` holds XGBoost's own portable format of the same 4 models, and `ml/portable.py` uses it automatically when the original cannot be loaded. `tests/test_portable.py` shows the portable copies give identical predictions (max difference below 1e-9), and the model-dependent tests pass with the original files deliberately broken. Not yet confirmed on the affected Windows PC at the time of writing.

## Demo upload, exit guard and input validation (added)
`tests/test_demo_upload.py` (14 tests): own-data check keeps only the first 500 rows and says so; small files are not marked truncated; wrong type, empty, header-only and binary files are refused; 5 MB size limit; 5 checks per session; CSRF required; nothing saved to disk; pages send `Cache-Control: no-store`; sign-out makes protected pages redirect to login; the exit-guard script contains the Yes/No dialog and Back handling; login/register reject bad input. A browser run (Playwright/Chromium) also confirmed: the Back button signs the user out, the dialog appears while work is running, *No* stays and *Yes* signs out and goes home. Full suite after this change in the working tree: 192 passed.
