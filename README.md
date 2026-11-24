# ECX-Intelligence

**E-Commerce Customer Experience Intelligence System: an explainable-AI framework for customer-satisfaction prediction using transactional data and aspect-based sentiment analysis.**
B.Tech (IT) major project, built on the public Olist Brazilian E-Commerce dataset (99,441 orders, 2016-2018).

## What it does
- **Database:** SQLite (via SQLAlchemy), 8 tables with keys and foreign keys, loaded and validated.
- **Preprocessing:** one row per order (items, payments and reviews aggregated *before* joining, so nothing is double counted); delivery, freight, distance and repeat-customer features.
- **Statistics:** hypothesis tests with effect sizes and confidence intervals (late delivery, price, freight, distance, sellers, categories, states, repeat purchase), plus an "AI hypothesis vs data" verdict table.
- **Review NLP (Portuguese):** complaint taxonomy (10 classes), aspect-based sentiment (6 aspects), severity, and a TF-IDF + Logistic Regression sentiment model; evaluated on a hand-labelled sample.
- **Machine learning:** Logistic Regression, Decision Tree, Random Forest, XGBoost; leakage-safe, time-based split; class imbalance handled; paired-bootstrap model comparison.
- **Explainable AI:** SHAP global and per-order explanations.
- **Web app (Flask):** 10 pages + REST API: Executive Dashboard, Data Management (CSV validation), Review Intelligence, Satisfaction Prediction, Explainable AI, Seller / Product / Geographical Intelligence, Research & Statistics, Recommendations.
- **168 automated tests** (17 of them need the optional full rebuild and are skipped in the shipped ZIP; see docs/TESTING_REPORT.md) and a full written report generated from the real results.

## Headline results (all generated from the data, see docs/PROJECT_REPORT.md)
| Finding | Result |
|---|---|
| Late vs on-time orders, 1-2 star reviews | 62.4% vs 9.3% (odds ratio 16.3, Cliff's delta -0.64); smaller but clear when only reviews written after delivery are used |
| Price, freight, distance | statistically significant but negligible effect sizes |
| Most common complaint in negative reviews | late / non-delivery (40.3%) |
| Best checkout-time model (test ROC-AUC) | 0.61 (XGBoost) |
| Best delivery-time model (test ROC-AUC) | 0.71 (XGBoost) |
| Which algorithm is best? | no clear winner; features matter far more than algorithm |
| Repeat purchase after a bad first order | no significant difference |

Prediction is genuinely hard; scores are modest and reported honestly. Findings are associations, not proven causes.

## Requirements
- Windows 10/11 (also works on macOS/Linux), **Python 3.12 or newer** (https://www.python.org/downloads/ , tick "Add python.exe to PATH")
- About 1 GB free disk space, internet only for the first `pip install`
- The Olist dataset (free Kaggle account): https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

## Quick start (PowerShell)
```powershell
cd path\to\ECX-Intelligence
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
python run.py                           # open http://127.0.0.1:5000
```
This ZIP already contains the trained models and pre-computed results, so the web app runs immediately. The Data Management page shows the database as "Not available" until you complete the full rebuild below (everything else works).

## ECX-Intelligence 2.0 (added on top of v1)
New sidebar section "2.0 modules": Command Center 2.0, Predictive Risk Center (early late-delivery risk), Customer Voice 2.0, Anomaly Center, AI Investigation Center (Supervisor + 5 agents), AI Research Assistant (grounded, refuses unsupported questions, works without an LLM), Business Simulation (hypothetical what-if), Action Center & Evidence Ledger (human approval required; nothing is executed automatically), Model Evaluation Center.
- Code: `ecx2/`, `app/routes/v2.py`; results `v2_results/`; models `models/v2/`; ledger DB `database/ecx_app.db` (created on first use).
- All v1 files, models and pages are unchanged. 101 tests (70 v1 + 31 v2).
- Rebuild 2.0 artifacts: `python scripts\run_v2.py` (after `python scripts\run_all.py`).
- Read `docs/V2_UPGRADE.md` for measured results and honest limitations, `docs/V2_EXPERIMENT_PROTOCOLS.md` for experiments that were not run, `docs/V2_HANDOVER.md` for the handover.
- Optional local LLM (not required, untested with a real model): set `ECX_LLM_URL` (e.g. Ollama at http://localhost:11434).

## Recommendation Intelligence Center (ECX 2.0 add-on)
Sidebar: **Product Recommendations** (`/recsys`). Six tabs: recommendations for a customer (with purchase history and a reason for every item), similar products, bought together, new-customer suggestions, "How good is it?" (real test results) and pattern insights.
- Six models built in the project: popularity, item-based and user-based collaborative filtering, implicit ALS, content-based similarity, association rules, plus a hybrid whose weights were chosen on a validation period.
- **Honest result:** the Olist data is very sparse (about 6% of customers bought 2+ different products). On the held-out test period every method scores low and none clearly beats the popularity baseline. Details and numbers: `docs/RECOMMENDATION_SYSTEM.md`.
- Rebuild (needs the raw CSVs): `python -m recsys.data`, `python -m recsys.evaluate`, `python -m recsys.make_doc`. The ready-made files are included, so the page works without a rebuild.
- API: `/api/rec/customer/<id>`, `/similar/<pid>`, `/together/<pid>`, `/new`, `/explain/<cid>/<pid>`, `/evaluation`, `/insights`, `/customers`, `/products`, `/categories`, `/status`.

## Multi-Dataset Analysis (added after 2.0)
Sidebar: **Multi-Dataset Analysis**. Upload a CSV/Excel/JSONL file (or use the Olist entry), confirm the column mapping, run the analyses your data supports and download a PDF report. See `docs/MULTIDATASET_MODULE.md`. Code: `multids/`, `app/routes/multids.py`. Sample files in `samples/` are SYNTHETIC (format demos only). Real data: UCI Online Retail II and Amazon Reviews 2023 (links in the page). 154 tests in total (including authentication, demo, ownership and admin tests).

## Professional SaaS layer (accounts, demo, workspace, admin)
The app now follows the flow **Landing page -> Register/Login -> My Workspace -> Analytics**, plus a restricted **guest demo**.
- `/` is the public landing page; the old dashboard is at `/dashboard`. **Every other page and API requires login** (default-deny guard).
- Guest demo at `/demo`: 500 real Olist orders, limited predictions (10) and investigations (3), enforced on the server. Limits apply to guests only.
- Registered users: own datasets (private), analyses, PDF reports, History, account settings. Administrators: user/role management, dataset overview, activity, audit log, system log, demo limits.
- Docs: `docs/SYSTEM_ARCHITECTURE.md`, `docs/AUTHENTICATION_GUIDE.md`, `docs/USER_GUIDE.md`, `docs/ADMIN_GUIDE.md`, `docs/TECHNOLOGY_STACK.md`, `docs/TESTING_REPORT.md`.

**Plain-English advice:** *Insights & Advice* turns the analyses into cards: **Problem -> Why it is happening -> What you can do**, with evidence and a confidence label (code: `app/insights/`). The PDF report includes the same advice. The interface uses the Sora (headings) and Manrope (text) fonts (bundled in `app/static/fonts`, no internet needed). See `docs/USER_GUIDE.md`.

### First run on Windows (PowerShell)
```powershell
cd path\to\ECX-Intelligence
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\make_demo_sample.py     # only if data\demo\demo_sample.parquet is missing
python scripts\create_admin.py         # asks for your admin email and a hidden password (nothing is stored in code)
python run.py                           # open http://127.0.0.1:5000
```
**Access control.** Sign-up only ever creates a normal user; only an existing administrator can make another administrator (first one: `python scripts\create_admin.py`). The admin page is at **/control-room** (not /admin): open that address and sign in with an administrator email and password. Anyone else is refused with one generic message, and repeated failures lock the sign-in for a while. Change the address with `ECX_ADMIN_PATH` in `.env`.
How a new account becomes active is set by `ECX_SIGNUP_MODE`: `email` = a 6-digit code is e-mailed to the person (15 minutes, 5 tries); `admin` = an administrator approves it on the admin page; `open` = immediately; `auto` (default) = `email` when the `ECX_SMTP_*` settings in `.env` are filled in, otherwise `open`. An administrator can also approve any waiting account by hand.

No password or secret key is stored in the project. The session key is generated on first run into `database/.secret_key` (or set the `SECRET_KEY` environment variable).
This is a local research prototype: it has not been load-tested or security-audited for public hosting, so it is **not** claimed to be production-ready.

## Full rebuild from the raw data (optional, ~5-10 minutes)
1. Download the Olist dataset from Kaggle and unzip the **9 CSV files into `data\raw\`**:
   `olist_customers_dataset.csv, olist_geolocation_dataset.csv, olist_order_items_dataset.csv, olist_order_payments_dataset.csv, olist_order_reviews_dataset.csv, olist_orders_dataset.csv, olist_products_dataset.csv, olist_sellers_dataset.csv, product_category_name_translation.csv`
2. Run the whole pipeline (database, preprocessing, statistics, NLP, ML, SHAP, charts, recommendations):
   ```powershell
   python scripts\run_all.py
   python scripts\make_report.py          # regenerates docs\PROJECT_REPORT.md
   python -m pytest tests -q              # 119 tests (needs the rebuild above)
   ```
   The pipeline is deterministic: a clean rebuild reproduced every result file exactly.

## Project structure
```
app/            Flask app: routes/ (pages + REST API), services/, templates/, static/ (Bootstrap + Chart.js bundled locally)
config.py, run.py
database/       schema.sql, queries/kpis.sql, ecx.db (created by the pipeline)
data/raw/       the 9 Olist CSVs (you add these)
data/processed/ analytic datasets and BI tables (created)
ml/             preprocessing/, sentiment_analysis/, prediction/, explainability/, evaluation/
research/       statistical_analysis/, run_eda_bi.py, results/
scripts/        run_all.py, 01_load_database.py, profile_raw.py, make_report.py
models/         trained models (created)
tests/          pytest suite (168 tests)
notebooks/      01_eda_walkthrough.ipynb (executed, real outputs)
recsys/         recommendation engine (data, models, evaluate, service)
ecx2/, multids/ ECX 2.0 engines (risk, agents, simulator, assistant) and the multi-dataset engine
samples/        SYNTHETIC test files for the upload module (not real data)
docs/           PROJECT_REPORT.md/.docx, RESEARCH_PAPER_DRAFT.md, PRESENTATION_OUTLINE.md, DATA_DICTIONARY.md, SRS.md,
                VIVA_PREPARATION.md, guides, figures/, screenshots/, diagrams/
```

## Screenshots
Real screenshots of the running app are in `docs/screenshots/` (recommendations, evaluation, landing, login, guest demo, overview, advice, my data, risk, simulator, assistant, history, account). Architecture and pipeline diagrams: `docs/diagrams/`.

## REST API (JSON)
`GET /api/health, /api/kpis, /api/monthly, /api/sellers?segment=&limit=, /api/categories, /api/states, /api/research, /api/recommendations, /api/options, /api/explain/<order_id>`
`POST /api/review/analyze {text}, /api/predict {items_price, freight_value, estimated_days, seller_id, customer_state, category, ...}, /api/upload (multipart file)`

## Important limitations
Observational 2016-2018 data from one marketplace (association, not causation); only reviewers are observed; many late-order reviews pre-date the recorded delivery; complaint labels were checked on 150 reviews labelled by one AI annotator (not independent human labels); sentiment labels derive from star ratings; prediction scores are modest and not calibrated probabilities. Full list in docs/PROJECT_REPORT.md section 9.

## Technology notes
Python, pandas, NumPy, SciPy, statsmodels, scikit-learn, XGBoost, SHAP, SQLAlchemy/SQLite, Flask, Bootstrap 5, Chart.js, matplotlib, pytest. (NLTK and seaborn from the original plan were not needed: Portuguese text is handled with custom rules and scikit-learn TF-IDF.) Power BI from the original brief was replaced by the Flask dashboard.


## Operations layer (alerts, root cause, playbooks, outcome tracking, API)
See `docs/OPERATIONS.md`. Quick start: open `/ops` after logging in and press **Run monitor now**, or run `python -m ecx3.digest --run-monitor`.
Tests: `python -m pytest tests/test_ops.py -q`.

## Quality gates and monitoring
`ruff check .`, `bandit -q -r app ecx2 ecx3 multids recsys -ll`, `pip-audit -r requirements.txt` and `python -m pytest -q --cov` (Windows: `.\scripts\check.ps1`). Request ids, JSON logs and the optional `/metrics` endpoint are described in `docs/OPERATIONS.md`.
