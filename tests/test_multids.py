"""Tests for the Multi-Dataset module (uses SYNTHETIC fixtures only - never real findings)."""
import gzip
import io
import json
import numpy as np
import pandas as pd
import pytest

from multids import adapters, capabilities, engine, ingest, mapping as M, registry as R, report
from multids.profile import LoadError, load_table, profile


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIDS_DIR", str(tmp_path / "datasets"))


def retail_df(n=800, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"Invoice": [str(500000 + i // 2) for i in range(n)], "StockCode": rng.choice([str(21000 + i) for i in range(20)] + ["POST", "M"], n),
                      "Description": "ITEM", "Quantity": rng.integers(1, 10, n), "InvoiceDate": pd.Timestamp("2011-01-01") + pd.to_timedelta(rng.integers(0, 300 * 1440, n), unit="m"),
                      "Price": np.round(rng.uniform(1, 20, n), 2), "Customer ID": rng.integers(12000, 12100, n).astype(float), "Country": rng.choice(["UK", "FR", "DE"], n)})
    d.loc[:9, "Invoice"] = "C" + d.loc[:9, "Invoice"]; d.loc[:9, "Quantity"] *= -1
    d.loc[10:14, "Price"] = 0; d.loc[15:24, "Customer ID"] = np.nan
    d.loc[30, "Quantity"] = 99999                                   # planted extreme
    return pd.concat([d, d.iloc[:5]], ignore_index=True)           # 5 exact duplicates


def xlsx_bytes(df):
    b = io.BytesIO(); df.to_excel(b, index=False); return b.getvalue()


def amazon_rows(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        r = int(rng.choice([1, 2, 3, 4, 5], p=[.15, .1, .1, .2, .45]))
        rows.append({"rating": float(r), "title": "t", "text": ("arrived damaged and stopped working, late delivery" if r <= 2 else "works great, love it, easy to install") + f" number {i}",
                     "asin": f"A{i % 30}", "parent_asin": f"P{i % 30}", "user_id": f"U{i % 200}", "timestamp": 1546300800000 + i * 86400000 * 3, "helpful_vote": 0, "verified_purchase": True, "images": []})
    rows.append(rows[0])
    return rows


# ---------------------------------------------------------------- loading / profiling
def test_load_csv_xlsx_jsonl_and_errors():
    df = retail_df()
    a, _ = load_table("x.csv", df.to_csv(index=False).encode()); assert a.shape[1] == 8
    b, _ = load_table("x.xlsx", xlsx_bytes(df)); assert len(b) == len(df)
    raw = "\n".join(json.dumps(r) for r in amazon_rows())
    c, _ = load_table("a.jsonl", raw.encode()); assert "rating" in c.columns
    d, _ = load_table("a.jsonl.gz", gzip.compress(raw.encode())); assert len(d) == len(c)
    for name, data in [("x.exe", b"zz"), ("x.csv", b""), ("x.xlsx", b"not a zip"), ("x.jsonl", b"{bad json")]:
        with pytest.raises(LoadError):
            load_table(name, data)


def test_xlsx_with_two_matching_sheets_is_stacked():
    df = retail_df(100); b = io.BytesIO()
    with pd.ExcelWriter(b) as w:
        df.iloc[:50].to_excel(w, sheet_name="Y1", index=False); df.iloc[50:].to_excel(w, sheet_name="Y2", index=False)
    out, notes = load_table("x.xlsx", b.getvalue())
    assert len(out) == len(df) and any("stacked" in n for n in notes)


def test_profile_detects_kinds_missing_duplicates():
    df = retail_df()
    p = profile(df)
    assert "InvoiceDate" in p["by_kind"]["date"] or df["InvoiceDate"].dtype.kind == "M"
    assert p["duplicate_rows"] >= 5 and 0 < p["quality_score"] <= 100
    assert any("duplicated" in i["issue"] for i in p["issues"]) and any("negative" in i["issue"] for i in p["issues"])


# ---------------------------------------------------------------- mapping + capabilities
def test_mapping_suggestion_validation_and_conflicts():
    df = retail_df()
    s = M.suggest(df)["suggested"]
    assert s["order_id"]["column"] == "Invoice" and s["region"]["column"] == "Country" and s["unit_price"]["column"] == "Price"
    ok, errs, _ = M.validate(df, {"date": "InvoiceDate", "quantity": "Quantity"}); assert not errs and len(ok) == 2
    _, errs, _ = M.validate(df, {"date": "Country"}); assert errs                           # text is not a date
    _, errs, _ = M.validate(df, {"revenue": "Missing"}); assert errs
    _, errs, _ = M.validate(df, {"quantity": "Quantity", "revenue": "Quantity"}); assert errs  # one column, one meaning
    _, errs, _ = M.validate(df, {"nonsense": "Quantity"}); assert errs


def test_capabilities_never_run_without_required_roles():
    caps = {c["id"]: c for c in capabilities.evaluate({"region": "Country", "order_id": "Invoice"}, n_numeric_cols=2)}
    assert caps["geographic_analysis"]["available"] and caps["numeric_statistics"]["available"]
    for a in ("sales_overview", "review_analysis", "delivery_analysis", "seller_analysis", "customer_segmentation"):
        assert not caps[a]["available"] and "needs" in caps[a]["reason"]
    assert not capabilities.evaluate({}, 0)[-1]["available"]                                   # no numeric columns -> no stats
    assert capabilities.can_run("sales_overview", {"date": "d", "quantity": "q", "unit_price": "p"})["available"]   # qty x price alternative
    assert not capabilities.can_run("bogus", {})["available"]


def test_model_compatibility_never_applies_olist_model():
    r = capabilities.model_compatibility({"rating": "r"})
    assert r["compatible"] is False and r["olist_model_applied"] is False and r["reasons"]


# ---------------------------------------------------------------- adapters
def test_online_retail_cleaning_steps_are_counted():
    df = retail_df()
    clean, log = adapters.clean_online_retail(df)
    steps = {l["step"]: l["detail"] for l in log}
    assert "5 removed" in steps["removed exact duplicate rows"] or "removed" in steps["removed exact duplicate rows"]
    assert (clean.Price > 0).all() and clean.Invoice.str.startswith("C").sum() == clean.is_cancellation.sum() > 0
    assert not clean.StockCode.isin(["POST", "M"]).any()
    assert clean["Customer ID"].isna().sum() > 0 and clean["Customer ID"].dropna().map(lambda v: "." not in v).all()
    assert (clean.Revenue == clean.Quantity * clean.Price).all()
    assert adapters.detect(df) == "online_retail_ii"


def test_amazon_cleaning_and_detection():
    df = pd.DataFrame(amazon_rows())
    assert adapters.detect(df) == "amazon_reviews"
    clean, log, note = adapters.clean_amazon(df, "Appliances")
    assert "images" not in clean.columns and clean.rating.between(1, 5).all() and "Appliances" in note
    assert pd.api.types.is_datetime64_any_dtype(clean.timestamp) and len(clean) == len(df) - 1


# ---------------------------------------------------------------- end-to-end + isolation
def _ingest_retail():
    return ingest.ingest("retail.xlsx", xlsx_bytes(retail_df()))["dataset_id"]


def _confirm(did, extra=None):
    m = {k: v["column"] for k, v in engine.suggest_mapping(did)["suggested"].items()}
    m.update(extra or {})
    return engine.confirm_mapping(did, m)


def test_end_to_end_retail_values_match_independent_pandas():
    did = _ingest_retail(); c = _confirm(did)
    assert c["ok"]
    out = engine.run(did, ["sales_overview", "customer_segmentation", "numeric_statistics", "review_analysis"])
    st = {s["id"]: s["status"] for s in out["summary"]}
    assert st["sales_overview"] == "done" and st["review_analysis"] == "not_run"
    df = R.load_df(did)
    r = R.results(did)["sales_overview"]
    net = dict((k["label"], k["value"]) for k in r["kpis"])["Net revenue"]
    assert net == f"{(df.Quantity * df.Price).sum():,.2f}"                                   # independent computation
    anomalies = pd.read_csv(R.report_dir(did) / "anomalies.csv")
    assert 99999 in anomalies.Quantity.values                                                  # planted extreme found
    assert report.build(did)["ok"] and (R.report_dir(did) / "report.pdf").read_bytes()[:4] == b"%PDF"


def test_dataset_isolation_between_datasets():
    a, b = _ingest_retail(), ingest.ingest("amz.jsonl", "\n".join(json.dumps(r) for r in amazon_rows()).encode(), category="Appliances")["dataset_id"]
    assert a != b
    _confirm(a); engine.run(a, ["sales_overview"])
    _confirm(b); engine.run(b, ["review_analysis"])
    assert set(R.results(a)) == {"sales_overview"} and set(R.results(b)) == {"review_analysis"}
    assert R.dpath(a) != R.dpath(b) and R.meta(a)["mapping"] != R.meta(b)["mapping"]
    assert "Appliances" in R.meta(b)["sample_note"]
    R.delete(a); assert R.meta(b) is not None and len(R.list_all()) == 2   # olist + b


def test_amazon_reviews_analysis_and_report():
    did = ingest.ingest("amz.jsonl", "\n".join(json.dumps(r) for r in amazon_rows()).encode(), category="Appliances")["dataset_id"]
    _confirm(did)
    out = engine.run(did, ["review_analysis", "sales_overview"])
    assert {s["id"]: s["status"] for s in out["summary"]} == {"review_analysis": "done", "sales_overview": "not_run"}
    res = R.results(did)["review_analysis"]
    cats = {r["category"] for r in res["tables"]["complaint_categories"]}
    assert {"damaged_defective", "delivery_shipping"} & cats
    assert any(f["type"] == "association" for f in res["findings"]) and all(f["type"] in ("observed", "association", "suggestion") for f in res["findings"])
    assert report.build(did)["ok"]


def test_run_requires_confirmed_mapping_and_known_analyses():
    did = _ingest_retail()
    assert not engine.run(did, ["sales_overview"])["ok"]
    _confirm(did)
    assert not engine.run(did, [])["ok"]


def test_failing_analysis_does_not_stop_others(monkeypatch):
    from multids import analyses
    did = _ingest_retail(); _confirm(did)
    monkeypatch.setitem(analyses.REGISTRY, "product_analysis", lambda *a, **k: 1 / 0)
    out = engine.run(did, ["product_analysis", "sales_overview"])
    st = {s["id"]: s["status"] for s in out["summary"]}
    assert st["product_analysis"] == "error" and st["sales_overview"] == "done"


def test_report_needs_results_and_olist_is_read_only():
    did = _ingest_retail()
    assert not report.build(did)["ok"]
    with pytest.raises(PermissionError):
        R.delete("olist")
    with pytest.raises(PermissionError):
        R.save_meta("olist", {})
    assert not R.valid_id("../etc") and not R.valid_id("a b")


# ---------------------------------------------------------------- API
def test_api_flow_and_security(client):
    assert client.get("/multidata").status_code == 200
    r = client.post("/api/md/upload", data={"file": (io.BytesIO(xlsx_bytes(retail_df())), "r.xlsx")}, content_type="multipart/form-data")
    assert r.status_code == 201
    did = r.json["dataset_id"]
    d = client.get(f"/api/md/{did}").json
    mp = {k: v["column"] for k, v in d["suggestion"]["suggested"].items()}
    assert client.post(f"/api/md/{did}/run", json={"analyses": ["sales_overview"]}).status_code == 400          # not confirmed
    assert client.post(f"/api/md/{did}/mapping", json={"mapping": {"date": "Country"}}).status_code == 400
    assert client.post(f"/api/md/{did}/mapping", json={"mapping": mp}).status_code == 200
    assert client.post(f"/api/md/{did}/run", json={"analyses": ["nope"]}).status_code == 400
    assert client.post(f"/api/md/{did}/run", json={"analyses": ["sales_overview"]}).status_code == 200
    assert client.post(f"/api/md/{did}/report").status_code == 200
    assert client.get(f"/api/md/{did}/files/report.pdf").status_code == 200
    for bad in ("meta.json", "data.parquet", "..%2Fmeta.json"):
        assert client.get(f"/api/md/{did}/files/{bad}").status_code == 404
    assert client.get("/api/md/not_a_dataset").status_code == 404
    assert client.get("/api/md/..%2F..").status_code == 404
    assert client.get(f"/api/md/{did}/model-check").json["olist_model_applied"] is False
    assert client.delete(f"/api/md/{did}").status_code == 200 and client.get(f"/api/md/{did}").status_code == 404
    assert client.delete("/api/md/olist").status_code in (403, 404)


def test_api_upload_errors(client):
    assert client.post("/api/md/upload", data={}, content_type="multipart/form-data").status_code == 400
    assert client.post("/api/md/upload", data={"file": (io.BytesIO(b"x"), "a.exe")}, content_type="multipart/form-data").status_code == 400
    assert client.post("/api/md/upload", data={"file": (io.BytesIO(b""), "a.csv")}, content_type="multipart/form-data").status_code == 400
    assert client.post("/api/md/upload", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "a.csv"), "adapter": "weird"}, content_type="multipart/form-data").status_code == 400


def test_olist_entry_unavailable_message_not_crash(client, monkeypatch):
    monkeypatch.setattr(R, "olist_meta", lambda: {"dataset_id": "olist", "name": "x", "available": False})
    assert client.get("/api/md/olist").status_code == 404


def test_v1_olist_upload_limit_unchanged(client):
    big = io.BytesIO(b"a\n" + b"1\n" * (11 * 1024 * 1024))
    assert client.post("/api/upload", data={"file": (big, "big.csv")}, content_type="multipart/form-data").status_code == 413
