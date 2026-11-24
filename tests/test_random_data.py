"""Robustness on arbitrary / random uploads (synthetic fixtures only)."""
import io
import numpy as np
import pandas as pd
import pytest

from multids import engine, ingest, registry as R
from multids.profile import coerce_numeric_text, profile
from app.insights import dataset as ID


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIDS_DIR", str(tmp_path / "datasets"))


def _up(df, name="x.csv"):
    b = io.BytesIO(); df.to_csv(b, index=False)
    return ingest.ingest(name, b.getvalue())["dataset_id"]


def _run_all(did):
    m = {k: v["column"] for k, v in engine.suggest_mapping(did)["suggested"].items()}
    assert engine.confirm_mapping(did, m).get("ok")
    ids = [c["id"] for c in engine.capability_list(did) if c.get("available")]
    out = engine.run(did, ids) if ids else {"summary": []}
    assert not [s for s in out.get("summary", []) if s["status"] == "error"]
    return m


def test_numeric_text_is_converted():
    df = pd.DataFrame({"price": ["$1.50", "$2,000.00", "3", "4", "5", "6"], "name": list("abcdef")})
    out, notes = coerce_numeric_text(df)
    assert pd.api.types.is_numeric_dtype(out["price"]) and out["price"].iloc[1] == 2000 and notes
    assert out["name"].tolist() == list("abcdef")


def test_price_written_with_currency_symbol_is_mapped():
    rng = np.random.default_rng(0)
    did = _up(pd.DataFrame({"price": [f"${x:.2f}" for x in rng.uniform(1, 99, 200)], "qty": rng.integers(1, 9, 200)}))
    m = {k: v["column"] for k, v in engine.suggest_mapping(did)["suggested"].items()}
    assert m.get("unit_price") == "price" and m.get("quantity") == "qty"


def test_random_text_reviews_do_not_crash():
    rng = np.random.default_rng(1)
    w = lambda: "".join(rng.choice(list("abcdefghij"), 12))
    did = _up(pd.DataFrame({"review": [w() + " " + w() for _ in range(400)], "rating": rng.integers(1, 6, 400)}))
    _run_all(did)


@pytest.mark.parametrize("shape", ["pure_numbers", "strings", "constant", "one_col", "three_rows", "all_null"])
def test_random_shapes_never_crash(shape):
    rng = np.random.default_rng(2); n = 300
    w = lambda: "".join(rng.choice(list("abcdefgh"), 6))
    df = {"pure_numbers": pd.DataFrame(rng.normal(size=(n, 5)), columns=list("ABCDE")),
          "strings": pd.DataFrame({"a": [w() for _ in range(n)], "b": [w() for _ in range(n)]}),
          "constant": pd.DataFrame({"a": [1] * n, "b": ["x"] * n}),
          "one_col": pd.DataFrame({"a": rng.integers(0, 5, n)}),
          "three_rows": pd.DataFrame({"a": [1, 2, 3], "b": list("xyz")}),
          "all_null": pd.DataFrame({"a": rng.normal(size=n), "b": [None] * n})}[shape]
    did = _up(df)
    _run_all(did)
    r = ID.build(did)
    assert r["ready"] in (True, False) and isinstance(r["cards"], list)


def test_random_sales_data_gives_no_made_up_season_or_win_back():
    rng = np.random.default_rng(3); n = 600
    df = pd.DataFrame({"customer": rng.integers(1, 80, n), "item": rng.choice(list("abcdefghijklmnop"), n), "qty": rng.integers(1, 5, n),
                       "price": rng.uniform(1, 50, n), "date": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")})
    did = _up(df); _run_all(did)
    ids = {c["key"] for c in ID.build(did)["cards"]}
    assert "seasonality" not in ids and "win_back" not in ids


def test_real_seasonal_pattern_is_still_reported():
    rng = np.random.default_rng(4); rows = []
    for mth in range(1, 13):
        k = 400 if mth == 12 else 40
        for _ in range(k):
            rows.append({"date": pd.Timestamp(2022, mth, int(rng.integers(1, 28))), "amount": float(rng.uniform(10, 50)), "product": f"p{rng.integers(0, 30)}"})
    did = _up(pd.DataFrame(rows)); _run_all(did)
    assert "seasonality" in {c["key"] for c in ID.build(did)["cards"]}


def test_small_file_gets_honest_message_and_warning():
    did = _up(pd.DataFrame({"a": [1, 2, 3], "b": list("xyz")}))
    assert any("rows" in i["issue"] for i in profile(R.load_df(did))["issues"])
    engine.confirm_mapping(did, {})
    assert ID.build(did)["ready"] is False
