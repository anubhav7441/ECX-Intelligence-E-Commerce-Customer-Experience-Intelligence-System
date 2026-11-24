"""Create SYNTHETIC sample files that mimic the *format* of UCI Online Retail II (xlsx) and Amazon Reviews 2023 (jsonl).
They contain RANDOM data - never report their numbers as real findings. They exist so you can try the module without downloading the real files."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "samples"


def retail(seed=1, n=6000):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2010-12-01")
    dates = start + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n), unit="m")
    inv = [str(500000 + i // 3) for i in range(n)]
    cust = rng.integers(12000, 12400, n).astype(float)
    cust[rng.random(n) < 0.2] = np.nan
    sc = rng.choice([f"{x}" for x in range(21000, 21060)] + ["POST", "M", "DOT"], n, p=[0.97 / 60] * 60 + [0.01] * 3)
    q = rng.integers(1, 24, n)
    price = np.round(rng.lognormal(1.0, 0.7, n), 2)
    df = pd.DataFrame({"Invoice": inv, "StockCode": sc, "Description": [f"SAMPLE ITEM {s}" for s in sc], "Quantity": q, "InvoiceDate": dates, "Price": price, "Customer ID": cust,
                       "Country": rng.choice(["United Kingdom", "Germany", "France", "EIRE", "Spain"], n, p=[.8, .06, .06, .04, .04])})
    canc = rng.random(n) < 0.03
    df.loc[canc, "Invoice"] = "C" + df.loc[canc, "Invoice"]; df.loc[canc, "Quantity"] *= -1
    df.loc[rng.random(n) < 0.004, "Price"] = 0
    df.loc[rng.random(n) < 0.002, "Quantity"] = 5000   # planted extreme order
    return pd.concat([df, df.iloc[:20]], ignore_index=True)


def amazon(seed=2, n=4000):
    rng = np.random.default_rng(seed)
    good = ["works great", "love it", "excellent quality", "easy to install", "fast delivery", "very quiet", "good value for the price", "highly recommend"]
    bad = ["stopped working after a month", "arrived damaged", "poor quality and flimsy", "wrong part sent", "terrible customer service", "late delivery, took weeks", "not worth the money", "leaks and is very noisy", "had to return it for a refund"]
    rows = []
    for i in range(n):
        r = int(rng.choice([1, 2, 3, 4, 5], p=[.1, .07, .1, .2, .53]))
        pool = bad if r <= 2 else good
        txt = ". ".join(rng.choice(pool, 2, replace=False)) + "."
        rows.append({"rating": float(r), "title": txt.split(".")[0][:30], "text": txt, "images": [], "asin": f"B0{rng.integers(10000, 10200)}", "parent_asin": f"B0{rng.integers(10000, 10200)}",
                     "user_id": f"U{rng.integers(1, 3000)}", "timestamp": int(pd.Timestamp("2019-01-01").timestamp() * 1000 + rng.integers(0, 3 * 365 * 86400) * 1000), "helpful_vote": int(rng.integers(0, 5)), "verified_purchase": bool(rng.random() < .9)})
    return rows


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    df = retail()
    with pd.ExcelWriter(OUT / "SYNTHETIC_online_retail_like.xlsx") as w:
        df.iloc[:3000].to_excel(w, sheet_name="Year 2009-2010", index=False); df.iloc[3000:].to_excel(w, sheet_name="Year 2010-2011", index=False)
    (OUT / "SYNTHETIC_amazon_appliances_like.jsonl").write_text("\n".join(json.dumps(r) for r in amazon()))
    (OUT / "README.txt").write_text("These files are SYNTHETIC (random) and only mimic the file formats. Do not report their numbers as findings.\nReal data: https://archive.ics.uci.edu/dataset/502/online+retail+ii and https://amazon-reviews-2023.github.io/\n")
    print("written to", OUT)
