"""Visual overview: picks the charts that suit the columns of THIS dataset and explains each in plain words.
Every number in an explanation is computed from the data. Nothing is invented; if a chart does not fit, it is not drawn."""
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multids import mapping as M, registry as R
from multids.profile import column_kind

# dark dashboard theme (purple / blue / red on navy cards)
BG, INK, MUTE, GRID = "#141a31", "#e8eaf8", "#9aa0c4", "#2a3157"
BLUE, ORANGE, GREEN, GREY, PURPLE, RED = "#3b82f6", "#f97316", "#22c55e", "#6b7299", "#8b5cf6", "#ef4444"
MAX_CHARTS = 6
W, H = 4.6, 2.5


def _style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTE, labelsize=8.5, length=2)
    ax.grid(axis="y", color=GRID, lw=.5, ls=":")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(MUTE); ax.yaxis.label.set_color(MUTE)


def _fig(w=W, h=H):
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    fig.patch.set_facecolor(BG); _style(ax)
    return fig, ax


def _save(fig, path):
    for ax in fig.axes:
        tt = ax.get_title()
        if tt:
            ax.set_title("", loc="center"); ax.set_title(tt, loc="left", color=INK, fontsize=11)
    fig.tight_layout(pad=.8); fig.savefig(path, facecolor=BG); plt.close(fig)


def _k(v, _=None):
    v = abs(v)
    return f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k" if v >= 1e4 else f"{v:,.0f}"


def _n(x):
    x = float(x)
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}"


def _hist_spec(s, bins, label="Count"):
    cnt, edges = np.histogram(np.asarray(s, dtype=float), bins=max(1, bins))
    labels = [f"{edges[i]:,.1f} to {edges[i + 1]:,.1f}" if edges[1] - edges[0] < 10 else f"{edges[i]:,.0f} to {edges[i + 1]:,.0f}" for i in range(len(cnt))]
    return {"type": "bar", "labels": labels, "values": [int(c) for c in cnt], "label": label, "color": BLUE, "gap": True}


def build(did, force=False):
    out = R.report_dir(did)
    meta_f = out / "visuals.json"
    cur_map = (R.meta(did) or {}).get("mapping") or {}
    if meta_f.exists() and not force:
        try:
            cached = json.loads(meta_f.read_text(encoding="utf-8"))
            if cached.get("mapping") == cur_map and cached.get("v") == 2:
                return cached
        except Exception:
            pass
    for old in out.glob("viz_*.png"):
        old.unlink()
    meta = R.meta(did) or {}
    df = R.load_df(did)
    if len(df) > 400_000:
        df = df.sample(400_000, random_state=42)
    mp = {k: v for k, v in (meta.get("mapping") or {}).items() if v in df.columns}
    std, _ = M.prepare(df, mp) if mp else (pd.DataFrame(index=df.index), [])
    charts, used = [], set(mp.values())

    box = {}

    def add(name, title, why, explain, fig, wide=False):
        _save(fig, out / name)
        charts.append({"file": name, "title": title, "why": why, "explain": explain, "wide": wide, "spec": box.pop("s", None)})

    rev = std["revenue"] if "revenue" in std else None
    pos = rev[rev > 0] if rev is not None else None
    has_date = "date" in std and std.date.notna().sum() > 10
    # 1. over time: area chart (best for a trend)
    if has_date:
        if pos is not None:
            ser = pos.groupby(std.date[pos.index].dt.to_period("M")).sum(); what, unit = "sales", "Sales per month"
        else:
            ser = std.date.dropna().dt.to_period("M").value_counts().sort_index(); what, unit = "rows", "Records per month"
        if len(ser) >= 3:
            x = np.arange(len(ser)); fig, ax = _fig(W * 1.35, H)
            ax.fill_between(x, ser.values, color=PURPLE, alpha=.35); ax.plot(x, ser.values, color=PURPLE, lw=1.8)
            ax.scatter([int(np.argmax(ser.values))], [ser.values.max()], color="#fff", s=18, zorder=3)
            step = max(1, len(ser) // 8); ax.set_xticks(x[::step]); ax.set_xticklabels([str(i) for i in ser.index[::step]], rotation=0, fontsize=6.5)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_k)); ax.set_title(unit)
            hi, lo = ser.idxmax(), ser.idxmin(); third = max(1, len(ser) // 3)
            first, last = ser.iloc[:third].mean(), ser.iloc[-third:].mean()
            direction = "higher" if last > first * 1.1 else "lower" if last < first * 0.9 else "about the same"
            box["s"] = {"type": "line", "labels": [str(i) for i in ser.index], "values": [round(float(v), 2) for v in ser.values], "label": "Sales" if what == "sales" else "Records", "color": PURPLE}
            add("viz_sales_time.png", "Trend over time", "You have a date column, so an area chart over months shows growth, dips and seasons best.",
                f"Peak month {hi} ({_n(ser.max())}), lowest {lo} ({_n(ser.min())}). The last third of the period is {direction} than the first third. Check the final month is complete.", fig, wide=True)
    # 2. top products: horizontal bars
    prod = std["product_name"] if "product_name" in std else std["product_id"] if "product_id" in std else None
    if prod is not None and pos is not None:
        g = pos.groupby(prod[pos.index]).sum().sort_values(ascending=False)
        if len(g) >= 3:
            tp = g.head(8); share = tp.sum() / g.sum()
            fig, ax = _fig(); ax.barh([str(i)[:22] for i in tp.index][::-1], tp.values[::-1], color=BLUE); ax.set_title("Top products by sales"); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_k)); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=.5, ls=":")
            box["s"] = {"type": "hbar", "labels": [str(i)[:40] for i in tp.index], "values": [round(float(v), 2) for v in tp.values], "label": "Sales", "color": BLUE}
            add("viz_top_products.png", "Best-selling products", "A ranking is clearest as horizontal bars: long labels stay readable and sizes compare at a glance.",
                f"The top {len(tp)} make {share:.0%} of sales ({len(g)} products in total); best is '{str(g.index[0])[:30]}' ({_n(g.iloc[0])}). "
                + ("Income depends on a few products: keep them in stock." if share >= 0.5 else "Sales are fairly spread, which lowers risk."), fig)
    # 3. regions: donut for few parts of a whole, bars when many
    if "region" in std and pos is not None:
        g = pos.groupby(std.region[pos.index]).sum().sort_values(ascending=False)
        if len(g) >= 2:
            if len(g) <= 6:
                fig, ax = plt.subplots(figsize=(W, H), dpi=130); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                cols = [PURPLE, BLUE, RED, GREEN, ORANGE, GREY]
                ax.pie(g.values, colors=cols[:len(g)], startangle=90, counterclock=False, wedgeprops=dict(width=.38, edgecolor=BG, linewidth=2))
                ax.legend([f"{str(i)[:14]}  {v / g.sum():.0%}" for i, v in g.items()], loc="center left", bbox_to_anchor=(1, .5), frameon=False, labelcolor=INK, fontsize=7.5)
                ax.set_title("Share of sales by region", color=INK, fontsize=9, loc="left")
            else:
                tp = g.head(8); fig, ax = _fig(); ax.bar([str(i)[:10] for i in tp.index], tp.values, color=GREEN); ax.set_title("Sales by region"); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_k)); ax.tick_params(axis="x", rotation=35)
            box["s"] = {"type": "donut", "labels": [str(i)[:20] for i in g.index], "values": [round(float(v), 2) for v in g.values], "label": "Sales"} if len(g) <= 6 else {"type": "bar", "labels": [str(i)[:20] for i in tp.index], "values": [round(float(v), 2) for v in tp.values], "label": "Sales", "color": GREEN}
            add("viz_regions.png", "Sales by region", "Few regions are parts of one whole, so a donut shows each share; many regions need bars." if len(g) <= 6 else "Many regions are easier to compare as bars.",
                f"'{g.index[0]}' is the largest region with {g.iloc[0] / g.sum():.0%} of sales; the smallest shown is '{g.index[min(len(g), 8) - 1]}' with {g.iloc[min(len(g), 8) - 1] / g.sum():.0%}. If one region dominates, test small marketing budgets in the next best.", fig)
    # 4. ratings
    if "rating" in std and std.rating.notna().sum() >= 20:
        r = std.rating.dropna(); vc = r.round().value_counts().sort_index()
        fig, ax = _fig(); ax.bar(vc.index.astype(int).astype(str), vc.values, color=[RED if i <= 2 else PURPLE if i >= 4 else GREY for i in vc.index], width=.65); ax.set_title("Ratings given")
        low, high = (r <= 2).mean(), (r >= 4).mean()
        box["s"] = {"type": "bar", "labels": [str(int(i)) for i in vc.index], "values": [int(v) for v in vc.values], "label": "Ratings", "colors": [RED if i <= 2 else PURPLE if i >= 4 else GREY for i in vc.index]}
        add("viz_ratings.png", "Customer ratings", "A score with a few fixed values is best shown as one bar per value.",
            f"Average {r.mean():.2f} from {len(r):,} ratings: {high:.0%} high (4-5), {low:.0%} low (1-2). A tall red bar means unhappy customers; read those reviews for the cause.", fig)
    # 5. customers
    if "customer_id" in std and len(std) > 50:
        oc = std.groupby("customer_id").size()
        if oc.nunique() > 1 and len(oc) >= 30:
            fig, ax = _fig(); ax.hist(oc.clip(upper=oc.quantile(.99)), bins=min(25, int(oc.max())), color=BLUE); ax.set_title("Purchases per customer"); ax.set_xlabel("purchases")
            box["s"] = _hist_spec(oc.clip(upper=oc.quantile(.99)), min(25, int(oc.max())), "Customers")
            add("viz_customers.png", "How often customers buy", "A histogram shows how loyalty is spread across customers.",
                f"{len(oc):,} customers; typical is {oc.median():.0f} purchase(s), most active {oc.max():,}. {(oc == 1).mean():.0%} bought only once, so a tall first bar suggests a win-back offer could pay off.", fig)
    # 6. delivery
    if "delivery_days" in std and std.delivery_days.dropna().ge(0).sum() >= 20:
        d = std.delivery_days.dropna(); d = d[d >= 0]
        fig, ax = _fig(); ax.hist(d.clip(upper=d.quantile(.99)), bins=30, color=PURPLE); ax.axvline(d.median(), color="#fff", ls="--", lw=1); ax.set_title("Delivery time (days)")
        box["s"] = _hist_spec(d.clip(upper=d.quantile(.99)), 30, "Orders")
        add("viz_delivery.png", "Delivery time", "A histogram shows both the typical wait and the slow tail customers complain about.",
            f"Median {d.median():.1f} days (dashed line); 90% arrive within {d.quantile(.9):.1f} days.", fig)
    # generic charts: only when the file has too little business structure for the charts above
    num = [c for c in df.columns if c not in used and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() > 5 and np.isfinite(df[c].dropna()).all()]
    if len(charts) < 2 and num:
        for c in num[:3]:
            s = df[c].dropna(); sc = s.clip(s.quantile(.01), s.quantile(.99))
            fig, ax = _fig(); ax.hist(sc, bins=22, color=PURPLE); ax.set_title(f"Spread of '{str(c)[:20]}'")
            box["s"] = _hist_spec(sc, 22, "Rows")
            add(f"viz_dist_{len(charts)}.png", f"Spread of '{c}'", "No business columns were recognised, so number columns are shown by how their values are spread.",
                f"Typical (median) value {s.median():,.2f}, average {s.mean():,.2f}, range {s.min():,.2f} to {s.max():,.2f}. Skewness {s.skew():.1f}: " + ("a few extreme values pull the average, so the median is fairer." if abs(s.skew()) > 1 else "values are fairly balanced around the middle."), fig)
        if len(num) >= 3 and len(df[num[:8]].dropna()) >= 30:
            cols = num[:8]; corr = df[cols].dropna().corr(method="spearman")
            fig, ax = _fig(W, H * 1.4); ax.grid(False); im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols))); ax.set_xticklabels([str(c)[:10] for c in cols], rotation=60, ha="right", fontsize=6); ax.set_yticklabels([str(c)[:10] for c in cols], fontsize=6)
            ax.set_title("Which columns move together?"); a = corr.where(~np.eye(len(cols), dtype=bool)).abs(); i, j = np.unravel_index(np.nanargmax(a.values), a.shape); r = corr.iloc[i, j]
            box["s"] = {"type": "heat", "labels": [str(c)[:14] for c in cols], "matrix": [[round(float(x), 2) for x in row] for row in corr.values]}
            add("viz_correlation.png", "Relationships between number columns", "With several number columns, a colour grid is the quickest way to see which are related.",
                (f"Strongest link: '{cols[i]}' and '{cols[j]}' (correlation {r:.2f}). Red = rise together, blue = one rises as the other falls. Correlation is not cause." if abs(r) >= 0.3 else "All squares are pale: no columns clearly move together. That is a real result, not an error."), fig)
    if len(charts) < 2:
        cats = [c for c in df.columns if c not in used and column_kind(df[c]) == "categorical" and 2 <= df[c].nunique() <= 60]
        if cats:
            c = cats[0]; vc = df[c].value_counts().head(8)
            fig, ax = _fig(); ax.barh([str(i)[:20] for i in vc.index][::-1], vc.values[::-1], color=GREEN); ax.set_title(f"Most common: {str(c)[:20]}"); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=.5, ls=":")
            box["s"] = {"type": "hbar", "labels": [str(i)[:30] for i in vc.index], "values": [int(v) for v in vc.values], "label": "Rows", "color": GREEN}
            add("viz_categories.png", f"Most common values: {c}", "A column of repeated labels is best shown as a ranked count.", f"'{vc.index[0]}' is the most common value ({vc.iloc[0] / len(df):.0%} of rows); {df[c].nunique()} different values exist.", fig)
    # missing data: only when it really matters
    miss = df.isna().mean().sort_values(ascending=False); miss = miss[miss > 0.2].head(6)
    if len(miss) and len(charts) < MAX_CHARTS:
        fig, ax = _fig(); ax.barh([str(i)[:20] for i in miss.index][::-1], (miss.values * 100)[::-1], color=RED); ax.set_title("Empty cells (%)"); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=.5, ls=":")
        box["s"] = {"type": "hbar", "labels": [str(i)[:30] for i in miss.index], "values": [round(float(v) * 100, 1) for v in miss.values], "label": "Empty %", "color": RED}
        add("viz_missing.png", "Missing data", "Columns that are mostly empty weaken results, so only those above 20% are shown.",
            f"'{miss.index[0]}' is {miss.iloc[0]:.0%} empty. Ask the data owner to fill it, or avoid relying on it.", fig)
    res = {"charts": charts[:MAX_CHARTS], "rows_used": int(len(df)), "mapping": cur_map, "v": 2}
    meta_f.write_text(json.dumps(res), encoding="utf-8")
    return res
