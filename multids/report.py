"""Evidence-based report generator (PDF via reportlab + supporting CSV exports).
The report is assembled ONLY from stored computation results; nothing is invented. Sections follow the 14-point specification."""
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape
from multids import capabilities, registry as R
from multids.profile import profile as make_profile

SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=SS["Heading1"], fontSize=15, textColor=colors.HexColor("#0b0b0b"), spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=SS["Heading2"], fontSize=11.5, textColor=colors.HexColor("#2a78d6"), spaceBefore=8, spaceAfter=3)
B = ParagraphStyle("B", parent=SS["BodyText"], fontSize=9.2, leading=12.5)
SM = ParagraphStyle("SM", parent=B, fontSize=7.8, leading=10, textColor=colors.HexColor("#52514e"))
TAG = {"observed": ("OBSERVED", "#1baf7a"), "association": ("ASSOCIATION", "#2a78d6"), "suggestion": ("SUGGESTION (hypothesis)", "#eb6834")}
P = lambda t, s=B: Paragraph(escape(str(t)), s)


def _tbl(rows, header=None, widths=None, font=7.5):
    data = ([header] if header else []) + rows
    data = [[Paragraph(escape(str(c)), ParagraphStyle("c", parent=SM, fontSize=font, leading=font + 2)) for c in r] for r in data]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f0ea")), ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#e1e0d9")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def _findings(res, kinds=None):
    return [f for r in res.values() if r.get("status") == "done" for f in r.get("findings", []) if not kinds or f["type"] in kinds]


def write_exports(did, meta, prof, res, out):
    files = []
    pd.DataFrame(prof["columns"]).to_csv(out / "data_quality_columns.csv", index=False); files.append("data_quality_columns.csv")
    pd.DataFrame(prof["issues"] or [{"severity": "", "issue": "none"}]).to_csv(out / "data_quality_issues.csv", index=False); files.append("data_quality_issues.csv")
    pd.DataFrame(R.history(did)).to_csv(out / "processing_history.csv", index=False); files.append("processing_history.csv")
    for aid, r in res.items():
        if r.get("status") != "done":
            continue
        for tname, rows in r.get("tables", {}).items():
            if rows:
                fn = f"{aid}__{tname}.csv"
                pd.DataFrame(rows).to_csv(out / fn, index=False); files.append(fn)
        files += [f for f in r.get("exports", []) if (out / f).exists()]
    return sorted(set(files))


def _img(path, width):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def build(did):
    meta = R.meta(did)
    res = R.results(did) or {}
    done = {k: v for k, v in res.items() if v.get("status") == "done"}
    if not done:
        return {"ok": False, "error": "Run at least one analysis before generating a report."}
    df = R.load_df(did)
    prof = make_profile(df)
    out = R.report_dir(did)
    exports = write_exports(did, meta, prof, res, out)
    pdf = out / "report.pdf"
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title=f"ECX-Intelligence report - {meta.get('name')}", author="ECX-Intelligence Multi-Dataset Module")
    W = A4[0] - 3.6 * cm
    S = []
    S += [P("ECX-Intelligence - Automated Analysis Report", ParagraphStyle("T", parent=SS["Title"], fontSize=20)), P(f"Dataset: {meta.get('name')}  |  ID: {did}", B),
          P(f"Generated from {meta.get('n_rows', len(df)):,} rows x {df.shape[1]} columns. Every number below comes from computation in this software; findings are labelled OBSERVED, ASSOCIATION (statistical, not causal) or SUGGESTION (hypothesis).", SM)]
    obs, asc = _findings(res, {"observed"}), _findings(res, {"association"})
    # 1 executive summary
    S.append(P("1. Executive summary", H1))
    S.append(P(f"This report analyses '{meta.get('name')}' ({len(df):,} rows, {df.shape[1]} columns, data quality score {prof['quality_score']}/100). "
               f"{len(done)} of {len(res)} requested analyses ran; the rest could not run because required columns were not available (see section 6). "
               f"The analyses produced {len(obs)} observed facts and {len(asc)} statistical associations."))
    for r in done.values():
        for f in r["findings"][:2]:
            if f["type"] != "suggestion":
                tag, col = TAG[f["type"]]
                S.append(Paragraph(f'<font color="{col}"><b>{tag}</b></font> {escape(f["text"])}', B))
    # 2 description
    S.append(P("2. Dataset description and source", H1))
    S.append(P(meta.get("description") or "User-supplied dataset."))
    S.append(P(f"Source / attribution: {meta.get('source') or 'user upload (no source stated)'}" + (f" - {meta.get('source_url')}" if meta.get("source_url") else "")))
    if meta.get("sample_note"):
        S.append(P("Sample: " + meta["sample_note"]))
    S.append(P(f"Adapter: {meta.get('adapter')}. Created: {meta.get('created_at') or 'n/a (built-in)'}.", SM))
    # 3 dimensions
    S.append(P("3. Dataset dimensions and column information", H1))
    S.append(P(f"{prof['n_rows']:,} rows, {prof['n_cols']} columns, {prof['memory_mb']} MB in memory. Numeric: {len(prof['by_kind']['numeric'])}; categorical: {len(prof['by_kind']['categorical'])}; date: {len(prof['by_kind']['date'])}; text: {len(prof['by_kind']['text'])}; identifier-like: {len(prof['by_kind']['id'])}."))
    rows = [[c["name"], c["kind"], c["dtype"], f"{c['missing_pct']}%", c["unique"], c["example"] or ""] for c in prof["columns"][:60]]
    S.append(_tbl(rows, ["Column", "Kind", "Type", "Missing", "Unique", "Example"], [W * .26, W * .1, W * .12, W * .09, W * .1, W * .33]))
    if prof["n_cols"] > 60:
        S.append(P(f"(first 60 of {prof['n_cols']} columns shown; all in data_quality_columns.csv)", SM))
    S.append(P("Confirmed column mapping (role -> column): " + (", ".join(f"{k} -> {v}" for k, v in (meta.get("mapping") or {}).items()) or "none"), SM))
    # 4 quality
    S.append(P("4. Data quality assessment", H1))
    S.append(P(f"Quality score {prof['quality_score']}/100 (penalises missing values and duplicates; a simple heuristic). Duplicate rows: {prof['duplicate_rows']:,}."))
    for i in prof["issues"][:15]:
        S.append(P(f"[{i['severity']}] {i['issue']}", SM))
    if not prof["issues"]:
        S.append(P("No quality issues were detected by the automatic checks."))
    # 5 cleaning
    S.append(P("5. Cleaning and preprocessing performed", H1))
    for h in R.history(did):
        d = h.get("detail")
        if isinstance(d, dict) and "summary" in d:
            d = "; ".join(f"{x['id']}: {x['status']}" for x in d["summary"])
        elif isinstance(d, dict) and "mapping" in d:
            d = "mapping confirmed by the user (" + ", ".join(f"{k}->{v}" for k, v in d["mapping"].items()) + ")"
        elif isinstance(d, (dict, list)):
            d = str(d)[:300]
        S.append(P(f"{h.get('step')}: {d}", SM))
    for r in done.values():
        for n in r.get("data_prep_notes", [])[:1] + r.get("notes", []):
            S.append(P(f"[{r['title']}] {n}", SM))
    # 6 EDA
    S.append(P("6. Exploratory data analysis", H1))
    ran = [(a["id"], a["title"]) for a in capabilities.evaluate(meta.get("mapping") or {}, 0)]
    for aid, r in res.items():
        if r.get("status") != "done":
            S.append(P(f"{r.get('title', aid)}: not performed - {r.get('reason', 'skipped')}", SM))
    for r in done.values():
        S.append(P(r["title"], H2))
        for tname, rows in r.get("tables", {}).items():
            if rows:
                cols = list(rows[0].keys())[:7]
                S += [P(tname.replace("_", " "), SM), _tbl([[(f"{v:.3f}" if isinstance(v, float) else v) for v in (row.get(c, "") for c in cols)] for row in rows[:10]], cols)]
    S.append(P("Charts are shown in section 8; statistical findings in section 9.", SM))
    # 7 KPIs
    S.append(P("7. Relevant KPIs", H1))
    krows = [[r["title"], k["label"], k["value"]] for r in done.values() for k in r.get("kpis", [])]
    S.append(_tbl(krows, ["Analysis", "KPI", "Value"], [W * .3, W * .4, W * .3]) if krows else P("No KPIs available."))
    # 8 charts
    S.append(P("8. Charts and visualizations", H1))
    try:
        from multids import visuals
        for ch in visuals.build(did).get("charts", []):
            f = out / ch["file"]
            if f.exists():
                S += [P(ch["title"], H2) if "H2" in globals() else P("<b>" + ch["title"] + "</b>", B), _img(f, W * (.8 if ch.get("wide") else .6)), P(ch["explain"], SM), Spacer(1, 6)]
    except Exception:
        pass
    for r in done.values():
        for ch in r.get("charts", []):
            f = out / ch["file"]
            if f.exists():
                S += [Image(str(f), width=W * .85, height=W * .85 * .48), P(f"Figure: {ch['caption']}", SM), Spacer(1, 4)]
    # 9 stats
    S.append(P("9. Statistical findings", H1))
    for r in done.values():
        S.append(P(r["title"], H2))
        for f in r["findings"]:
            tag, col = TAG[f["type"]]
            S.append(Paragraph(f'<font color="{col}"><b>{tag}</b></font> {escape(f["text"])}', B))
    # 10 ML
    S.append(P("10. Machine learning results", H1))
    S.append(P("No predictive model was trained or applied. The Olist-trained models were NOT applied to this dataset: their features, target definition and training data are Olist-specific. "
               "A new model on this dataset would need a confirmed target and separate training (not performed here), so no model metrics are reported."))
    # 11 anomalies
    S.append(P("11. Anomaly findings", H1))
    an = done.get("numeric_statistics")
    if an and an.get("tables", {}).get("top_anomalies"):
        S.append(P("Unsupervised Isolation Forest flagged the most unusual rows (top 1%). These are investigation signals, not proof of errors; the full list is in anomalies.csv.", B))
        rows = an["tables"]["top_anomalies"][:8]
        cols = list(rows[0].keys())[:8]
        S.append(_tbl([[(f"{v:.3f}" if isinstance(v, float) else v) for v in (r.get(c, "") for c in cols)] for r in rows], cols))
    else:
        S.append(P("Anomaly detection was not applicable or not run for this dataset."))
    # 12 insights
    S.append(P("12. Business insights and recommendations", H1))
    S.append(P("Recommendations are hypotheses derived from the findings above; they were not tested and do not prove cause and effect.", SM))
    sug = _findings(res, {"suggestion"})
    for f in sug:
        S.append(Paragraph(f'<font color="#eb6834"><b>SUGGESTION</b></font> {escape(f["text"])}', B))
    if not sug:
        S.append(P("No specific recommendations were generated from the available findings."))
    try:                                   # plain-English advice cards (problem / why / what to do), same engine as the Insights page
        from app.insights import dataset as _adv
        cards = [c for c in _adv.build(did).get("cards", []) if c["priority"] != "Good news"][:8] if did != R.OLIST_ID else []
    except Exception:
        cards = []
    for c in cards:
        S.append(Spacer(1, 6))
        S.append(Paragraph(f'<b>{escape(c["title"])}</b> <font color="#6b7280">({escape(c["priority"])} priority, {escape(c["confidence"])})</font>', B))
        S.append(Paragraph(f'<font color="#dc2626"><b>Problem:</b></font> {escape(c["problem"])}', B))
        S.append(Paragraph(f'<font color="#d97706"><b>Why it is happening:</b></font> {escape(c["why"])}', B))
        S.append(Paragraph('<font color="#16a34a"><b>What you can do:</b></font> ' + escape(" ".join(f"({i + 1}) {t}" for i, t in enumerate(c["solutions"]))), B))
    # 13 limitations
    S.append(P("13. Limitations and assumptions", H1))
    lim = ["Observational data: associations are not causal effects.", "Results describe this dataset only and may not generalise.", "Automatic column mapping was confirmed by a user; wrong mapping would change results."]
    for r in done.values():
        lim += r.get("assumptions", [])
    for l in dict.fromkeys(lim):
        S.append(P("- " + l, B))
    nr = [r for r in res.values() if r.get("status") != "done"]
    for r in nr:
        S.append(P(f"- Not performed ({r.get('title')}): {r.get('reason')}", SM))
    # 14 conclusion
    S.append(P("14. Conclusion", H1))
    S.append(P(f"{len(done)} analyses were completed on '{meta.get('name')}'. The report lists {len(obs)} observed facts and {len(asc)} statistical associations, plus {len(sug)} hypothetical suggestions for human review. "
               "All figures are reproducible from the dataset and the confirmed mapping using the exported CSVs and the processing history."))
    S.append(P("Supporting files: " + ", ".join(exports[:25]) + (" ..." if len(exports) > 25 else ""), SM))
    doc.build(S, onFirstPage=_foot, onLaterPages=_foot)
    R.log(did, "report generated", {"pdf": "report.pdf", "exports": len(exports)})
    return {"ok": True, "pdf": "report.pdf", "exports": exports, "analyses_included": list(done)}


def _foot(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 7); canvas.setFillColor(colors.HexColor("#898781"))
    canvas.drawString(1.8 * cm, 1 * cm, "ECX-Intelligence - Multi-Dataset Module | generated automatically; see limitations section")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1 * cm, f"Page {doc.page}"); canvas.restoreState()
