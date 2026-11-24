"""Orchestrates profiling, mapping confirmation and analyses for ONE dataset at a time (isolation by dataset id)."""
import pandas as pd
from multids import analyses as A, capabilities, mapping as M, profile as P, registry as R


def get_profile(did):
    df = R.load_df(did)
    return P.profile(df)


def numeric_count(did):
    df = R.load_df(did)
    return sum(1 for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() > 2)


def suggest_mapping(did):
    return M.suggest(R.load_df(did))


def confirm_mapping(did, mapping):
    df = R.load_df(did)
    clean, errs, warns = M.validate(df, mapping)
    if errs:
        return {"ok": False, "errors": errs, "warnings": warns}
    m = R.meta(did)
    m["mapping"], m["mapping_confirmed"] = clean, True
    R.save_meta(did, m)
    R.log(did, "mapping confirmed", {"mapping": clean, "warnings": warns})
    return {"ok": True, "mapping": clean, "warnings": warns, "capabilities": capabilities.evaluate(clean, numeric_count(did))}


def capability_list(did):
    m = R.meta(did)
    return capabilities.evaluate(m.get("mapping") or {}, numeric_count(did))


def run(did, analysis_ids):
    m = R.meta(did)
    if not m.get("mapping_confirmed"):
        return {"ok": False, "error": "Confirm the column mapping first."}
    if not analysis_ids:
        return {"ok": False, "error": "Select at least one analysis."}
    df = R.load_df(did)
    mp = m["mapping"]
    std, prep_notes = M.prepare(df, mp)
    out = R.report_dir(did)
    results = R.results(did) or {}
    ncols = numeric_count(did)
    summary = []
    for aid in analysis_ids:
        cap = capabilities.can_run(aid, mp, ncols)
        if not cap["available"]:
            results[aid] = {"status": "not_run", "reason": cap["reason"], "title": next((a["title"] for a in capabilities.evaluate(mp, ncols) if a["id"] == aid), aid)}
            summary.append({"id": aid, "status": "not_run", "reason": cap["reason"]})
            continue
        try:
            if aid == "numeric_statistics":
                excl = [mp[r] for r in ("order_id", "customer_id", "product_id", "seller_id") if r in mp]
                r = A.numeric_statistics(df, out, exclude=excl)
            else:
                r = A.REGISTRY[aid](std, out)
            r["status"] = "skipped" if r.get("skipped") else "done"
            r["title"] = capabilities.ANALYSES[aid]["title"]
            r["data_prep_notes"] = prep_notes
            results[aid] = r
            summary.append({"id": aid, "status": r["status"], "findings": len(r["findings"])})
        except Exception as e:   # one failing analysis must not stop the others
            results[aid] = {"status": "error", "title": capabilities.ANALYSES[aid]["title"], "reason": f"{e.__class__.__name__}: {str(e)[:200]}"}
            summary.append({"id": aid, "status": "error", "reason": results[aid]["reason"]})
    R.save_results(did, results)
    R.log(did, "analyses run", {"requested": analysis_ids, "summary": summary})
    return {"ok": True, "summary": summary}
