"""Upload pipeline: load -> optional adapter cleaning -> register as an isolated dataset (own id, folder, history)."""
import re
from pathlib import Path
from multids import adapters, registry as R
from multids.profile import LoadError, load_table, coerce_numeric_text


def clean_name(n, fallback):
    n = re.sub(r"[^\w .\-()]", "", (n or "").strip())[:80]
    return n or re.sub(r"[^\w .\-()]", "", Path(fallback or "dataset").stem)[:80] or "dataset"


def ingest(filename, data, name=None, adapter="auto", category=None, sheet=None, source=None):
    df, notes = load_table(filename, data, sheet)           # raises LoadError
    history = [{"time": R.now(), "step": "file received", "detail": f"{filename}: {len(df):,} rows x {df.shape[1]} columns."}] + [{"time": R.now(), "step": "load note", "detail": n} for n in notes]
    kind = adapters.detect(df) if adapter in (None, "", "auto") else (adapter if adapter in adapters.INFO else None)
    if adapter not in (None, "", "auto", "none") and adapter not in adapters.INFO:
        raise LoadError(f"Unknown adapter '{adapter}'.")
    meta = {"name": clean_name(name, filename), "adapter": kind or "generic", "original_filename": Path(filename).name[:120],
            "source": source or "user upload", "description": "User-supplied dataset.", "preset_mapping": {}}
    if kind == "online_retail_ii":
        df, log = adapters.clean_online_retail(df)
        history += [{"time": R.now(), **l} for l in log]
        meta.update({k: adapters.INFO[kind][k] for k in ("source", "source_url", "description")})
        meta["preset_mapping"] = adapters.PRESET[kind]
    elif kind == "amazon_reviews":
        cat = (category or "unspecified")[:60]
        df, log, note = adapters.clean_amazon(df, cat)
        history += [{"time": R.now(), **l} for l in log]
        meta.update({k: adapters.INFO[kind][k] for k in ("source", "source_url", "description")})
        meta["sample_note"] = note
        meta["category"] = cat
        meta["preset_mapping"] = adapters.PRESET[kind]
    elif source:
        meta["source"] = source
    df, cn = coerce_numeric_text(df)
    history += [{"time": R.now(), "step": "cleaning", "detail": n} for n in cn]
    did = R.new_id("ds")
    m = R.create(did, meta, df, history)
    return {"ok": True, "dataset_id": did, "meta": m, "adapter": kind or "generic", "notes": notes}
