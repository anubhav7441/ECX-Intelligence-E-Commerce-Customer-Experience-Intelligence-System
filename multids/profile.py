"""Safe file loading (CSV / XLSX / JSONL[.gz]) and dataset quality profiling. No code execution, no pickle."""
import gzip
import io
import json
import re
import pandas as pd

MAX_BYTES = 80 * 1024 * 1024
MAX_SAMPLE_JSONL = 100_000


class LoadError(ValueError):
    pass


def load_table(filename, data, sheet=None):
    """Return (DataFrame, notes[list of str]). Raises LoadError with a beginner-friendly message."""
    name = (filename or "").lower()
    notes = []
    if not data:
        raise LoadError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise LoadError("File is larger than 80 MB.")
    try:
        if name.endswith(".xlsx"):
            xl = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
            names = xl.sheet_names
            if sheet:
                if sheet not in names:
                    raise LoadError(f"Sheet '{sheet}' not found. Sheets: {names}")
                df = xl.parse(sheet)
                notes.append(f"Read sheet '{sheet}'.")
            else:
                frames = [xl.parse(s) for s in names]
                same = all(list(f.columns) == list(frames[0].columns) for f in frames)
                if len(frames) > 1 and same:
                    df = pd.concat(frames, ignore_index=True)
                    notes.append(f"Workbook has {len(names)} sheets with identical columns; they were stacked ({', '.join(names)}).")
                else:
                    df = frames[0]
                    if len(frames) > 1:
                        notes.append(f"Workbook has {len(names)} sheets with different columns; only the first ('{names[0]}') was read.")
        elif name.endswith((".jsonl", ".jsonl.gz", ".json", ".json.gz")):
            raw = gzip.decompress(data) if name.endswith(".gz") else data
            rows = []
            for i, line in enumerate(io.BytesIO(raw)):
                if i >= MAX_SAMPLE_JSONL:
                    notes.append(f"Only the first {MAX_SAMPLE_JSONL:,} JSON lines were read (sample cap).")
                    break
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            df = pd.DataFrame(rows)
        elif name.endswith((".csv", ".tsv", ".txt")):
            df = None
            for enc in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python", on_bad_lines="skip")
                    if enc != "utf-8-sig":
                        notes.append(f"Decoded as {enc}.")
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise LoadError("Could not decode the text file.")
        else:
            raise LoadError("Unsupported file type. Use .csv, .xlsx or .jsonl(.gz).")
    except LoadError:
        raise
    except (pd.errors.EmptyDataError, pd.errors.ParserError, json.JSONDecodeError, ValueError, KeyError, OSError, EOFError) as e:
        raise LoadError(f"Could not read the file: {str(e)[:150]}") from e
    except Exception as e:   # corrupt zip (xlsx) etc.
        raise LoadError(f"Could not read the file ({e.__class__.__name__}). Is it a valid {name.split('.')[-1]} file?") from e
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise LoadError("The file has no data rows or no columns.")
    df.columns = [str(c).strip() for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        seen = {}
        cols = []
        for c in df.columns:
            seen[c] = seen.get(c, 0) + 1
            cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
        df.columns = cols
        notes.append("Duplicate column names were made unique.")
    return df, notes


_NUM_JUNK = re.compile(r"[\s,$\u20b9\u20ac\u00a3\u00a5%]|INR|USD|EUR|Rs\.?", re.I)


def coerce_numeric_text(df):
    """Turn text columns that really hold numbers ('$12.50', '1,200', '15%') into numeric columns.
    Returns (new_df, notes). Only converts when at least 90% of the non-empty values parse."""
    notes, out = [], df
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        v = s.dropna().astype(str)
        if len(v) < 5 or v.str.len().mean() > 20:
            continue
        cleaned = v.str.replace(_NUM_JUNK, "", regex=True)
        conv = pd.to_numeric(cleaned, errors="coerce")
        if conv.notna().mean() >= 0.9 and conv.nunique() > 1:
            if out is df:
                out = df.copy()
            full = pd.to_numeric(s.astype("string").str.replace(_NUM_JUNK, "", regex=True), errors="coerce")
            out[c] = full
            notes.append(f"Column '{c}' was stored as text but holds numbers; it was converted to numeric.")
    return out, notes


def column_kind(s):
    """numeric / date / boolean / id / categorical / text"""
    n = int(s.notna().sum())
    if n == 0:
        return "empty"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    nun = s.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(s):
        return "id" if (nun / n > 0.95 and pd.api.types.is_integer_dtype(s) and n > 50 and s.min() >= 0 and s.std() > 0 and (s.is_monotonic_increasing)) else "numeric"
    sample = s.dropna().astype(str).head(2000)
    if sample.str.match(r"^\d{4}-\d{2}-\d{2}|^\d{1,2}/\d{1,2}/\d{2,4}").mean() > 0.8:
        try:
            pd.to_datetime(sample, errors="raise", format="mixed")
            return "date"
        except Exception:
            pass
    if sample.str.contains(r"\d").mean() > 0.9 and sample.str.len().mean() < 30 and nun > 1:
        try:
            parsed = pd.to_datetime(sample.head(300), errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.9 and parsed.dt.year.between(1900, 2100).mean() > 0.9 and sample.str.contains(r"[-/ .]").mean() > 0.9:
                return "date"
        except Exception:
            pass
    avg_len = sample.str.len().mean()
    if avg_len > 40 or (sample.str.count(" ").mean() >= 4 and nun / n > 0.5):
        return "text"
    if nun / n > 0.9 and n > 50:
        return "id"
    return "categorical"


def profile(df):
    n = len(df)
    cols = []
    for c in df.columns:
        s = df[c]
        k = column_kind(s)
        info = {"name": c, "dtype": str(s.dtype), "kind": k, "missing": int(s.isna().sum()), "missing_pct": round(float(s.isna().mean() * 100), 2),
                "unique": int(s.nunique(dropna=True)), "example": None if s.dropna().empty else str(s.dropna().iloc[0])[:60]}
        if k == "numeric":
            info.update({"min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()), "negatives": int((s < 0).sum()), "zeros": int((s == 0).sum())})
        cols.append(info)
    dups = int(df.duplicated().sum())
    issues = []
    if dups:
        issues.append({"severity": "warning", "issue": f"{dups:,} fully duplicated rows ({dups / n:.1%})."})
    for c in cols:
        if c["missing_pct"] >= 50:
            issues.append({"severity": "warning", "issue": f"Column '{c['name']}' is {c['missing_pct']}% empty."})
        elif c["missing_pct"] > 0 and c["missing_pct"] >= 5:
            issues.append({"severity": "info", "issue": f"Column '{c['name']}' has {c['missing_pct']}% missing values."})
        if c["kind"] == "numeric" and c.get("negatives"):
            issues.append({"severity": "info", "issue": f"Column '{c['name']}' has {c['negatives']:,} negative values (check whether these are returns/cancellations)."})
        if c["unique"] == 1 and n > 1:
            issues.append({"severity": "info", "issue": f"Column '{c['name']}' has a single constant value."})
    if n < 30:
        issues.insert(0, {"severity": "warning", "issue": f"Only {n:,} rows. Most analyses need at least 30 rows, so results will be limited or skipped."})
    if df.shape[1] < 2:
        issues.insert(0, {"severity": "warning", "issue": "Only one column found. Analyses need at least two (for example a date and an amount)."})
    usable = [c for c in cols if c["kind"] not in ("empty",) and c["unique"] > 1]
    if cols and len(usable) < max(1, len(cols) // 2):
        issues.append({"severity": "warning", "issue": "Most columns are empty or constant, so they carry no information."})
    score = 100 - min(40, sum(c["missing_pct"] for c in cols) / max(len(cols), 1)) - min(20, 100 * dups / n)
    return {"n_rows": n, "n_cols": df.shape[1], "duplicate_rows": dups, "columns": cols, "issues": issues,
            "by_kind": {k: [c["name"] for c in cols if c["kind"] == k] for k in ("numeric", "categorical", "date", "text", "id", "boolean", "empty")},
            "quality_score": round(float(score), 1),
            "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 1)}
