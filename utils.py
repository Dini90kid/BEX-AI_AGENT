# utils.py — shared helpers for all pages

from pathlib import Path
import io, os, re, json, zipfile, tempfile, textwrap
import pandas as pd
import numpy as np

# ------------------ General helpers ------------------

def zip_named_files(files: dict[str, bytes]) -> bytes:
    """files: {'path/in/zip.ext': b'...'} → returns zip bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    buf.seek(0)
    return buf.read()

def zip_dir_in_memory(folder_path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in folder_path.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(folder_path))
    buf.seek(0)
    return buf.read()

def extract_zip_to_tmp(uploaded_zip) -> Path:
    """Extract uploaded ZIP (Streamlit UploadedFile) to a temp dir."""
    tmp = Path(tempfile.mkdtemp())
    zf = zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue()))
    zf.extractall(tmp)
    return tmp

def iter_files(root: Path, exts: tuple[str, ...]) -> list[Path]:
    """Recursively list files under root with extensions in exts (case‑insensitive)."""
    out = []
    exts = tuple(e.lower() for e in exts)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in exts:
                out.append(Path(dirpath) / fn)
    return out

# ------------------ BEx helpers ------------------

def build_pyspark_from_spec(spec: dict, business_rules: str = "") -> str:
    """Generate a basic PySpark skeleton from a BEx JSON spec."""
    chars = [c for c in spec.get("characteristics", [])]
    kfs   = spec.get("key_figures", [])
    qname = spec.get("query_name") or "BEX_QUERY"

    code = f'''# AUTOGEN PySpark from BEx spec: {qname}
from pyspark.sql import functions as F

SOURCE_TABLE = "silver.bw_placeholder"  # TODO: change

CHAR_COLS = { [ (c.get("infoobject") or f"CH_{c.get('tech_id')}") for c in chars ] }
KF_COLS   = { kfs }

df = spark.table(SOURCE_TABLE)
sel_cols = [F.col(c) for c in CHAR_COLS if c] + [F.col(k) for k in KF_COLS if k]
df = df.select(*sel_cols)

if CHAR_COLS:
    out = df.groupBy(CHAR_COLS[0]).agg(*[F.sum(k).alias(k) for k in KF_COLS])
else:
    out = df

display(out)
'''
    if business_rules:
        code = "# BUSINESS RULES / NOTES:\n# " + "\n# ".join(business_rules.splitlines()) + "\n\n" + code
    return code

# ------------------ FM helpers ------------------

def parse_abap_function(text: str) -> dict:
    """Lightweight ABAP FM interface parser."""
    lines = [ln.rstrip() for ln in text.replace("\r\n","\n").split("\n")]
    src = "\n".join(lines)

    m_name = re.search(r"^\s*FUNCTION\s+([/\w]+)\.", src, flags=re.I|re.M)
    name = m_name.group(1) if m_name else "UNKNOWN_FM"

    def extract_block(block):
        pat = rf"^\s*{block}\b(.*?)(?=^\s*(IMPORTING|EXPORTING|CHANGING|TABLES|EXCEPTIONS|ENDFUNCTION)\b)"
        m = re.search(pat, src, flags=re.I|re.M|re.S)
        return m.group(1).strip() if m else ""

    def parse_params(txt):
        out = []
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("*"): continue
            n = re.search(r"VALUE\(([\w/]+)\)", ln, flags=re.I)
            t = re.search(r"\bTYPE\s+([/\w]+)", ln, flags=re.I)
            opt = bool(re.search(r"\bOPTIONAL\b", ln, flags=re.I))
            if n or t:
                out.append({"name": n.group(1) if n else None, "type": t.group(1) if t else None, "optional": opt, "raw": ln})
        return out

    def parse_tables(txt):
        out = []
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("*"): continue
            m = re.match(r"([\w/]+)\s+(STRUCTURE\s+([/\w]+))?", ln, flags=re.I)
            if m: out.append({"table": m.group(1), "structure": m.group(3)})
        return out

    def parse_exceptions(txt):
        return [ln.split()[0] for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("*")]

    spec = {
        "function_name": name,
        "importing": parse_params(extract_block("IMPORTING")),
        "exporting": parse_params(extract_block("EXPORTING")),
        "changing":  parse_params(extract_block("CHANGING")),
        "tables":    parse_tables(extract_block("TABLES")),
        "exceptions": parse_exceptions(extract_block("EXCEPTIONS")),
    }
    return spec

def fm_python_stub(spec: dict, rules: str="") -> str:
    fname = spec.get("function_name","Z_FM")
    def section(names):
        items = [p.get("name") for p in spec.get(names, []) if p.get("name")]
        return ", ".join([f"{x.lower()}=None" for x in items]) if items else ""
    sig = ", ".join([s for s in [section("importing"), section("changing")] if s])
    header = ""
    if rules:
        header = "# BUSINESS RULES / NOTES:\n" + "\n".join([f"# {ln}" for ln in rules.splitlines()]) + "\n\n"
    return header + f'''def {fname.lower()}({sig}):
    """
    Auto-generated stub for ABAP Function Module: {fname}
    """
    result = {{}}
    {textwrap.indent("\\n".join([f"result['{(p.get('name') or 'E').lower()}'] = None" for p in spec.get('exporting',[])]), '    ')}
    return result
'''

def fm_pytest_skeleton(spec: dict) -> str:
    fname = spec.get("function_name","Z_FM")
    i_args = [ (p.get("name") or "param").lower() for p in spec.get("importing",[]) if p.get("name") ]
    return f'''import pytest
from {fname.lower()}_stub import {fname.lower()}

def test_{fname.lower()}_happy_path():
    {"; ".join([f"{x}=''" for x in i_args])}
    out = {fname.lower()}({", ".join(i_args)})
    assert isinstance(out, dict)
'''

# ------------------ Analyse: profiling & reconcile ------------------

def quick_profile(df: pd.DataFrame, dataset: str) -> dict:
    cols = []
    for c in df.columns:
        s = df[c]
        info = {"column": c, "dtype": str(s.dtype),
                "rows": int(len(s)), "nulls": int(s.isna().sum()),
                "non_null": int(s.notna().sum()), "distinct": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            valid = s.dropna()
            if not valid.empty:
                info.update({"min": float(valid.min()), "max": float(valid.max()),
                             "mean": float(valid.mean()), "std": float(valid.std(ddof=0))})
        cols.append(info)
    return {"dataset": dataset, "rows": int(len(df)), "columns": int(df.shape[1]), "columns_profile": cols}

def reconcile(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> dict:
    merge = left.merge(right, on=keys, how="outer", indicator=True, suffixes=("_L","_R"))
    left_only  = merge[merge["_merge"]=="left_only"]
    right_only = merge[merge["_merge"]=="right_only"]
    both = merge[merge["_merge"]=="both"].drop(columns=["_merge"])
    non_keys = [c for c in left.columns if c not in keys]
    diffs = []
    for c in non_keys:
        lc, rc = f"{c}_L", f"{c}_R"
        if lc in both.columns and rc in both.columns:
            d = both[both[lc].astype(str) != both[rc].astype(str)]
            if not d.empty:
                diffs.append((c, d[keys + [lc, rc]]))
    return {"left_only": left_only, "right_only": right_only, "diffs": diffs}

# ------------------ Analyse: BW dependency ------------------

def find_dependency_logs(root: Path):
    hits = []
    for dirpath, _, filenames in os.walk(root):
        if os.path.basename(dirpath).lower().startswith("transformation"):
            trans_dir = Path(dirpath); usecase = trans_dir.parent.name
            for fn in filenames:
                low = fn.lower()
                if low.startswith("dependency_log") or low.startswith("dependencies_log"):
                    hits.append((usecase, trans_dir / fn))
    return hits

FM_NAME_RE = re.compile(r"(?<![A-Z0-9_/])(?:/[A-Z0-9_]+/)?[A-Z][A-Z0-9_]{2,}")

def extract_fms_from_text(text: str) -> set[str]:
    fms = set()
    for ln in text.splitlines():
        if re.search(r"\b(function\s*module|fm:|type\s*function\s*module)\b", ln, flags=re.I):
            for cand in FM_NAME_RE.findall(ln):
                if cand.upper() in {"FUNCTION","MODULE","TYPE","VALUE"}: continue
                fms.add(cand.upper())
    if not fms:
        for cand in FM_NAME_RE.findall(text):
            if len(cand) >= 4:
                fms.add(cand.upper())
    return fms

def analyse_dependencies(root: Path, notes: str = ""):
    logs = find_dependency_logs(root)
    if not logs:
        return {"error": "No dependency_log / dependencies_log files found under any 'transformation' folder."}

    usecase_to_fms, fm_to_usecases = {}, {}
    for usecase, path in logs:
        try:
            txt = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt = Path(path).read_text(encoding="latin-1", errors="ignore")
        fms = extract_fms_from_text(txt)
        if fms:
            usecase_to_fms.setdefault(usecase, set()).update(fms)
            for fm in fms:
                fm_to_usecases.setdefault(fm, set()).add(usecase)

    rows_uc = [{"use_case": uc, "function_module": fm}
               for uc, fms in sorted(usecase_to_fms.items()) for fm in sorted(fms)]
    df_uc = pd.DataFrame(rows_uc).sort_values(["use_case","function_module"]) if rows_uc else pd.DataFrame(columns=["use_case","function_module"])

    rows_fm = [{"function_module": fm, "use_case": uc}
               for fm, ucs in sorted(fm_to_usecases.items()) for uc in sorted(ucs)]
    df_fm = pd.DataFrame(rows_fm).sort_values(["function_module","use_case"]) if rows_fm else pd.DataFrame(columns=["function_module","use_case"])

    unique_fm_count = len(fm_to_usecases)
    files = {"summary/overview.md": f"# BW Dependency Analysis\n\n- Unique FMs: {unique_fm_count}\n\n## Notes\n{notes or ''}\n".encode()}
    if not df_uc.empty: files["tables/usecase_to_fms.csv"] = df_uc.to_csv(index=False).encode()
    if not df_fm.empty: files["tables/fm_to_usecases.csv"] = df_fm.to_csv(index=False).encode()

    return {"df_uc": df_uc, "df_fm": df_fm, "unique_fm_count": unique_fm_count, "zip_bytes": zip_named_files(files)}
