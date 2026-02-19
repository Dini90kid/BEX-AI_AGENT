# ui_streamlit.py — v2 stacked multi-tool UI
# Tools:
#   1) BEx conversion agent (uses agent.py)
#   2) Function Module conversion (ABAP FM -> JSON + Docs + Python stubs/tests)
#   3) Analyse data (CSV) with profiling + reconciliation (diff)
#
# Prompts: Each module has an "Advanced prompts" expander that lets you
#          capture business rules/notes; these are embedded into generated docs.

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import io, zipfile, tempfile, re, json, textwrap

# Import your existing BEx agent
import agent
# --- ADD these helpers near other helpers (top of file is fine) ---
import os, zipfile

def _extract_zip_to_tmp(uploaded_zip):
    """Extract an uploaded ZIP into a temp folder and return Path."""
    tmp = Path(tempfile.mkdtemp())
    zf = zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue()))
    zf.extractall(tmp)
    return tmp

def _iter_files(root: Path, exts: tuple[str, ...]) -> list[Path]:
    """Walk a folder and return every file whose suffix is in exts (case‑insensitive)."""
    out = []
    lower_exts = tuple(e.lower() for e in exts)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in lower_exts:
                out.append(Path(dirpath) / fn)
    return out

st.set_page_config(page_title="BEx / FM / Data Suite", page_icon="🧭", layout="wide")

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def zip_dir_in_memory(folder_path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in folder_path.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(folder_path))
    buf.seek(0)
    return buf.read()

def zip_named_files(files: dict[str, bytes]) -> bytes:
    """files: {"path/in/zip.ext": b"..."}"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    buf.seek(0)
    return buf.read()

def md_block(title: str, body: str) -> str:
    return f"# {title}\n\n{body}\n"

# ---------------------------------------------------------------------------
# 1) BEx — sub-tasks & logic
# ---------------------------------------------------------------------------
def build_pyspark_from_spec(spec: dict, business_rules: str = "") -> str:
    """Generate a basic PySpark runner from your spec (Databricks-friendly)."""
    # Extract known InfoObjects if present
    chars = [c for c in spec.get("characteristics", [])]
    kfs   = spec.get("key_figures", [])
    qname = spec.get("query_name") or "BEX_QUERY"
    code = f'''# AUTOGEN PySpark from BEx spec: {qname}
from pyspark.sql import functions as F

# ---- CONFIG (adjust column names to your landed schema) ----
SOURCE_TABLE = "silver.bw_placeholder"  # TODO: change

CHAR_COLS = { [ (c.get("infoobject") or f"CH_{c.get('tech_id')}") for c in chars ] }
KF_COLS   = { kfs }

df = spark.table(SOURCE_TABLE)

# Example: simple projection on known columns (adjust for real names)
sel_cols = [F.col(c) for c in CHAR_COLS if c] + [F.col(k) for k in KF_COLS if k]
df = df.select(*sel_cols)

# Optional basic groupBy on first char (change as needed)
if CHAR_COLS:
    out = df.groupBy(CHAR_COLS[0]).agg(*[F.sum(k).alias(k) for k in KF_COLS])
else:
    out = df

# Save or display
# out.write.format("delta").mode("overwrite").save("/mnt/output/bex_{qname.lower()}")

display(out)
'''
    if business_rules:
        code = "# BUSINESS RULES / NOTES:\n# " + "\n# ".join(business_rules.splitlines()) + "\n\n" + code
    return code

def page_bex():
    st.header("🧠 BEx conversion agent")
    st.caption("Upload GP .txt files → JSON spec + docs + test data (+ optional PySpark code).")

    subtask = st.selectbox("Choose BEx sub-task", [
        "Standard (Spec + Docs + Test data)",
        "Standard + Databricks PySpark code",
        "Spec only",
        "Docs only",
        "Test data only"
    ])

    with st.expander("Advanced prompts (optional)", expanded=False):
        bex_prompts = st.text_area(
            "Business rules / mapping notes / variable hints",
            placeholder="e.g., Use EUR as default, 0CALMONTH format = yyyymm, treat NCOST_COD as mandatory, etc.",
            height=120
        )

    uploads = st.file_uploader("Upload one or more GP .txt files", type=["txt"], accept_multiple_files=True)
    run = st.button("🚀 Run BEx agent", type="primary")

    if run:
        if not uploads:
            st.error("Please upload at least one .txt GP file.")
            return

        tmp = Path(tempfile.mkdtemp())
        in_dir = tmp / "input"
        out_dir = in_dir / "_agent_output"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in uploads:
            (in_dir / f.name).write_bytes(f.getvalue())

        overrides = agent.load_overrides(in_dir)
        gp_files = [p for p in in_dir.iterdir() if p.suffix.lower() == ".txt"]

        bundle = {}
        logs = []

        for p in gp_files:
            try:
                agent.process_gp_file(p, overrides, out_dir)
                logs.append(f"✅ Processed: {p.name}")

                # Collect spec (if exists) to optionally generate code
                # Find the *first* *_spec.json produced for this p
                produced = list(out_dir.glob(f"{p.stem}*_spec.json"))
                # If no matching by stem, include all specs
                if not produced:
                    produced = list(out_dir.glob("*_spec.json"))

                if "Databricks PySpark code" in subtask and produced:
                    # Use the first spec; (you can extend to all)
                    spec_json = json.loads(produced[0].read_text(encoding="utf-8"))
                    pys = build_pyspark_from_spec(spec_json, bex_prompts or "")
                    code_name = f"pyspark/{spec_json.get('query_name','bex_query').lower()}_spark.py"
                    bundle[code_name] = pys.encode()

            except Exception as e:
                logs.append(f"⚠️ Skipped {p.name}: {e}")

        # bundle existing outputs from out_dir
        for f in out_dir.glob("*"):
            bundle[f.name] = f.read_bytes()

        # If user picked Spec/Docs/Test only: prune bundle accordingly
        def keep(kind):
            if kind == "Spec only":
                return [k for k in bundle if k.endswith("_spec.json")]
            if kind == "Docs only":
                return [k for k in bundle if k.endswith("_documentation.md")]
            if kind == "Test data only":
                return [k for k in bundle if k.endswith("_testdata.csv")]
            return list(bundle.keys())

        chosen = keep(subtask)
        zip_bytes = zip_named_files({k: bundle[k] for k in chosen})

        st.success("BEx run completed.")
        st.code("\n".join(logs), language="text")
        st.download_button("📦 Download BEx outputs (ZIP)", data=zip_bytes,
                           file_name="bex_outputs.zip", mime="application/zip")

# ---------------------------------------------------------------------------
# 2) Function Module — sub-tasks & logic
# ---------------------------------------------------------------------------
FM_SAMPLE = """FUNCTION Z_FM_EXAMPLE.
*"----------------------------------------------------------------------
*"*"Local Interface:
*"  IMPORTING
*"     VALUE(I_MATNR) TYPE MATNR OPTIONAL
*"     VALUE(I_PLANT) TYPE WERKS_D
*"  EXPORTING
*"     VALUE(E_FLAG)  TYPE FLAG
*"  CHANGING
*"     VALUE(C_QTY)   TYPE MENGE_D
*"  TABLES
*"     T_ITEMS STRUCTURE ZITEMS
*"  EXCEPTIONS
*"     NOT_FOUND
*"     FAILED
*"----------------------------------------------------------------------
ENDFUNCTION.
"""

def parse_abap_function(text: str) -> dict:
    """Lightweight FM parser — extracts interface for documentation/spec generation."""
    lines = [ln.rstrip() for ln in text.replace("\r\n","\n").split("\n")]
    src = "\n".join(lines)

    m_name = re.search(r"^\s*FUNCTION\s+([/\w]+)\.", src, flags=re.IGNORECASE|re.MULTILINE)
    name = m_name.group(1) if m_name else "UNKNOWN_FM"

    def extract_block(block):
        pat = rf"^\s*{block}\b(.*?)(?=^\s*(IMPORTING|EXPORTING|CHANGING|TABLES|EXCEPTIONS|ENDFUNCTION)\b)"
        m = re.search(pat, src, flags=re.IGNORECASE|re.MULTILINE|re.DOTALL)
        return m.group(1).strip() if m else ""

    def parse_params(text_block):
        out = []
        for ln in text_block.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("*"):
                continue
            n = re.search(r"VALUE\(([\w/]+)\)", ln, flags=re.IGNORECASE)
            t = re.search(r"\bTYPE\s+([/\w]+)", ln, flags=re.IGNORECASE)
            opt = bool(re.search(r"\bOPTIONAL\b", ln, flags=re.IGNORECASE))
            if n or t:
                out.append({"name": n.group(1) if n else None, "type": t.group(1) if t else None, "optional": opt, "raw": ln})
        return out

    def parse_tables(text_block):
        out = []
        for ln in text_block.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("*"):
                continue
            m = re.match(r"([\w/]+)\s+(STRUCTURE\s+([/\w]+))?", ln, flags=re.IGNORECASE)
            if m:
                out.append({"table": m.group(1), "structure": m.group(3)})
        return out

    def parse_exceptions(text_block):
        return [ln.split()[0] for ln in text_block.splitlines() if ln.strip() and not ln.strip().startswith("*")]

    spec = {
        "function_name": name,
        "importing": parse_params(extract_block("IMPORTING")),
        "exporting": parse_params(extract_block("EXPORTING")),
        "changing":  parse_params(extract_block("CHANGING")),
        "tables":    parse_tables(extract_block("TABLES")),
        "exceptions": parse_exceptions(extract_block("EXCEPTIONS")),
    }
    return spec

def fm_python_stub(spec: dict, business_rules: str="") -> str:
    fname = spec.get("function_name","Z_FM")
    def section(names):
        items = [p.get("name") for p in spec.get(names, []) if p.get("name")]
        return ", ".join([f"{x.lower()}=None" for x in items]) if items else ""
    sig = ", ".join([s for s in [section("importing"), section("changing")] if s])  # tables not in signature for stub
    rules = ""
    if business_rules:
        rules = "# BUSINESS RULES / NOTES:\n" + "\n".join([f"# {ln}" for ln in business_rules.splitlines()]) + "\n\n"
    return rules + f'''def {fname.lower()}({sig}):
    """
    Auto-generated stub for ABAP Function Module: {fname}

    IMPORTING: {[p.get("name") for p in spec.get("importing",[])]}
    EXPORTING: {[p.get("name") for p in spec.get("exporting",[])]}
    CHANGING : {[p.get("name") for p in spec.get("changing",[])]}
    TABLES   : {[t.get("table") for t in spec.get("tables",[])]}
    EXCEPTIONS: {spec.get("exceptions", [])}
    """
    # TODO: implement business logic
    result = {{}}
    # Placeholders for export parameters
    {textwrap.indent("\\n".join([f"result['{(p.get('name') or 'E').lower()}'] = None" for p in spec.get('exporting',[])]), '    ')}
    return result
'''

def fm_pytest_skeleton(spec: dict) -> str:
    fname = spec.get("function_name","Z_FM")
    i_args = [ (p.get("name") or "param").lower() for p in spec.get("importing",[]) if p.get("name") ]
    return f'''import pytest
from {fname.lower()}_stub import {fname.lower()}

def test_{fname.lower()}_happy_path():
    # Arrange
    {", ".join([f"{x}=''" for x in i_args])}
    # Act
    out = {fname.lower()}({", ".join(i_args)})
    # Assert
    assert isinstance(out, dict)
'''

def page_fm():
    st.header("🔧 Function Module conversion")
    st.caption("Upload ABAP FM source → JSON spec + docs + Python stub (+ optional pytest).")

    subtask = st.selectbox("Choose FM sub-task", [
        "Spec + Docs + Stub",
        "Spec + Docs + Stub + Unit test",
        "Spec only",
        "Docs only",
        "Stub only",
        "Unit test only"
    ])

    with st.expander("Advanced prompts (optional)"):
        fm_prompts = st.text_area("Business rules / mapping notes", height=120, placeholder="Describe conversion hints, mappings, special cases…")

    files = st.file_uploader("Upload one or more FM files (.txt/.abap)", type=["txt","abap"], accept_multiple_files=True)
    run = st.button("⚙️ Convert FM(s)", type="primary")

    if run:
        if not files:
            st.error("Please upload at least one ABAP FM file.")
            return

        bundle = {}
        logs = []

        for f in files:
            src = f.getvalue().decode(errors="ignore")
            spec = parse_abap_function(src)
            name = spec.get("function_name","UNKNOWN_FM")

            # JSON spec
            if subtask in ["Spec + Docs + Stub","Spec + Docs + Stub + Unit test","Spec only"]:
                bundle[f"{name}/{name}_spec.json"] = json.dumps(spec, indent=2).encode()

            # Docs
            if subtask in ["Spec + Docs + Stub","Spec + Docs + Stub + Unit test","Docs only"]:
                md = [f"# Function Module: {name}", ""]
                md += ["## IMPORTING"] + [f"- **{p['name']}** : `{p['type']}` {'(OPTIONAL)' if p['optional'] else ''}" for p in spec.get("importing",[])]
                md += ["","## EXPORTING"] + [f"- **{p['name']}** : `{p['type']}`" for p in spec.get("exporting",[])]
                md += ["","## CHANGING"]  + [f"- **{p['name']}** : `{p['type']}`" for p in spec.get("changing",[])]
                md += ["","## TABLES"]    + [f"- **{t['table']}** STRUCTURE `{t.get('structure')}`" for t in spec.get("tables",[])]
                md += ["","## EXCEPTIONS"]+ [f"- {e}" for e in spec.get("exceptions",[])] if spec.get("exceptions") else ["- (none)"]
                if fm_prompts:
                    md += ["","## Notes", fm_prompts]
                bundle[f"{name}/{name}_documentation.md"] = "\n".join(md).encode()

            # Stub
            if subtask in ["Spec + Docs + Stub","Spec + Docs + Stub + Unit test","Stub only"]:
                stub = fm_python_stub(spec, fm_prompts or "")
                bundle[f"{name}/{name.lower()}_stub.py"] = stub.encode()

            # Unit test
            if subtask in ["Spec + Docs + Stub + Unit test","Unit test only"]:
                test_code = fm_pytest_skeleton(spec).encode()
                bundle[f"{name}/test_{name.lower()}.py"] = test_code

            logs.append(f"✅ Parsed: {name}")

        zip_bytes = zip_named_files(bundle)
        st.success("FM conversion completed.")
        st.code("\n".join(logs), language="text")
        st.download_button("📦 Download FM outputs (ZIP)", data=zip_bytes,
                           file_name="fm_outputs.zip", mime="application/zip")

# ---------------------------------------------------------------------------
# 3) Analyse data — sub-tasks & logic
# ---------------------------------------------------------------------------
def quick_profile(df: pd.DataFrame, dataset: str) -> dict:
    cols = []
    for c in df.columns:
        s = df[c]
        info = {
            "column": c,
            "dtype": str(s.dtype),
            "rows": int(len(s)),
            "nulls": int(s.isna().sum()),
            "non_null": int(s.notna().sum()),
            "distinct": int(s.nunique(dropna=True))
        }
        if pd.api.types.is_numeric_dtype(s):
            valid = s.dropna()
            if not valid.empty:
                info.update({
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                    "mean": float(valid.mean()),
                    "std": float(valid.std(ddof=0)),
                })
        cols.append(info)
    return {"dataset": dataset, "rows": int(len(df)), "columns": int(df.shape[1]), "columns_profile": cols}

def reconcile(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> dict:
    # Returns left_only, right_only, mismatches per key (row-level comparison)
    merge = left.merge(right, on=keys, how="outer", indicator=True, suffixes=("_L","_R"))
    left_only  = merge[merge["_merge"]=="left_only"]
    right_only = merge[merge["_merge"]=="right_only"]
    both = merge[merge["_merge"]=="both"].drop(columns=["_merge"])

    # Find differing non-key columns
    non_keys = [c for c in left.columns if c not in keys]
    diffs = []
    for c in non_keys:
        lc, rc = f"{c}_L", f"{c}_R"
        if lc in both.columns and rc in both.columns:
            d = both[both[lc].astype(str) != both[rc].astype(str)]
            if not d.empty:
                diffs.append((c, d[keys + [lc, rc]]))
    return {"left_only": left_only, "right_only": right_only, "diffs": diffs}

def page_analyse():
    import os, zipfile

    st.header("📊 Analyse data")
    st.caption("Profile CSVs, reconcile two CSVs, or analyse BW flow dependencies from your extractor output.")

    subtask = st.selectbox("Choose Data sub-task", [
        "Profile CSV(s)",
        "Reconcile two CSVs (diff by keys)",
        "BW Dependency (folder/zip)"          # <-- NEW
    ])

    # ---------- helpers specific to the new dependency analysis ----------
    def _extract_zip_to_tmp(uploaded_zip):
        tmp = Path(tempfile.mkdtemp())
        zf = zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue()))
        zf.extractall(tmp)
        return tmp

    def _find_dependency_logs(root: Path):
        """
        Return a list of (use_case_name, log_path) tuples.
        Rules we implement:
          - A 'use case' is each top-level folder under root (or any folder the user points at).
          - Under a use case, look for a folder named 'transformation' (case-insensitive).
          - Inside 'transformation', pick files named 'dependency_log*' or 'dependencies_log*'
            (e.g., 'dependency_log', 'dependencies_log.txt', etc.)
          - If multiple logs exist under a use case, collect them all.
        """
        hits = []
        for dirpath, dirnames, filenames in os.walk(root):
            # detect 'transformation' level
            if os.path.basename(dirpath).lower().startswith("transformation"):
                # use case = the folder two levels up (typically)
                # e.g., <root>/<usecase>/transformation/...
                # fallback: the parent of this 'transformation' folder
                trans_dir = Path(dirpath)
                parent = trans_dir.parent
                usecase = parent.name  # default
                # If there is yet another folder above (some exports nest), keep it simple for now.

                for fn in filenames:
                    low = fn.lower()
                    if low.startswith("dependency_log") or low.startswith("dependencies_log"):
                        hits.append((usecase, trans_dir / fn))
        return hits

    FM_NAME_RE = re.compile(
        r"(?<![A-Z0-9_/])"                # left boundary not a typical name char
        r"(?:/[A-Z0-9_]+/)?[A-Z][A-Z0-9_]{2,}"  # supports /SAPAPO/ prefixes and uppercase names
    )

    def _extract_fms_from_text(text: str) -> set[str]:
        """
        Try to be robust:
          - Pick up lines that mention 'Function module' or 'FUNCTION MODULE' or 'type function module'.
          - Extract words that look like FM names (Z_..., Y_..., standard like RS*, or /SAPAPO/*).
        """
        fms = set()
        for ln in text.splitlines():
            if re.search(r"\b(function\s*module|fm:|type\s*function\s*module)\b", ln, flags=re.I):
                # extract candidates
                for cand in FM_NAME_RE.findall(ln):
                    # filter obvious noise tokens
                    if cand.upper() in {"FUNCTION", "MODULE", "TYPE", "VALUE"}:
                        continue
                    fms.add(cand.upper())
        # fallback: if no lines match the phrase, still try to parse all-cap tokens heuristically
        if not fms:
            for cand in FM_NAME_RE.findall(text):
                # keep only custom/standard FM-like tokens (heuristic)
                if len(cand) >= 4:
                    fms.add(cand.upper())
        return fms

    def _analyse_dependencies(root: Path, user_notes: str = ""):
        """
        Crawl the tree, collect per-usecase FMs, invert mapping, and build artifacts.
        Returns: dict with dataframes and byte-encoded files for zipping.
        """
        logs = _find_dependency_logs(root)
        if not logs:
            return {"error": "No dependency_log / dependencies_log files were found under any 'transformation' folder."}

        # maps
        usecase_to_fms: dict[str, set[str]] = {}
        fm_to_usecases: dict[str, set[str]] = {}

        for usecase, log_path in logs:
            try:
                txt = Path(log_path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                txt = Path(log_path).read_text(encoding="latin-1", errors="ignore")

            fms = _extract_fms_from_text(txt)
            if fms:
                usecase_to_fms.setdefault(usecase, set()).update(fms)
                for fm in fms:
                    fm_to_usecases.setdefault(fm, set()).add(usecase)

        # turn into dataframes
        rows_uc = []
        for uc, fms in sorted(usecase_to_fms.items()):
            for fm in sorted(fms):
                rows_uc.append({"use_case": uc, "function_module": fm})
        df_uc = pd.DataFrame(rows_uc).sort_values(["use_case","function_module"]) if rows_uc else pd.DataFrame(columns=["use_case","function_module"])

        rows_fm = []
        for fm, ucs in sorted(fm_to_usecases.items()):
            for uc in sorted(ucs):
                rows_fm.append({"function_module": fm, "use_case": uc})
        df_fm = pd.DataFrame(rows_fm).sort_values(["function_module","use_case"]) if rows_fm else pd.DataFrame(columns=["function_module","use_case"])

        unique_fm_count = len(fm_to_usecases)
        summary_md = []
        summary_md.append("# BW Dependency Analysis")
        summary_md.append("")
        summary_md.append(f"- **Use cases scanned:** {len(usecase_to_fms)}")
        summary_md.append(f"- **Unique Function Modules (FMs):** {unique_fm_count}")
        # top UC by FM count
        if not df_uc.empty:
            top_uc = df_uc.groupby("use_case")["function_module"].nunique().sort_values(ascending=False)
            summary_md.append("")
            summary_md.append("## Top use cases by unique FM count")
            for uc, cnt in top_uc.head(20).items():
                summary_md.append(f"- {uc}: {cnt}")
        # top FM by number of use cases
        if not df_fm.empty:
            top_fm = df_fm.groupby("function_module")["use_case"].nunique().sort_values(ascending=False)
            summary_md.append("")
            summary_md.append("## Top function modules by number of use cases")
            for fm, cnt in top_fm.head(20).items():
                summary_md.append(f"- {fm}: {cnt}")
        if user_notes:
            summary_md.append("")
            summary_md.append("## Notes")
            summary_md.append(user_notes)

        # bundle files
        zip_files = {}
        zip_files["summary/overview.md"] = "\n".join(summary_md).encode()
        if not df_uc.empty:
            zip_files["tables/usecase_to_fms.csv"] = df_uc.to_csv(index=False).encode()
        if not df_fm.empty:
            zip_files["tables/fm_to_usecases.csv"] = df_fm.to_csv(index=False).encode()

        return {
            "df_uc": df_uc,
            "df_fm": df_fm,
            "unique_fm_count": unique_fm_count,
            "zip_bytes": zip_named_files(zip_files)
        }
    # ---------- end helpers ----------

    with st.expander("Advanced prompts / notes (optional)"):
        notes = st.text_area("Add any notes you want embedded into the generated report", height=120)

    # -------- Subtask A: Profile CSV(s) --------
    if subtask == "Profile CSV(s)":
        files = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)
        run = st.button("🔎 Profile", type="primary")
        if run:
            if not files:
                st.error("Please upload at least one CSV.")
                return
            outputs = {}
            tabs = st.tabs([f.name for f in files])
            for i,f in enumerate(files):
                df = pd.read_csv(io.BytesIO(f.getvalue()))
                rpt = quick_profile(df, f.name)
                with tabs[i]:
                    st.subheader(f"Preview — {f.name}")
                    st.dataframe(df.head(200))
                    nulls = {c["column"]: c["nulls"] for c in rpt["columns_profile"]}
                    st.caption("Nulls per column")
                    st.bar_chart(pd.Series(nulls))
                outputs[f"{f.name}/{Path(f.name).stem}_profile.json"] = json.dumps(rpt, indent=2).encode()
                md = ["# Data Profile", f"**File:** {f.name}", f"**Rows:** {rpt['rows']}", f"**Columns:** {rpt['columns']}", ""]
                if notes: md += ["## Notes", notes, ""]
                md += ["## Columns"]
                for c in rpt["columns_profile"]:
                    row = f"- **{c['column']}** (`{c['dtype']}`) non_null={c['non_null']}, nulls={c['nulls']}, distinct={c['distinct']}"
                    for k in ["min","max","mean","std"]:
                        if k in c:
                            row += f", {k}={c[k]}"
                    md.append(row)
                outputs[f"{f.name}/{Path(f.name).stem}_profile.md"] = "\n".join(md).encode()
            st.success("Profiling completed.")
            st.download_button("📦 Download all profiles (ZIP)", data=zip_named_files(outputs),
                               file_name="data_profiles.zip", mime="application/zip")

    # -------- Subtask B: Reconcile two CSVs --------
    elif subtask == "Reconcile two CSVs (diff by keys)":
        left = st.file_uploader("Left CSV", type=["csv"])
        right = st.file_uploader("Right CSV", type=["csv"])
        key_line = st.text_input("Join keys (comma-separated)", value="")
        run = st.button("🔁 Reconcile", type="primary")
        if run:
            if not (left and right and key_line.strip()):
                st.error("Upload two CSVs and provide join keys.")
                return
            keys = [k.strip() for k in key_line.split(",") if k.strip()]
            dfL = pd.read_csv(io.BytesIO(left.getvalue()))
            dfR = pd.read_csv(io.BytesIO(right.getvalue()))
            missing_keys = [k for k in keys if k not in dfL.columns or k not in dfR.columns]
            if missing_keys:
                st.error(f"Keys not found in both files: {missing_keys}")
                return
            out = reconcile(dfL, dfR, keys)
            st.success("Reconciliation completed.")
            st.subheader("Summary")
            st.write(f"Left only rows: {len(out['left_only'])}")
            st.write(f"Right only rows: {len(out['right_only'])}")
            st.write(f"Columns with differences: {len(out['diffs'])}")
            files = {}
            files["summary/notes.md"] = (notes or "").encode()
            if len(out["left_only"])>0:
                files["summary/left_only.csv"] = out["left_only"].to_csv(index=False).encode()
            if len(out["right_only"])>0:
                files["summary/right_only.csv"] = out["right_only"].to_csv(index=False).encode()
            for col, df in out["diffs"]:
                files[f"diffs/{col}_mismatch.csv"] = df.to_csv(index=False).encode()
            st.download_button("📦 Download reconciliation ZIP", data=zip_named_files(files),
                               file_name="reconcile.zip", mime="application/zip")

    # -------- Subtask C: BW Dependency (folder/zip) --------
    else:
        st.info("📁 This finds **transformation/dependency_log** (or **dependencies_log**) under each use-case folder and analyses which Function Modules are used where.")
        mode = st.radio("Choose input mode", ["Upload ZIP (works on Streamlit Cloud)", "Local folder path (run locally)"])
        zip_up = None
        root_path = None

        if mode == "Upload ZIP (works on Streamlit Cloud)":
            zip_up = st.file_uploader("Upload a ZIP of your extractor output root", type=["zip"])
        else:
            root_path = st.text_input("Full local path to extractor output root (only for local runs)", value="", placeholder=r"C:\temp\bw_export_root")

        run = st.button("🧩 Analyse dependencies", type="primary")
        if run:
            # resolve root folder
            if zip_up:
                root = _extract_zip_to_tmp(zip_up)
            else:
                if not root_path:
                    st.error("Provide a local path or upload a ZIP.")
                    return
                root = Path(root_path)
                if not root.exists():
                    st.error(f"Path not found: {root}")
                    return

            result = _analyse_dependencies(root, notes or "")
            if "error" in result:
                st.error(result["error"])
                return

            # on-screen views
            st.success(f"Unique Function Modules (FMs) across all use cases: **{result['unique_fm_count']}**")
            tabs = st.tabs(["Use case → FMs", "FM → Use cases"])
            with tabs[0]:
                if result["df_uc"].empty:
                    st.write("No FM references were found.")
                else:
                    # Top use-cases by FM count
                    top = result["df_uc"].groupby("use_case")["function_module"].nunique().sort_values(ascending=False).head(20)
                    st.caption("Top use cases by unique FM count")
                    st.bar_chart(top)
                    st.dataframe(result["df_uc"])
            with tabs[1]:
                if result["df_fm"].empty:
                    st.write("No FM references were found.")
                else:
                    # Top FMs by number of use cases
                    top = result["df_fm"].groupby("function_module")["use_case"].nunique().sort_values(ascending=False).head(20)
                    st.caption("Top function modules by number of use cases")
                    st.bar_chart(top)
                    st.dataframe(result["df_fm"])

            # download
            st.download_button("📦 Download dependency bundle (ZIP)",
                               data=result["zip_bytes"],
                               file_name="bw_dependency_analysis.zip",
                               mime="application/zip")
# ---------------------------------------------------------------------------
# Sidebar router
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Select tool")
    tool = st.radio("Module", ["BEx conversion agent", "Function Module conversion", "Analyse data"], index=0)
    st.caption("Each module comes with sub-options and advanced prompts.")

if tool == "BEx conversion agent":
    page_bex()
elif tool == "Function Module conversion":
    page_fm()
else:
    page_analyse()
