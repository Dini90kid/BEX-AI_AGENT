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
