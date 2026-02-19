# ui_streamlit.py
# 3-in-1 Streamlit UI:
#  1) BEx conversion agent (uses agent.py)
#  2) Function Module conversion (ABAP FM -> JSON spec + Markdown + Python stub)
#  3) Analyse data (CSV profiling)
#
# Deps: streamlit, pandas, numpy

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import io, zipfile, tempfile, re, json
import textwrap

# --- Import your existing BEx agent module (agent.py is in same repo root) ---
import agent

st.set_page_config(page_title="BEx / FM / Data Tools", page_icon="🧭", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _zip_dir_in_memory(folder_path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in folder_path.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(folder_path))
    buf.seek(0)
    return buf.read()

def _zip_named_files(files: dict[str, bytes]) -> bytes:
    """files: {"path/inside/zip.ext": b"..."}"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, data in files.items():
            zf.writestr(arcname, data)
    buf.seek(0)
    return buf.read()

# ----------------------------------------------------------------------------
# 1) PAGE: BEx conversion agent (what you already had)
# ----------------------------------------------------------------------------
def page_bex_agent():
    st.title("🧠 BEx GP Agent – Streamlit UI")
    st.write("Upload BEx GP `.txt` files and generate **JSON spec + documentation + test data**.")

    uploads = st.file_uploader("Upload one or more GP .txt files", type=["txt"], accept_multiple_files=True)
    run = st.button("🚀 Run Agent")

    if run:
        if not uploads:
            st.error("Please upload at least one .txt GP file.")
            return

        tmp_root = Path(tempfile.mkdtemp())
        in_dir = tmp_root / "input"
        out_dir = in_dir / "_agent_output"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save incoming files
        for f in uploads:
            (in_dir / f.name).write_bytes(f.getvalue())

        overrides = agent.load_overrides(in_dir)
        gp_files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]

        logs = []
        for p in gp_files:
            try:
                agent.process_gp_file(p, overrides, out_dir)
                logs.append(f"✅ Processed: {p.name}")
            except Exception as e:
                logs.append(f"⚠️ Skipped {p.name}: {e}")

        st.success("Agent run completed.")
        st.code("\n".join(logs), language="text")

        zip_bytes = _zip_dir_in_memory(out_dir)
        st.download_button("📦 Download BEx outputs (ZIP)", data=zip_bytes,
                           file_name="bex_agent_output.zip", mime="application/zip")

# ----------------------------------------------------------------------------
# 2) PAGE: Function Module conversion
#    Input: ABAP FM source (.txt or .abap)
#    Output: JSON spec, Markdown doc, Python stub
# ----------------------------------------------------------------------------
FM_EXAMPLE = """FUNCTION Z_FM_EXAMPLE.
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
*  "Your logic here"
ENDFUNCTION.
"""

def parse_abap_function(text: str) -> dict:
    """Very lightweight ABAP FM parser; enough for documentation/spec.
       Extracts: name, importing/exporting/changing/tables/exceptions."""
    # Normalize
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    src = "\n".join(lines)

    # FUNCTION name
    m_name = re.search(r"^\s*FUNCTION\s+([/\w]+)\.", src, flags=re.IGNORECASE | re.MULTILINE)
    name = m_name.group(1) if m_name else "UNKNOWN_FM"

    # Block extractor
    def extract_block(block_name):
        # From block header until next block header or ENDFUNCTION.
        pat = rf"^\s*{block_name}\b(.*?)(?=^\s*(IMPORTING|EXPORTING|CHANGING|TABLES|EXCEPTIONS|ENDFUNCTION)\b)"
        m = re.search(pat, src, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        return m.group(1).strip() if m else ""

    def parse_params(block_text):
        """Parse lines like: VALUE(I_MATNR) TYPE MATNR OPTIONAL"""
        out = []
        for ln in block_text.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("*"):
                continue
            # Remove leading '*' quotes area and quotes
            ln_norm = re.sub(r'^\*+"?','',ln).strip()
            # VALUE(name)
            m = re.search(r"VALUE\(([\w/]+)\)", ln_norm, flags=re.IGNORECASE)
            p_name = m.group(1) if m else None
            # TYPE xxx (optional)
            m2 = re.search(r"\bTYPE\s+([/\w]+)", ln_norm, flags=re.IGNORECASE)
            p_type = m2.group(1) if m2 else None
            optional = bool(re.search(r"\bOPTIONAL\b", ln_norm, flags=re.IGNORECASE))
            if p_name or p_type:
                out.append({"name": p_name, "type": p_type, "optional": optional, "raw": ln})
        return out

    def parse_tables(block_text):
        out = []
        for ln in block_text.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("*"):
                continue
            # Example: T_ITEMS STRUCTURE ZITEMS
            m = re.match(r"([\w/]+)\s+(STRUCTURE\s+([/\w]+))?", ln, flags=re.IGNORECASE)
            if m:
                out.append({"table": m.group(1), "structure": m.group(3)})
        return out

    def parse_exceptions(block_text):
        out = []
        for ln in block_text.splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("*"):
                out.append(ln.split()[0])
        return out

    bl_import  = extract_block("IMPORTING")
    bl_export  = extract_block("EXPORTING")
    bl_change  = extract_block("CHANGING")
    bl_tables  = extract_block("TABLES")
    bl_exc     = extract_block("EXCEPTIONS")

    spec = {
        "function_name": name,
        "importing": parse_params(bl_import),
        "exporting": parse_params(bl_export),
        "changing":  parse_params(bl_change),
        "tables":    parse_tables(bl_tables),
        "exceptions": parse_exceptions(bl_exc),
    }
    return spec

def fm_python_stub(spec: dict) -> str:
    """Builds a minimal Python stub mirroring parameters."""
    fname = spec.get("function_name","Z_FM")
    def fmt_params(section):
        # build kwargs signature: i_matnr=None, i_plant=None, ...
        items = [p.get("name") for p in spec.get(section, []) if p.get("name")]
        return ", ".join([f"{x.lower()}=None" for x in items]) if items else ""
    sig_parts = [fmt_params("importing"), fmt_params("changing"), fmt_params("tables")]
    sig = ", ".join([s for s in sig_parts if s])
    body = f'''def {fname.lower()}({sig}):
    """
    Auto-generated stub for ABAP Function Module: {fname}

    IMPORTING: {[p.get("name") for p in spec.get("importing",[])]}
    EXPORTING: {[(p.get("name")) for p in spec.get("exporting",[])]}
    CHANGING : {[(p.get("name")) for p in spec.get("changing",[])]}
    TABLES   : {[(t.get("table")) for t in spec.get("tables",[])]}
    EXCEPTIONS: {spec.get("exceptions", [])}
    """
    # TODO: Implement business logic
    result = {{}}
    # Return placeholders for EXPORTING parameters
    {textwrap.indent("\\n".join([f"result['{p.get('name').lower()}'] = None" for p in spec.get('exporting',[]) if p.get('name')]), '    ')}
    return result
'''
    return body

def page_fm_converter():
    st.title("🔧 Function Module conversion")
    st.write("Upload **ABAP Function Module** source (`.txt` or `.abap`). The tool will create a **JSON spec**, **Markdown doc**, and **Python stub** for each file.")
    with st.expander("Need an example FM? Click to paste a sample"):  # small helper
        st.code(FM_EXAMPLE, language="abap")

    files = st.file_uploader("Upload one or more FM files", type=["txt","abap"], accept_multiple_files=True)
    run = st.button("⚙️ Convert FM(s)")

    if run:
        if not files:
            st.error("Please upload at least one ABAP FM file.")
            return

        outputs: dict[str, bytes] = {}
        for f in files:
            src = f.getvalue().decode(errors="ignore")
            spec = parse_abap_function(src)

            # JSON spec
            json_bytes = json.dumps(spec, indent=2).encode()
            outputs[f"{spec['function_name']}/{spec['function_name']}_spec.json"] = json_bytes

            # Markdown documentation
            md = []
            md.append(f"# Function Module: {spec['function_name']}")
            md.append("## Parameters")
            def sec(title, arr):
                md.append(f"### {title}")
                if not arr: md.append("- (none)")
                else:
                    for p in arr:
                        if "name" in p:
                            md.append(f"- **{p.get('name')}**  — TYPE: `{p.get('type')}`  {'(OPTIONAL)' if p.get('optional') else ''}")
                        else:
                            md.append(f"- {p}")
            sec("IMPORTING", spec.get("importing",[]))
            sec("EXPORTING", spec.get("exporting",[]))
            sec("CHANGING",  spec.get("changing",[]))
            md.append("### TABLES")
            for t in spec.get("tables", []):
                md.append(f"- **{t.get('table')}**  STRUCTURE: `{t.get('structure')}`")
            md.append("### EXCEPTIONS")
            if spec.get("exceptions"):
                for e in spec["exceptions"]:
                    md.append(f"- {e}")
            else:
                md.append("- (none)")

            outputs[f"{spec['function_name']}/{spec['function_name']}_documentation.md"] = "\n".join(md).encode()

            # Python stub
            stub = fm_python_stub(spec).encode()
            outputs[f"{spec['function_name']}/{spec['function_name'].lower()}_stub.py"] = stub

        zip_bytes = _zip_named_files(outputs)
        st.success("FM conversion completed.")
        st.download_button("📦 Download FM outputs (ZIP)", data=zip_bytes,
                           file_name="fm_conversion_output.zip", mime="application/zip")

# ----------------------------------------------------------------------------
# 3) PAGE: Analyse data (CSV quick profiling)
# ----------------------------------------------------------------------------
def profile_dataframe(df: pd.DataFrame, name: str) -> dict:
    # Basic per-column metrics
    cols = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        non_null = int(s.notna().sum())
        nulls = int(s.isna().sum())
        distinct = int(s.nunique(dropna=True))
        sample_values = s.dropna().astype(str).head(5).tolist()
        col_stat = {
            "column": col,
            "dtype": dtype,
            "non_null": non_null,
            "nulls": nulls,
            "distinct": distinct,
            "sample_values": sample_values
        }
        # numeric stats
        if pd.api.types.is_numeric_dtype(s):
            col_stat.update({
                "min": float(np.nanmin(s)) if s.notna().any() else None,
                "max": float(np.nanmax(s)) if s.notna().any() else None,
                "mean": float(np.nanmean(s)) if s.notna().any() else None,
                "std": float(np.nanstd(s)) if s.notna().any() else None,
            })
        cols.append(col_stat)
    report = {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "columns_profile": cols,
    }
    return report

def page_analyse_data():
    st.title("📊 Analyse data (CSV)")
    st.write("Upload one or more **CSV** files. You’ll get a quick **profiling report**, **preview**, and **downloadable summary**.")

    csvs = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    run = st.button("🔎 Analyse")

    if run:
        if not csvs:
            st.error("Please upload at least one CSV file.")
            return

        all_reports = {}
        preview_tabs = st.tabs([f.name for f in csvs])

        zip_files = {}
        for i, f in enumerate(csvs):
            df = pd.read_csv(io.BytesIO(f.getvalue()))
            rep = profile_dataframe(df, f.name)
            all_reports[f.name] = rep

            # Preview (limit to 100 rows to keep UI snappy)
            with preview_tabs[i]:
                st.subheader(f"Preview — {f.name}")
                st.dataframe(df.head(100))
                # Nulls per column bar
                nn = pd.Series({c["column"]: c["nulls"] for c in rep["columns_profile"]})
                st.caption("Nulls per column")
                st.bar_chart(nn)

            # Per-file JSON and Markdown
            json_bytes = json.dumps(rep, indent=2).encode()
            zip_files[f"{f.name}/{Path(f.name).stem}_profile.json"] = json_bytes

            md = ["# Data Profile", f"**File:** {f.name}", f"**Rows:** {rep['rows']}", f"**Columns:** {rep['columns']}", ""]
            md.append("## Columns")
            for c in rep["columns_profile"]:
                md.append(f"- **{c['column']}**  (`{c['dtype']}`) — non_null={c['non_null']}, nulls={c['nulls']}, distinct={c['distinct']}")
                # include numeric stats if present
                for k in ["min","max","mean","std"]:
                    if c.get(k) is not None:
                        md.append(f"  - {k}: {c[k]}")
                if c["sample_values"]:
                    md.append(f"  - sample: {', '.join(c['sample_values'])}")
            zip_files[f"{f.name}/{Path(f.name).stem}_profile.md"] = "\n".join(md).encode()

        zip_bytes = _zip_named_files(zip_files)
        st.success("Analysis completed.")
        st.download_button("📦 Download all profiles (ZIP)", data=zip_bytes,
                           file_name="data_profiles.zip", mime="application/zip")

# ----------------------------------------------------------------------------
# Sidebar router
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Select tool")
    choice = st.radio(
        "Pick a module",
        ["BEx conversion agent", "Function Module conversion", "Analyse data"],
        index=0,
    )
    st.caption("All tools run in your browser. Files are processed in memory.")

if choice == "BEx conversion agent":
    page_bex_agent()
elif choice == "Function Module conversion":
    page_fm_converter()
else:
    page_analyse_data()
