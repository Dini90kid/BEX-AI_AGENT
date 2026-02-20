import streamlit as st
import os
from pathlib import Path
import json
import tempfile
import pandas as pd

# Import your utilities and agent
import agent
from utils import (
    zip_named_files, extract_zip_to_tmp, iter_files,
    build_pyspark_from_spec,
    parse_abap_function, fm_python_stub, fm_pytest_skeleton,
    quick_profile, reconcile,
    analyse_dependencies, extract_fms_from_text
)

st.set_page_config(page_title="BEx / FM / Data Suite", layout="wide")

st.title("🧭 BEx / FM / Data Suite")
st.write("Select what you want to do:")

tool = st.radio(
    "Choose a tool:",
    [
        "BEx Conversion",
        "Function Module Conversion",
        "Analyse Data (CSV / Reconcile)",
        "BW Dependency Analysis"
    ]
)

# ----------------------------------------------------------------------------------
# BEx
# ----------------------------------------------------------------------------------
if tool == "BEx Conversion":
    st.header("🧠 BEx Conversion")
    
    mode = st.selectbox("Input mode", [
        "Upload GP .txt",
        "Upload ZIP (folder of GP files)",
        "Local folder path"
    ])

    with st.expander("Advanced prompts (optional)"):
        bex_notes = st.text_area("Business rules / notes", height=120)

    uploads = None
    zip_up = None
    folder = None

    if mode == "Upload GP .txt":
        uploads = st.file_uploader("Upload GP files", type=["txt"], accept_multiple_files=True)
    elif mode == "Upload ZIP (folder of GP files)":
        zip_up = st.file_uploader("Upload ZIP", type=["zip"])
    else:
        folder = st.text_input("Local folder path", "")

    run = st.button("Run BEx Conversion")
    if run:
        tmp = Path(tempfile.mkdtemp())
        in_dir = tmp / "in"; out_dir = tmp / "out"
        in_dir.mkdir(); out_dir.mkdir()

        gp_paths = []

        if uploads:
            for f in uploads:
                (in_dir / f.name).write_bytes(f.getvalue())
            gp_paths = [p for p in in_dir.iterdir() if p.suffix.lower()==".txt"]

        elif zip_up:
            root = extract_zip_to_tmp(zip_up)
            gp_paths = iter_files(root, (".txt",))

        else:
            root = Path(folder)
            gp_paths = iter_files(root, (".txt",))

        bundle = {}

        overrides = agent.load_overrides(in_dir)
        logs = []

        for p in gp_paths:
            dst = in_dir / p.name
            if not p.exists():
                pass
            else:
                try:
                    dst.write_bytes(p.read_bytes())
                except:
                    pass

            agent.process_gp_file(dst, overrides, out_dir)
            logs.append(f"Processed {p.name}")

            # Optional PySpark
            spec_json_files = list(out_dir.glob("*_spec.json"))
            if spec_json_files:
                spec = json.loads(spec_json_files[0].read_text())
                pys = build_pyspark_from_spec(spec, bex_notes)
                bundle[f"{spec.get('query_name','bex').lower()}_spark.py"] = pys.encode()

        for f in out_dir.glob("*"):
            bundle[f.name] = f.read_bytes()

        st.success("BEx Conversion Completed")
        st.code("\n".join(logs))

        st.download_button(
            "Download Results (ZIP)",
            data=zip_named_files(bundle),
            file_name="bex_output.zip",
            mime="application/zip"
        )

# ----------------------------------------------------------------------------------
# Function Modules
# ----------------------------------------------------------------------------------
elif tool == "Function Module Conversion":
    st.header("🔧 Function Module Conversion")

    uploads = st.file_uploader("Upload FM files (.abap/.txt)", type=["abap","txt"], accept_multiple_files=True)

    with st.expander("Advanced notes (optional)"):
        fm_notes = st.text_area("Notes", height=150)

    run = st.button("Convert FM(s)")

    if run:
        tmp = Path(tempfile.mkdtemp())
        bundle = {}
        logs = []

        for f in uploads:
            raw = f.getvalue().decode("utf-8", errors="ignore")
            spec = parse_abap_function(raw)
            name = spec["function_name"]

            # Save spec
            bundle[f"{name}/{name}_spec.json"] = json.dumps(spec, indent=2).encode()

            # Docs
            md = [f"# Function Module: {name}", ""]
            md.append("## IMPORTING")
            for p in spec["importing"]:
                md.append(f"- {p['name']} : {p['type']} ({'OPTIONAL' if p['optional'] else ''})")
            md.append("")
            md.append("## EXPORTING")
            for p in spec["exporting"]:
                md.append(f"- {p['name']} : {p['type']}")
            md.append("")
            if fm_notes:
                md += ["## Notes", fm_notes]

            bundle[f"{name}/{name}_documentation.md"] = "\n".join(md).encode()

            # Stub
            stub = fm_python_stub(spec, fm_notes)
            bundle[f"{name}/{name.lower()}_stub.py"] = stub.encode()

            logs.append(f"Processed FM: {name}")

        st.success("FM Conversion Completed")
        st.code("\n".join(logs))

        st.download_button(
            "Download Results (ZIP)",
            data=zip_named_files(bundle),
            file_name="fm_output.zip",
            mime="application/zip"
        )

# ----------------------------------------------------------------------------------
# Analyse Data
# ----------------------------------------------------------------------------------
elif tool == "Analyse Data (CSV / Reconcile)":
    st.header("📊 Analyse Data")

    sub = st.radio("Choose operation:", [
        "Profile CSV(s)",
        "Reconcile CSVs"
    ])

    with st.expander("Notes"):
        notes = st.text_area("Notes", height=100)

    if sub == "Profile CSV(s)":
        uploads = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
        run = st.button("Run Profiling")
        if run:
            bundle = {}
            for f in uploads:
                df = pd.read_csv(f)
                rpt = quick_profile(df, f.name)
                bundle[f"{f.name}/{f.name}_profile.json"] = json.dumps(rpt, indent=2).encode()
            st.success("Profile completed")
            st.download_button(
                "Download Profiles ZIP",
                data=zip_named_files(bundle),
                file_name="profiles.zip",
                mime="application/zip"
            )

    else:
        left = st.file_uploader("Left CSV", type=["csv"])
        right = st.file_uploader("Right CSV", type=["csv"])
        keys = st.text_input("Join keys (comma separated)")
        run = st.button("Reconcile")
        if run:
            dfL = pd.read_csv(left)
            dfR = pd.read_csv(right)
            key_list = [k.strip() for k in keys.split(",")]
            out = reconcile(dfL, dfR, key_list)
            bundle = {}
            bundle["left_only.csv"] = out["left_only"].to_csv(index=False).encode()
            bundle["right_only.csv"] = out["right_only"].to_csv(index=False).encode()
            for col, diffdf in out["diffs"]:
                bundle[f"diff_{col}.csv"] = diffdf.to_csv(index=False).encode()
            st.success("Reconcile completed")
            st.download_button(
                "Download ZIP",
                data=zip_named_files(bundle),
                file_name="reconcile.zip",
                mime="application/zip"
            )

# ----------------------------------------------------------------------------------
# BW Dependency Analysis
# ----------------------------------------------------------------------------------
else:
    st.header("🧩 BW Dependency Analysis")

    mode = st.selectbox("Input mode", [
        "Upload dependency logs",
        "Upload ZIP",
        "Local folder path"
    ])

    logs_upload = None
    zip_up = None
    folder = None

    if mode == "Upload dependency logs":
        logs_upload = st.file_uploader("Upload dependency_log files", accept_multiple_files=True)
    elif mode == "Upload ZIP":
        zip_up = st.file_uploader("Upload ZIP", type=["zip"])
    else:
        folder = st.text_input("Local folder path", "")

    run = st.button("Run Dependency Analysis")

    if run:
        if logs_upload:
            # Multiple log files
            usecase_to_fms = {}
            fm_to_usecases = {}
            for f in logs_upload:
                raw = f.getvalue().decode("utf-8", errors="ignore")
                ucase = Path(f.name).stem
                fms = extract_fms_from_text(raw)
                for fm in fms:
                    usecase_to_fms.setdefault(ucase, set()).add(fm)
                    fm_to_usecases.setdefault(fm, set()).add(ucase)
            # Build summary
            bundle = {}
            out_md = ["# Dependency Analysis", ""]
            for u, fmlist in usecase_to_fms.items():
                out_md.append(f"## {u}")
                for fm in fmlist:
                    out_md.append(f"- {fm}")
            bundle["dependency.md"] = "\n".join(out_md).encode()

            st.download_button(
                "Download Results (ZIP)",
                data=zip_named_files(bundle),
                file_name="dependency.zip",
                mime="application/zip"
            )
