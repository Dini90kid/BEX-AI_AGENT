# pages/1_BEx_Conversion.py
import streamlit as st
from pathlib import Path
import json
import agent
from utils import (zip_named_files, extract_zip_to_tmp, iter_files, build_pyspark_from_spec)

st.title("🧠 BEx conversion")

subtask = st.selectbox("Choose BEx sub-task", [
    "Standard (Spec + Docs + Test data)",
    "Standard + Databricks PySpark code",
    "Spec only", "Docs only", "Test data only"
])

with st.expander("Advanced prompts (optional)"):
    bex_prompts = st.text_area("Business rules / notes", height=120)

mode = st.radio("Input mode", [
    "Upload .txt file(s)",
    "Upload ZIP of a folder",
    "Local folder path (run locally)"
])

files_upload = zip_upload = None
folder_path = None

if mode == "Upload .txt file(s)":
    files_upload = st.file_uploader("Upload GP .txt", type=["txt"], accept_multiple_files=True)
elif mode == "Upload ZIP of a folder":
    zip_upload = st.file_uploader("Upload ZIP containing GP .txt files", type=["zip"])
else:
    folder_path = st.text_input("Local folder path (contains GP .txt files)", value="", placeholder=r"C:\path\to\gp_exports")

run = st.button("🚀 Run BEx agent", type="primary")

if run:
    tmp_root = Path(st.session_state.get("tmp_bex", Path.cwd()))
    in_dir = Path(tempfile.mkdtemp()) / "bex_in"
    out_dir = in_dir / "_agent_output"
    in_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve inputs to list[Path]
    gp_paths = []
    if files_upload:
        for f in files_upload: (in_dir / f.name).write_bytes(f.getvalue())
        gp_paths = [p for p in in_dir.iterdir() if p.suffix.lower()==".txt"]
    elif zip_upload:
        root = extract_zip_to_tmp(zip_upload)
        gp_paths = iter_files(root, (".txt",))
        if not gp_paths: st.error("No .txt found in ZIP."); st.stop()
    else:
        if not folder_path: st.error("Provide a folder path or pick another mode."); st.stop()
        root = Path(folder_path)
        if not root.exists(): st.error(f"Path not found: {root}"); st.stop()
        gp_paths = iter_files(root, (".txt",))
        if not gp_paths: st.error("No .txt files under that folder."); st.stop()

    # Run agent
    bundle = {}; logs = []
    overrides = agent.load_overrides(in_dir if files_upload else (root if folder_path else root))
    for src in gp_paths:
        try:
            # ensure agent writes outputs into out_dir
            if src.parent != in_dir:
                dst = in_dir / src.name; dst.write_bytes(src.read_bytes()); src_for_agent = dst
            else:
                src_for_agent = src

            agent.process_gp_file(src_for_agent, overrides, out_dir)
            logs.append(f"✅ Processed: {src.name}")

            if "Databricks PySpark code" in subtask:
                produced = list(out_dir.glob(f"{src.stem}*_spec.json")) or list(out_dir.glob("*_spec.json"))
                if produced:
                    spec_json = json.loads(produced[0].read_text(encoding="utf-8"))
                    pys = build_pyspark_from_spec(spec_json, bex_prompts or "")
                    code_name = f"pyspark/{spec_json.get('query_name','bex_query').lower()}_spark.py"
                    bundle[code_name] = pys.encode()

        except Exception as e:
            logs.append(f"⚠️ Skipped {src.name}: {e}")

    for f in out_dir.glob("*"): bundle[f.name] = f.read_bytes()

    def keep(kind):
        if kind=="Spec only": return [k for k in bundle if k.endswith("_spec.json")]
        if kind=="Docs only": return [k for k in bundle if k.endswith("_documentation.md")]
        if kind=="Test data only": return [k for k in bundle if k.endswith("_testdata.csv")]
        return list(bundle.keys())

    zip_bytes = zip_named_files({k: bundle[k] for k in keep(subtask)})
    st.success("BEx run completed.")
    st.code("\n".join(logs), language="text")
    st.download_button("📦 Download BEx outputs (ZIP)", data=zip_bytes, file_name="bex_outputs.zip", mime="application/zip")
