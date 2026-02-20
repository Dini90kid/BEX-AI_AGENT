# pages/2_Function_Module_Conversion.py
import streamlit as st
from pathlib import Path
import tempfile
from utils import (zip_named_files, extract_zip_to_tmp, iter_files,
                   parse_abap_function, fm_python_stub, fm_pytest_skeleton)

st.title("🔧 Function Module conversion")

subtask = st.selectbox("Choose FM sub-task", [
    "Spec + Docs + Stub",
    "Spec + Docs + Stub + Unit test",
    "Spec only", "Docs only", "Stub only", "Unit test only"
])

with st.expander("Advanced prompts (optional)"):
    fm_prompts = st.text_area("Business rules / mapping notes", height=120)

mode = st.radio("Input mode", [
    "Upload .abap/.txt file(s)",
    "Upload ZIP of a folder",
    "Local folder path (run locally)"
])

files_upload = zip_upload = None
folder_path = None

if mode == "Upload .abap/.txt file(s)":
    files_upload = st.file_uploader("Upload FM files", type=["abap","txt"], accept_multiple_files=True)
elif mode == "Upload ZIP of a folder":
    zip_upload = st.file_uploader("Upload ZIP containing FM files", type=["zip"])
else:
    folder_path = st.text_input("Local folder path (contains FM .abap/.txt)", value="", placeholder=r"C:\path\to\fm_sources")

run = st.button("⚙️ Convert FM(s)", type="primary")

if run:
    fm_paths = []
    if files_upload:
        in_dir = Path(tempfile.mkdtemp()) / "fm_in"; in_dir.mkdir(parents=True, exist_ok=True)
        for f in files_upload: (in_dir / f.name).write_bytes(f.getvalue())
        fm_paths = [p for p in in_dir.iterdir() if p.suffix.lower() in (".abap",".txt")]
    elif zip_upload:
        root = extract_zip_to_tmp(zip_upload)
        fm_paths = iter_files(root, (".abap",".txt"))
        if not fm_paths: st.error("No .abap/.txt found in ZIP."); st.stop()
    else:
        if not folder_path: st.error("Provide a folder path or pick another mode."); st.stop()
        root = Path(folder_path)
        if not root.exists(): st.error(f"Path not found: {root}"); st.stop()
        fm_paths = iter_files(root, (".abap",".txt"))
        if not fm_paths: st.error("No .abap/.txt files under that folder."); st.stop()

    bundle = {}; logs = []
    for src in fm_paths:
        try:
            raw = src.read_text(encoding="utf-8", errors="ignore")
            spec = parse_abap_function(raw)
            name = spec.get("function_name","UNKNOWN_FM")

            if subtask in ["Spec + Docs + Stub","Spec + Docs + Stub + Unit test","Spec only"]:
                bundle[f"{name}/{name}_spec.json"] = (temp:=__import__('json')).dumps(spec, indent=2).encode()

            if subtask in ["Spec + Docs + Stub","Spec + Docs + Stub + Unit test","Docs only"]:
                md = [f"# Function Module: {name}", ""]
                md += ["## IMPORTING"] + [f"- **{p['name']}** : `{p['type']}` {'(OPTIONAL)' if p['optional'] else ''}" for p in spec.get("importing",[])]
                md += ["","## EXPORTING"] + [f"- **{p['name']}** : `{p['type']}`" for p in spec.get("exporting",[])]
                md += ["","## CHANGING"]  + [f"- **{p['name']}** : `{p['type']}`" for p in spec.get("changing",[])]
                md += ["","## TABLES"]    + [f"- **{t['table']}** STRUCTURE `{t.get('structure')}`" for t in spec.get("tables",[])]
                md += ["","## EXCEPTIONS"]+ ([f"- {e}" for e in spec.get("exceptions",[])] if spec.get("exceptions") else ["- (none)"])
                if fm_prompts: md += ["","## Notes", fm_prompts]
                bundle[f"{name}/{name}_documentation.md"] = "\n".join(md).encode()

            if subtask in ["Spec + Docs + Stub","Spec + Docs + Stub + Unit test","Stub only"]:
                bundle[f"{name}/{name.lower()}_stub.py"] = fm_python_stub(spec, fm_prompts or "").encode()

            if subtask in ["Spec + Docs + Stub + Unit test","Unit test only"]:
                bundle[f"{name}/test_{name.lower()}.py"] = fm_pytest_skeleton(spec).encode()

            logs.append(f"✅ Parsed: {name}")
        except Exception as e:
            logs.append(f"⚠️ Skipped {src.name}: {e}")

    st.success("FM conversion completed.")
    st.code("\n".join(logs), language="text")
    st.download_button("📦 Download FM outputs (ZIP)", data=zip_named_files(bundle),
                       file_name="fm_outputs.zip", mime="application/zip")
