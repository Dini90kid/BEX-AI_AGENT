import streamlit as st
from pathlib import Path
import zipfile
import io
import tempfile
import agent

st.set_page_config(page_title="BEx Query Agent", page_icon="🧠")

st.title("🧠 BEx GP Agent – Streamlit UI")
st.write("Upload BEx GP .txt files and generate JSON spec + documentation + test data.")

uploaded_files = st.file_uploader(
    "Upload one or more GP .txt files",
    type=["txt"],
    accept_multiple_files=True
)

run_button = st.button("🚀 Run Agent")

def zip_folder(folder_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for file_path in Path(folder_path).rglob("*"):
            if file_path.is_file():
                z.write(file_path, arcname=file_path.relative_to(folder_path))
    buffer.seek(0)
    return buffer

if run_button:
    if not uploaded_files:
        st.error("Please upload at least one .txt GP file.")
    else:
        temp_dir = Path(tempfile.mkdtemp())
        input_dir = temp_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded .txt files
        for f in uploaded_files:
            (input_dir / f.name).write_bytes(f.getvalue())

        out_dir = input_dir / "_agent_output"
        out_dir.mkdir(exist_ok=True)

        # Optional overrides file if user put it in the same folder later
        overrides = agent.load_overrides(input_dir)

        # Collect .txt files and run the agent
        gp_files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]

        logs = []
        for p in gp_files:
            try:
                agent.process_gp_file(p, overrides, out_dir)
                logs.append(f"✅ Processed: {p.name}")
            except Exception as e:
                logs.append(f"⚠️ Skipped {p.name}: {e}")

        st.success("Agent run completed.")
        st.code("\n".join(logs))

        # Offer ZIP download of outputs
        zip_bytes = zip_folder(out_dir)
        st.download_button(
            "📦 Download outputs (ZIP)",
            data=zip_bytes,
            file_name="bex_agent_output.zip",
            mime="application/zip",
        )
