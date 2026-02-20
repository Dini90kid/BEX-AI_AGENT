# --- HARD PROOF v1 ---
import streamlit as st, os, glob, time, platform, sys
st.set_page_config(page_title="HARD PROOF", layout="wide")
st.title("✅ HARD PROOF — LIVE FILE CONTENT")
st.write("Server time:", time.strftime("%Y-%m-%d %H:%M:%S"))
st.code(f"__file__ = {__file__}")
st.code(f"CWD     = {os.getcwd()}")
st.code("Files in CWD:\n" + "\n".join(sorted(glob.glob('*'))))
st.code(f"Python={sys.version.split()[0]}  Platform={platform.platform()}")
