# --- RADIO SMOKE TEST v0 ---
import streamlit as st
st.set_page_config(page_title="RADIO SMOKE TEST", layout="wide")
st.title("🔥 RADIO SMOKE TEST")
choice = st.radio("Pick one:", ["BEx Conversion", "Function Module Conversion", "Analyse Data", "BW Dependency"])
st.success(f"You picked: {choice}")
