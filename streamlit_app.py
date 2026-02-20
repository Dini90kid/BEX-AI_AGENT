# streamlit_app.py — Home
st.set_page_config(
    page_title="BEx / FM / Data Suite",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)
import streamlit as st, os
st.set_page_config(page_title="BEx / FM / Data Suite", page_icon="🧭", layout="wide")

st.title("🧭 BEx / FM / Data Suite")
st.write("""
Welcome! Use the sidebar to open:
- **BEx conversion** — Convert BEx GP `.txt` files → JSON spec + docs + test data (+ optional PySpark).
- **Function Module conversion** — Parse ABAP FM source → spec + docs + Python stub (+ pytest).
- **Analyse data** — Profile CSVs, reconcile datasets, or analyse BW dependency logs.
""")

st.caption(f"__file__ = {__file__}")
st.caption(f"CWD = {os.getcwd()}")
