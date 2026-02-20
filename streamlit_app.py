import streamlit as st, time
st.set_page_config(page_title="VERSION STAMP", layout="wide")
st.title("✅ VERSION STAMP")
st.write("This page proves the app is reading **feature/streamlit_app.py**.")
st.write("Server time:", time.strftime("%Y-%m-%d %H:%M:%S"))
