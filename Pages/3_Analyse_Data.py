# pages/3_Analyse_Data.py
import streamlit as st
from pathlib import Path
import json, tempfile
import pandas as pd
from utils import (
    zip_named_files, extract_zip_to_tmp, iter_files,
    quick_profile, reconcile,
    analyse_dependencies, extract_fms_from_text
)

st.title("📊 Analyse data")

subtask = st.selectbox("Choose Data sub-task", [
    "Profile CSV(s)",
    "Reconcile two CSVs (diff by keys)",
    "BW Dependency (folder/zip/logs)"
])

with st.expander("Advanced prompts / notes (optional)"):
    notes = st.text_area("Notes to embed into generated reports", height=120)

# ---------- A) Profile CSV(s) ----------
if subtask == "Profile CSV(s)":
    mode = st.radio("Input mode", [
        "Upload CSV file(s)",
        "Upload ZIP of a folder (Cloud)",
        "Local folder path (run locally)"
    ])

    csv_uploads = zip_upload = None
    folder_path = None
    if mode == "Upload CSV file(s)":
        csv_uploads = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    elif mode == "Upload ZIP of a folder (Cloud)":
        zip_upload = st.file_uploader("Upload ZIP containing CSVs", type=["zip"])
    else:
        folder_path = st.text_input("Local folder path (recursive *.csv)", value="", placeholder=r"C:\data\profiles")

    run = st.button("🔎 Profile", type="primary")
    if run:
        csv_paths = []
        if csv_uploads:
            in_dir = Path(tempfile.mkdtemp()) / "csv_in"; in_dir.mkdir(parents=True, exist_ok=True)
            for f in csv_uploads: (in_dir / f.name).write_bytes(f.getvalue())
            csv_paths = [p for p in in_dir.iterdir() if p.suffix.lower()==".csv"]
        elif zip_upload:
            root = extract_zip_to_tmp(zip_upload)
            csv_paths = iter_files(root, (".csv",))
            if not csv_paths: st.error("No CSV in ZIP."); st.stop()
        else:
            if not folder_path: st.error("Enter folder or choose another mode."); st.stop()
            root = Path(folder_path)
            if not root.exists(): st.error(f"Path not found: {root}"); st.stop()
            csv_paths = iter_files(root, (".csv",))
            if not csv_paths: st.error("No CSVs under folder."); st.stop()

        outputs = {}; tabs = st.tabs([p.name for p in csv_paths[:10]])
        for i, p in enumerate(csv_paths):
            try:
                df = pd.read_csv(p)
            except Exception:
                df = pd.read_csv(p, encoding="latin-1")
            rpt = quick_profile(df, p.name)
            if i < len(tabs):
                with tabs[i]:
                    st.subheader(f"Preview — {p.name}")
                    st.dataframe(df.head(200))
                    nulls = {c["column"]: c["nulls"] for c in rpt["columns_profile"]}
                    st.caption("Nulls per column")
                    st.bar_chart(pd.Series(nulls))
            outputs[f"{p.name}/{p.stem}_profile.json"] = json.dumps(rpt, indent=2).encode()
            md = [f"# Data Profile: {p.name}", f"Rows: {rpt['rows']}", f"Columns: {rpt['columns']}", ""]
            if notes: md += ["## Notes", notes, ""]
            md += ["## Columns"]
            for c in rpt["columns_profile"]:
                row = f"- **{c['column']}** (`{c['dtype']}`) non_null={c['non_null']}, nulls={c['nulls']}, distinct={c['distinct']}"
                for k in ["min","max","mean","std"]:
                    if k in c: row += f", {k}={c[k]}"
                md.append(row)
            outputs[f"{p.name}/{p.stem}_profile.md"] = "\n".join(md).encode()

        st.success(f"Profiling completed. Files profiled: {len(csv_paths)}")
        st.download_button("📦 Download all profiles (ZIP)", data=zip_named_files(outputs),
                           file_name="data_profiles.zip", mime="application/zip")

# ---------- B) Reconcile two CSVs ----------
elif subtask == "Reconcile two CSVs (diff by keys)":
    left = st.file_uploader("Left CSV", type=["csv"])
    right = st.file_uploader("Right CSV", type=["csv"])
    keys_line = st.text_input("Join keys (comma-separated)", value="")
    run = st.button("🔁 Reconcile", type="primary")
    if run:
        if not (left and right and keys_line.strip()):
            st.error("Upload both CSVs and provide keys."); st.stop()
        keys = [k.strip() for k in keys_line.split(",") if k.strip()]
        import pandas as pd, io
        dfL = pd.read_csv(io.BytesIO(left.getvalue()))
        dfR = pd.read_csv(io.BytesIO(right.getvalue()))
        missing = [k for k in keys if k not in dfL.columns or k not in dfR.columns]
        if missing: st.error(f"Keys not in both files: {missing}"); st.stop()
        out = reconcile(dfL, dfR, keys)
        st.success("Reconciliation completed.")
        st.subheader("Summary")
        st.write(f"Left only: {len(out['left_only'])} | Right only: {len(out['right_only'])} | Columns diff: {len(out['diffs'])}")
        files = {"summary/notes.md": (notes or "").encode()}
        if len(out["left_only"])>0: files["summary/left_only.csv"] = out["left_only"].to_csv(index=False).encode()
        if len(out["right_only"])>0: files["summary/right_only.csv"] = out["right_only"].to_csv(index=False).encode()
        for col, df in out["diffs"]:
            files[f"diffs/{col}_mismatch.csv"] = df.to_csv(index=False).encode()
        st.download_button("📦 Download reconciliation ZIP", data=zip_named_files(files),
                           file_name="reconcile.zip", mime="application/zip")

# ---------- C) BW Dependency ----------
else:
    st.info("Analyse **transformation/dependency_log** (or **dependencies_log**) under each use case.")
    mode = st.radio("Input mode", [
        "Upload ZIP (Cloud)",
        "Upload multiple dependency_log files",
        "Local folder path (run locally)"
    ])

    zip_up = logs_multi = None
    folder_path = None
    if mode == "Upload ZIP (Cloud)":
        zip_up = st.file_uploader("Upload ZIP of extractor root", type=["zip"])
    elif mode == "Upload multiple dependency_log files":
        logs_multi = st.file_uploader("Upload dependency_log files (multi-select)", type=None, accept_multiple_files=True)
    else:
        folder_path = st.text_input("Local folder path to extractor root", value="", placeholder=r"D:\BW\Exports\Root")

    run = st.button("🧩 Analyse dependencies", type="primary")
    if run:
        from utils import analyse_dependencies
        if zip_up:
            root = extract_zip_to_tmp(zip_up)
            result = analyse_dependencies(root, notes or "")
            if "error" in result: st.error(result["error"]); st.stop()
        elif logs_multi:
            # Build mapping directly from multiple uploaded log files
            usecase_to_fms, fm_to_usecases = {}, {}
            for up in logs_multi:
                raw = up.getvalue().decode("utf-8", errors="ignore")
                path_guess = Path(up.name)
                uc = path_guess.parent.name if path_guess.parent.name else path_guess.stem
                if uc.lower().startswith(("dependency_log","dependencies_log")): uc = "UNKNOWN_USECASE"
                fms = extract_fms_from_text(raw)
                if fms:
                    usecase_to_fms.setdefault(uc, set()).update(fms)
                    for fm in fms: fm_to_usecases.setdefault(fm, set()).add(uc)

            rows_uc = [{"use_case": uc, "function_module": fm}
                       for uc, fms in sorted(usecase_to_fms.items()) for fm in sorted(fms)]
            rows_fm = [{"function_module": fm, "use_case": uc}
                       for fm, ucs in sorted(fm_to_usecases.items()) for uc in sorted(ucs)]
            import pandas as pd
            df_uc = pd.DataFrame(rows_uc)
            df_fm = pd.DataFrame(rows_fm)
            files = {
                "summary/overview.md": f"# BW Dependency Analysis\n\n- Unique FMs: {len(fm_to_usecases)}\n\n## Notes\n{notes or ''}\n".encode()
            }
            if not df_uc.empty: files["tables/usecase_to_fms.csv"] = df_uc.to_csv(index=False).encode()
            if not df_fm.empty: files["tables/fm_to_usecases.csv"] = df_fm.to_csv(index=False).encode()
            result = {"df_uc": df_uc, "df_fm": df_fm, "unique_fm_count": len(fm_to_usecases), "zip_bytes": zip_named_files(files)}

        else:
            if not folder_path: st.error("Enter local path or choose another mode."); st.stop()
            root = Path(folder_path)
            if not root.exists(): st.error(f"Path not found: {root}"); st.stop()
            result = analyse_dependencies(root, notes or "")
            if "error" in result: st.error(result["error"]); st.stop()

        st.success(f"Unique FMs across use cases: **{result['unique_fm_count']}**")
        tabs = st.tabs(["Use case → FMs", "FM → Use cases"])
        with tabs[0]:
            if result["df_uc"] is None or result["df_uc"].empty:
                st.write("No FM references found.")
            else:
                top = result["df_uc"].groupby("use_case")["function_module"].nunique().sort_values(ascending=False).head(20)
                st.caption("Top use cases by unique FM count"); st.bar_chart(top)
                st.dataframe(result["df_uc"])
        with tabs[1]:
            if result["df_fm"] is None or result["df_fm"].empty:
                st.write("No FM references found.")
            else:
                top = result["df_fm"].groupby("function_module")["use_case"].nunique().sort_values(ascending=False).head(20)
                st.caption("Top FMs by number of use cases"); st.bar_chart(top)
                st.dataframe(result["df_fm"])

        st.download_button("📦 Download dependency bundle (ZIP)", data=result["zip_bytes"],
                           file_name="bw_dependency_analysis.zip", mime="application/zip")
