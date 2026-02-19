# agent.py
# Basic AI Agent for BEx GP programs → JSON spec + Docs + Test Data by Dinesh
# Usage: python agent.py  (you will be prompted for the input folder)

import os
import re
import json
import random
from pathlib import Path

# Optional dependency: pandas (only for CSV creation).
# If pandas is unavailable, the agent will emit CSV via stdlib.
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


# ===========================================================
# 0) Utilities
# ===========================================================

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

def _write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=4, ensure_ascii=False), encoding="utf-8")

def _write_csv(path: Path, rows: list[dict]):
    if _HAS_PANDAS:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        # Minimal CSV writer without pandas
        import csv
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


# ===========================================================
# 1) Ask user for input folder
# ===========================================================

def ask_user_for_folder() -> Path:
    print("Enter folder path containing BEx GP reports (.txt):")
    folder = input("Path: ").strip().strip('"').strip("'")
    p = Path(folder)
    if not p.is_dir():
        raise SystemExit(f" ❌ Folder not found: {p}")
    return p


# ===========================================================
# 2) Load optional overrides (ID → InfoObject)
#    Put a 'char_map_overrides.json' inside the same folder, e.g.:
#    { "009": "0CALMONTH", "282": "0DOC_CURRCY" }
# ===========================================================

def load_overrides(folder: Path) -> dict:
    override_file = folder / "char_map_overrides.json"
    if override_file.exists():
        try:
            return json.loads(override_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️  Could not read {override_file.name}: {e}")
    return {}


# ===========================================================
# 3) Core regex helpers
# ===========================================================

RE_INFOCUBE   = re.compile(r"INFOCUBE\.*:\s*([A-Z0-9_/]+)")
RE_REPORTNAME = re.compile(r"REPORT\.*:\s*([A-Z0-9_]+)")
RE_KF         = re.compile(r"\bZ____\d+_SUM\b")
RE_S_IDS      = re.compile(r"\bS____(\d+)\b")
RE_K_IDS      = re.compile(r"\bK____(\d+)\b")

# FORM headers for variable wiring; A_S_ (SID flavor), A_K_ (KEY flavor)
RE_FORM_START = re.compile(r"\bFORM\s+(A_[SK])_0*([0-9]{1,9})\b", re.IGNORECASE)
RE_FORM_END   = re.compile(r"\bENDFORM\b", re.IGNORECASE)

# In-form variable references like: <G>-GVAR-LS0155, -HS0167, -LK0155, -HK0155, KS0130, MS0161...
RE_GVAR_TOKEN = re.compile(r"\bGVAR-(LS|HS|LK|HK|KS|KK|MS|MK)(\d{4})\b")


# ===========================================================
# 4) Parsers
# ===========================================================

def parse_infocube(text: str) -> str | None:
    m = RE_INFOCUBE.search(text)
    return m.group(1) if m else None

def parse_query_name(text: str) -> str | None:
    m = RE_REPORTNAME.search(text)
    return m.group(1) if m else None

def parse_key_figures(text: str) -> list[str]:
    return sorted(set(RE_KF.findall(text)))

def parse_characteristic_ids(text: str) -> list[str]:
    ids = set(RE_S_IDS.findall(text)) | set(RE_K_IDS.findall(text))
    return sorted(ids)

def parse_variable_wiring(text: str) -> dict:
    """
    Returns { char_id : { "forms": [{"mode":"A_S"|"A_K", "vars":[tokens]}], "var_index": {"LS":[...], "HS":[...], ...} } }
    We don't assume InfoObject names here. This mirrors GP A_S_/A_K_ FORMs.
    """
    vars_by_id: dict[str, dict] = {}

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = RE_FORM_START.search(line)
        if not m:
            i += 1
            continue

        mode, cid = m.group(1).upper(), m.group(2).lstrip("0") or "0"
        block_vars = []

        # Scan until ENDFORM
        i += 1
        while i < len(lines) and not RE_FORM_END.search(lines[i]):
            for t in RE_GVAR_TOKEN.findall(lines[i]):
                # t = (token_type, digits)
                block_vars.append(f"{t[0]}{t[1]}")
            i += 1

        # Store
        v = vars_by_id.setdefault(cid, {"forms": [], "var_index": {}})
        v["forms"].append({"mode": mode, "vars": sorted(set(block_vars))})
        for tok in block_vars:
            prefix = tok[:2]  # LS/HS/LK/HK/KS/KK/MS/MK
            v["var_index"].setdefault(prefix, []).append(tok)

        i += 1  # skip ENDFORM line
    return vars_by_id


def infer_infoobject_names(text: str) -> set[str]:
    """
    Heuristic scan for tokens that *look like* BW InfoObjects (e.g., 0CALMONTH, 0DOC_CURRCY, NCOST_COD).
    This does NOT link them to IDs — that mapping is resolved via overrides if provided.
    """
    candidates = set(re.findall(r"\b[0N][A-Z0-9_]{3,}\b", text))
    # Common noise reduction (very light)
    noise = {"NOW", "NOT", "NULL", "NONE"}
    return {c for c in candidates if c not in noise}


# ===========================================================
# 5) Documentation generator
# ===========================================================

def build_markdown(query_name: str, cube: str | None, char_ids: list[str], id_to_io: dict[str, str], kfs: list[str], var_wiring: dict, loose_ios: set[str]) -> str:
    out = []
    out.append(f"# Documentation — {query_name or 'UNKNOWN_QUERY'}")
    out.append("")
    out.append("## InfoCube")
    out.append(cube or "UNKNOWN")
    out.append("")
    out.append("## Characteristics")
    if not char_ids:
        out.append("- (none found)")
    else:
        for cid in char_ids:
            io = id_to_io.get(cid)
            label = f"{io} (Tech ID {cid})" if io else f"ID {cid} (InfoObject: unknown)"
            out.append(f"- {label}")
    out.append("")
    if loose_ios:
        out.append("**Other InfoObject-like tokens detected in GP:**")
        for io in sorted(loose_ios):
            out.append(f"- {io}")
        out.append("")

    out.append("## Variables & Selection Wiring (from A_S_/A_K_ FORMs)")
    if not var_wiring:
        out.append("- (no A_S_/A_K_ FORMs detected)")
    else:
        for cid, meta in sorted(var_wiring.items(), key=lambda x: int(x[0])):
            io = id_to_io.get(cid, "(unknown)")
            out.append(f"### {io} — ID {cid}")
            for blk in meta["forms"]:
                out.append(f"- **{blk['mode']}** uses tokens: {', '.join(blk['vars']) if blk['vars'] else '(none)'}")
            # Compact index by token prefix
            if meta["var_index"]:
                out.append("  - Index by token prefix:")
                for pref, toks in sorted(meta["var_index"].items()):
                    out.append(f"    - {pref}: {', '.join(sorted(set(toks)))}")
            out.append("")

    out.append("## Key Figures")
    if kfs:
        for k in kfs:
            out.append(f"- {k}")
    else:
        out.append("- (none found)")

    return "\n".join(out)


# ===========================================================
# 6) Test data generator
# ===========================================================

def make_test_rows(char_ids: list[str], id_to_io: dict[str, str], kfs: list[str], rows: int = 50) -> list[dict]:
    out = []
    # Build deterministic characteristic columns
    ch_cols = []
    for cid in char_ids:
        io = id_to_io.get(cid)
        ch_cols.append(io if io else f"CH_{cid}")

    for i in range(rows):
        row = {}
        for c in ch_cols:
            row[c] = f"{c}_VAL_{i:03d}"
        for k in kfs:
            # Numeric sample (simple)
            row[k] = round(random.uniform(10, 1000), 2)
        out.append(row)
    return out


# ===========================================================
# 7) Main agent
# ===========================================================

def process_gp_file(path: Path, overrides: dict, out_root: Path):
    text = _read_text(path)

    cube  = parse_infocube(text)
    qname = parse_query_name(text) or path.stem
    kfs   = parse_key_figures(text)
    cids  = parse_characteristic_ids(text)
    vw    = parse_variable_wiring(text)
    loose = infer_infoobject_names(text)

    # Resolve ID → InfoObject mapping
    #  - Use overrides if provided (recommended).
    #  - You can extend this with your own heuristics.
    id_to_io: dict[str, str] = {}
    for cid in cids:
        if cid in overrides:
            id_to_io[cid] = overrides[cid]

    # Build JSON spec
    spec = {
        "query_name": qname,
        "infocube": cube,
        "characteristics": [
            {"tech_id": cid, "infoobject": id_to_io.get(cid)} for cid in cids
        ],
        "key_figures": kfs,
        "variables": vw,               # raw A_S_/A_K_ wiring tokens
        "detected_infoobjects": sorted(loose),  # extra tokens found (not linked)
    }

    # Write outputs
    qslug = re.sub(r"[^A-Za-z0-9_]+", "_", qname)[:120] or path.stem

    json_path = out_root / f"{qslug}_spec.json"
    md_path   = out_root / f"{qslug}_documentation.md"
    csv_path  = out_root / f"{qslug}_testdata.csv"

    _write_json(json_path, spec)
    md = build_markdown(qname, cube, cids, id_to_io, kfs, vw, loose)
    _write_text(md_path, md)

    rows = make_test_rows(cids, id_to_io, kfs, rows=50)
    _write_csv(csv_path, rows)

    print(f"  ✅ JSON: {json_path.name}")
    print(f"  📄 DOC:  {md_path.name}")
    print(f"  🧪 CSV:  {csv_path.name}")


def main():
    folder = ask_user_for_folder()
    out_root = folder / "_agent_output"
    out_root.mkdir(exist_ok=True)

    overrides = load_overrides(folder)
    if overrides:
        print(f"🔧 Loaded overrides: {len(overrides)} mappings")

    gp_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    if not gp_files:
        raise SystemExit(" ❌ No .txt GP files found in the folder.")

    print(f"\nFound {len(gp_files)} file(s). Processing...\n")

    for p in gp_files:
        print(f"• {p.name}")
        try:
            process_gp_file(p, overrides, out_root)
        except Exception as e:
            print(f"  ⚠️  Skipped due to error: {e}")

    print(f"\n🎉 Done. Outputs written to: {out_root}\n")
    print("Tip: create 'char_map_overrides.json' in the same folder to map characteristic IDs to InfoObjects, e.g.:")
    print('     { "009": "0CALMONTH", "282": "0DOC_CURRCY" }')


if __name__ == "__main__":
    main()
