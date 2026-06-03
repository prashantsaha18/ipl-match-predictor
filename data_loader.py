"""
data_loader.py — Centralised CSV Registry
==========================================
Drop any CSV into data/ → add one entry here → use it everywhere.
"""
import os
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── REGISTRY ─────────────────────────────────────────────────────
# To add a new CSV: drop file in data/, add entry below, use load_data("key")
DATA_REGISTRY = {
    "matches": {
        "filename"   : "ipl_matches_data.csv",
        "encoding"   : "utf-8",
        "description": "IPL match metadata — results, venues, toss, stage (2008-2025)",
    },
    "teams": {
        "filename"   : "teams_data.csv",
        "encoding"   : "utf-8",
        "description": "Team names, short codes and official logo URLs",
    },
    "players": {
        "filename"   : "players-data-updated.csv",
        "encoding"   : "latin1",
        "description": "Player profiles — batting/bowling style, field position, photo URLs",
    },
    # ── ADD NEW DATASETS HERE ─────────────────────────────────────
    # "my_dataset": {
    #     "filename"   : "my_file.csv",
    #     "encoding"   : "utf-8",
    #     "description": "Describe the dataset",
    # },
}


@st.cache_data(show_spinner=False)
def load_data(key: str) -> pd.DataFrame:
    if key not in DATA_REGISTRY:
        raise KeyError(f"Unknown key '{key}'. Available: {list(DATA_REGISTRY)}")
    meta = DATA_REGISTRY[key]
    path = os.path.join(DATA_DIR, meta["filename"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, encoding=meta["encoding"])
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().replace("NULL", pd.NA)
    return df


def registry_info() -> pd.DataFrame:
    rows = []
    for key, meta in DATA_REGISTRY.items():
        path   = os.path.join(DATA_DIR, meta["filename"])
        exists = os.path.exists(path)
        try:
            df    = load_data(key) if exists else None
            shape = f"{df.shape[0]:,} × {df.shape[1]}" if df is not None else "—"
        except Exception:
            shape = "error"
        rows.append({
            "Key": key, "File": meta["filename"],
            "Status": "✅" if exists else "❌",
            "Shape": shape, "Description": meta["description"],
        })
    return pd.DataFrame(rows)
