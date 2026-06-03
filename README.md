# 🏏 IPL Analytics Hub — v2

Advanced 8-page ML-powered Streamlit app for IPL match analysis.

## 📁 Structure
```
ipl_app_v2/
├── app.py               ← Main app (8 pages)
├── train_model.py       ← ML training pipeline
├── analytics.py         ← Cached analytics helpers
├── data_loader.py       ← ⭐ CSV registry (add new CSVs here)
├── ui_components.py     ← Shared UI building blocks
├── requirements.txt
├── pipe.pkl             ← Pre-trained model
├── .streamlit/config.toml
└── data/
    ├── ipl_matches_data.csv
    ├── teams_data.csv
    └── players-data-updated.csv
```

## 🚀 Deploy (Streamlit Cloud — Free)
1. Push folder to GitHub repo
2. [share.streamlit.io](https://share.streamlit.io) → New app → `app.py`
3. Deploy — live in ~2 min

## 💻 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ➕ Add a New CSV
1. Drop `.csv` in `data/`
2. Add entry in `data_loader.py → DATA_REGISTRY`
3. Call `load_data("your_key")` anywhere

## 8 Pages
| Page | Features |
|------|----------|
| 🎯 Win Predictor | Live probability + scenario comparison table |
| 📊 Season Dashboard | 18-season trends, win methods, season leaders |
| 🏆 Team Stats | All-time leaderboard, deep-dive per team |
| ⚔️ Head to Head | H2H record, season-wise dominance |
| 🏟️ Venue Analysis | Chase vs defend win% per ground + city |
| 🎲 Toss Intelligence | Toss advantage by team, city, season |
| 👤 Player Profiles | 772 players, POM leaders, style breakdown |
| 🗂️ Data Explorer | Schema/stats/preview of all CSVs |
