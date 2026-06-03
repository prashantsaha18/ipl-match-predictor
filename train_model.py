"""
train_model.py — IPL Win Probability ML Pipeline
Run once to produce pipe.pkl before launching app.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd, numpy as np, pickle
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import OneHotEncoder
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score, roc_auc_score

SEED        = 42
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PKL  = os.path.join(os.path.dirname(__file__), "pipe.pkl")
FEATURE_COLS = ["team_batting","team_bowling","city","runs_left","balls_left",
                "wickets_left","target","crr","rrr"]
np.random.seed(SEED)

print("="*58)
print("  IPL Win Probability — Training Pipeline v2")
print("="*58)

# ── Load ─────────────────────────────────────────────────────
print("\n[1/6] Loading & cleaning match data …")
m = pd.read_csv(os.path.join(DATA_DIR,"ipl_matches_data.csv"))
for c in m.select_dtypes(include="object").columns:
    m[c] = m[c].str.strip().replace("NULL", np.nan)
m = m[(m["overs"]==20)&(m["result"]=="win")].dropna(subset=["match_winner","team1","team2","city"])
m.reset_index(drop=True, inplace=True)
print(f"    ✓ {len(m):,} valid matches")

# ── Innings structure ────────────────────────────────────────
print("\n[2/6] Deriving batting order …")
m["losing_team"]         = m.apply(lambda r: r["team2"] if r["match_winner"]==r["team1"] else r["team1"], axis=1)
wbr = m["win_by_runs"].fillna(0)>0
wbw = m["win_by_wickets"].fillna(0)>0
m["team_batting_first"]  = np.where(wbr, m["match_winner"], m["losing_team"])
m["team_batting_second"] = np.where(wbr, m["losing_team"],  m["match_winner"])

# ── Innings scores ───────────────────────────────────────────
print("\n[3/6] Generating realistic innings totals …")
m["inns1_total"] = np.clip(np.random.normal(165,22,len(m)).astype(int),130,225)
m["target"]      = m["inns1_total"]+1
m.loc[wbr,"inns2_total"] = (m.loc[wbr,"inns1_total"]-m.loc[wbr,"win_by_runs"]).clip(lower=30)
m.loc[wbw,"inns2_total"] = m.loc[wbw,"target"]
print(f"    ✓ Target range {m['target'].min()}–{m['target'].max()}")

# ── Simulate chase ───────────────────────────────────────────
print("\n[4/6] Simulating ball-by-ball chase progressions …")
def simulate_chase(row):
    target=int(row["target"]); chasing=row["team_batting_second"]
    defending=row["team_batting_first"]; city=row["city"]
    chasing_won=(row["match_winner"]==chasing)
    cum_r=0; cum_w=0; recs=[]
    for ov in range(20):
        for bl in range(1,7):
            bd=ov*6+bl
            r=np.random.random()
            runs=0 if r<.35 else 1 if r<.60 else 2 if r<.75 else 3 if r<.82 else 4 if r<.90 else 5 if r<.94 else 6
            wkt=1 if (np.random.random()<.05 and cum_w<9) else 0
            cum_r+=runs; cum_w+=wkt
            bl_left=max(120-bd,0); rl=max(target-cum_r,0)
            crr=(cum_r*6/bd) if bd>0 else 0.0
            rrr=(rl*6/bl_left) if bl_left>0 else (0.0 if rl==0 else 99.9)
            recs.append({"match_id":row["match_id"],"team_batting":chasing,
                "team_bowling":defending,"city":city,"ball_number":bl,
                "balls_completed":bd,"balls_left":bl_left,"current_score":cum_r,
                "runs_left":rl,"wickets_lost":cum_w,"wickets_left":max(10-cum_w,0),
                "target":target,"crr":round(crr,4),"rrr":round(min(rrr,99.9),4),
                "result":1 if chasing_won else 0})
            if cum_w>=10 or cum_r>=target: return recs
    return recs

rows=[]
for _,row in m.iterrows(): rows.extend(simulate_chase(row))
bbb=pd.DataFrame(rows)
print(f"    ✓ {len(bbb):,} snapshots, {m['match_id'].nunique():,} matches")

# ── Features ─────────────────────────────────────────────────
print("\n[5/6] Feature engineering …")
bbb["crr"]=bbb["crr"].replace([np.inf,-np.inf],0).clip(0,36)
bbb["rrr"]=bbb["rrr"].replace([np.inf,-np.inf],99.9).clip(0,99.9)
sampled=bbb[bbb["ball_number"].isin([2,4,6])][FEATURE_COLS+["result"]].dropna().reset_index(drop=True)
print(f"    ✓ {len(sampled):,} training samples | balance {sampled['result'].value_counts().to_dict()}")
X=sampled[FEATURE_COLS]; y=sampled["result"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)

# ── Pipeline ─────────────────────────────────────────────────
print("\n[6/6] Training LogisticRegression pipeline …")
cat=["team_batting","team_bowling","city"]
num=["runs_left","balls_left","wickets_left","target","crr","rrr"]
pre=ColumnTransformer([("cat",OneHotEncoder(drop="first",handle_unknown="ignore"),cat),("num","passthrough",num)])
pipe=Pipeline([("pre",pre),("clf",LogisticRegression(solver="liblinear",max_iter=1000,C=1.0))])
pipe.fit(X_train,y_train)
acc=accuracy_score(y_test,pipe.predict(X_test))
auc=roc_auc_score(y_test,pipe.predict_proba(X_test)[:,1])
print(f"    ✓ Accuracy: {acc*100:.2f}%  |  AUC: {auc:.4f}")
with open(OUTPUT_PKL,"wb") as f: pickle.dump(pipe,f)
print(f"\n✅  Saved → {OUTPUT_PKL}")
print("="*58)
