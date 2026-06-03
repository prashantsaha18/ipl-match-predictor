"""
app.py — IPL Win Probability Predictor v2
8-page Streamlit app with full analytics suite
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np

from data_loader   import load_data, registry_info, DATA_REGISTRY
from analytics     import (get_matches, get_teams, get_players, get_ui_lists,
                            team_all_time_stats, season_summary, head_to_head,
                            venue_stats, toss_analysis, win_margin_dist,
                            player_of_match_leaders, city_performance)
from ui_components import (inject_css, page_hero, section_hdr, glass_card,
                            team_prob_card, dominance_banner, stat_pill, info_box)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Analytics Hub",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Model loader ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    p = os.path.join(os.path.dirname(__file__), "pipe.pkl")
    if not os.path.exists(p): return None
    with open(p,"rb") as f: return pickle.load(f)

# ── Sidebar navigation ────────────────────────────────────────
PAGES = {
    "🎯 Win Predictor"   : "predictor",
    "📊 Season Dashboard": "season",
    "🏆 Team Stats"      : "teams",
    "⚔️  Head to Head"   : "h2h",
    "🏟️  Venue Analysis" : "venue",
    "🎲 Toss Intelligence": "toss",
    "👤 Player Profiles" : "players",
    "🗂️  Data Explorer"  : "data",
}

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:.6rem 0 1rem;">
      <div style="font-size:2.6rem;">🏏</div>
      <div style="font-weight:800;font-size:1.05rem;color:#a78bfa!important;">IPL Analytics Hub</div>
      <div style="font-size:.68rem;color:#475569!important;">ML-Powered · 2008–2025</div>
    </div>""", unsafe_allow_html=True)

    choice = st.radio("", list(PAGES), label_visibility="collapsed")
    page   = PAGES[choice]

    st.markdown("---")
    st.markdown(
        "<p style='font-size:.65rem;color:#334155!important;text-align:center;'>"
        "Logistic Regression · Scikit-Learn<br/>1,169 matches · 18 seasons</p>",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — WIN PREDICTOR
# ══════════════════════════════════════════════════════════════
if page == "predictor":
    pipe = load_model()
    cities, all_teams, logo_map, short_map = get_ui_lists()
    page_hero("🎯","Live Win Probability","Real-time ML prediction — configure the match scenario below")

    if not pipe:
        st.error("pipe.pkl missing — run `python train_model.py` first.", icon="⚠️"); st.stop()

    # Sidebar inputs
    with st.sidebar:
        st.markdown("#### 🏙️ Venue"); city = st.selectbox("City", cities,
            index=cities.index("Mumbai") if "Mumbai" in cities else 0, label_visibility="collapsed")
        st.markdown("#### 🏃 Chasing Team"); chasing = st.selectbox("Chasing", all_teams,
            index=0, label_visibility="collapsed")
        st.markdown("#### 🛡️ Defending Team")
        defending = st.selectbox("Defending",[t for t in all_teams if t!=chasing],
            index=0, label_visibility="collapsed")
        st.markdown("#### 🎯 Target"); target = st.number_input("Target",50,300,175,1,label_visibility="collapsed")

    # Live inputs
    section_hdr("Live Match Situation","📡")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown("<p style='font-size:.8rem;font-weight:600;color:#94a3b8!important;margin-bottom:.2rem;'>🏃 Current Score</p>",unsafe_allow_html=True)
        score = st.number_input("score",0,300,70,1,label_visibility="collapsed",key="sc")
    with c2:
        st.markdown("<p style='font-size:.8rem;font-weight:600;color:#94a3b8!important;margin-bottom:.2rem;'>❌ Wickets Fallen</p>",unsafe_allow_html=True)
        wkts = st.number_input("wkts",0,10,3,1,label_visibility="collapsed",key="wk")
    with c3:
        st.markdown("<p style='font-size:.8rem;font-weight:600;color:#94a3b8!important;margin-bottom:.2rem;'>⏱️ Overs Completed</p>",unsafe_allow_html=True)
        overs = st.number_input("overs",0.0,20.0,10.0,.1,format="%.1f",label_visibility="collapsed",key="ov")
    with c4:
        st.markdown("<p style='font-size:.8rem;font-weight:600;color:#94a3b8!important;margin-bottom:.2rem;'>&nbsp;</p>",unsafe_allow_html=True)
        st.button("⚡ Calculate Probabilities",type="primary")

    # Derived
    ov_i=int(overs); bx=min(round((overs-ov_i)*10),5)
    bd=ov_i*6+bx; bl=max(120-bd,0); rl=max(target-score,0); wl=max(10-wkts,0)
    crr=(score*6/bd) if bd>0 else 0.0
    rrr=(rl*6/bl) if bl>0 else (0.0 if rl==0 else 99.9)

    # Formula dashboard
    section_hdr("Live Formula Dashboard","📐")
    fc1,fc2,fc3,fc4 = st.columns(4)
    with fc1: st.markdown(glass_card("Runs Required",str(rl),"🎯","#6366f1"),unsafe_allow_html=True)
    with fc2: st.markdown(glass_card("Wickets Left",str(wl),"🪃","#10b981" if wl>=7 else "#f59e0b" if wl>=4 else "#ef4444"),unsafe_allow_html=True)
    with fc3: st.markdown(glass_card("Current RR",f"{crr:.2f}","📈","#10b981" if crr>=rrr else "#f59e0b"),unsafe_allow_html=True)
    with fc4:
        rc="#ef4444" if rrr>12 else "#f59e0b" if rrr>9 else "#10b981"
        st.markdown(glass_card("Required RR",f"{rrr:.2f}" if rrr<99 else "—","🔥",rc),unsafe_allow_html=True)

    # Guards
    if score>=target: st.success(f"🎉 {chasing} have WON! Target reached."); st.stop()
    if wkts>=10 and score<target: st.error(f"💀 All out! {defending} WIN."); st.stop()
    if bl<=0 and score<target: st.error(f"⏰ Overs up! {defending} WIN by {rl-1} run(s)."); st.stop()

    # Prediction
    inp=pd.DataFrame([{"team_batting":chasing,"team_bowling":defending,"city":city,
        "runs_left":rl,"balls_left":bl,"wickets_left":wl,"target":target,
        "crr":round(crr,4),"rrr":round(min(rrr,99.9),4)}])
    pr=pipe.predict_proba(inp)[0]; cp=pr[1]*100; dp=pr[0]*100

    section_hdr("Win Probability Analysis","🏆")
    L,M,R=st.columns([5,1,5])
    with L:
        st.markdown(team_prob_card(chasing,cp,logo_map.get(chasing),cp>dp,short_map.get(chasing,chasing[:3])),unsafe_allow_html=True)
        st.markdown("<div style='height:.35rem'></div>",unsafe_allow_html=True)
        st.progress(int(min(cp,100)))
    with M: st.markdown("<div style='display:flex;height:100%;align-items:center;justify-content:center;font-size:1.4rem;font-weight:900;color:#334155!important;'>VS</div>",unsafe_allow_html=True)
    with R:
        st.markdown(team_prob_card(defending,dp,logo_map.get(defending),dp>cp,short_map.get(defending,defending[:3])),unsafe_allow_html=True)
        st.markdown("<div style='height:.35rem'></div>",unsafe_allow_html=True)
        st.progress(int(min(dp,100)))

    st.markdown(dominance_banner(chasing,defending,cp,dp),unsafe_allow_html=True)

    # Context
    st.markdown("<div style='height:1rem'></div>",unsafe_allow_html=True)
    section_hdr("Match Context","🔬")
    b1,b2,b3=st.columns(3)
    with b1:
        st.markdown(glass_card("Overs Used",f"{overs:.1f}/20","📅","#6366f1"),unsafe_allow_html=True)
        st.progress(int(bd/120*100))
    with b2:
        wc="#10b981" if wkts<4 else "#f59e0b" if wkts<7 else "#ef4444"
        st.markdown(glass_card("Wickets Fallen",f"{wkts}/10","💥",wc),unsafe_allow_html=True)
        st.progress(int(wkts/10*100))
    with b3:
        rp=min(score/target*100,100)
        st.markdown(glass_card("Target Done",f"{rp:.1f}%","🎯","#10b981" if rp>=50 else "#f59e0b"),unsafe_allow_html=True)
        st.progress(int(rp))

    if rrr<99:
        diff=crr-rrr
        col="#10b981" if diff>0 else "#ef4444"
        msg=(f"✅ CRR leads RRR by <strong>{diff:.2f}</strong> — {chasing} on comfortable pace"
             if diff>0 else
             f"⚠️ RRR ahead of CRR by <strong>{abs(diff):.2f}</strong> — {chasing} must accelerate!")
        st.markdown(f"<p style='color:{col}!important;font-size:.88rem;text-align:center;"
                    f"padding:.65rem;background:{col}0f;border-radius:10px;"
                    f"border:1px solid {col}33;margin-top:.8rem;'>{msg}</p>",unsafe_allow_html=True)

    # Scenario comparison table
    section_hdr("Scenario Comparison","🔄")
    info_box("Below shows how probability shifts under different over snapshots for this setup.", "#6366f1")
    snapshots=[]
    for snap_ov in [6,8,10,12,14,16,18]:
        snap_bd=snap_ov*6; snap_bl=max(120-snap_bd,0)
        # assume linear run rate
        projected_score=int(crr*snap_bd) if bd>0 else score
        snap_rl=max(target-projected_score,0)
        snap_crr=(projected_score*6/snap_bd) if snap_bd>0 else 0
        snap_rrr=(snap_rl*6/snap_bl) if snap_bl>0 else 0
        snap_inp=pd.DataFrame([{"team_batting":chasing,"team_bowling":defending,"city":city,
            "runs_left":snap_rl,"balls_left":snap_bl,"wickets_left":wl,
            "target":target,"crr":round(snap_crr,4),"rrr":round(min(snap_rrr,99.9),4)}])
        try:
            sp=pipe.predict_proba(snap_inp)[0]
            snapshots.append({"Over":snap_ov,"Projected Score":projected_score,
                "Runs Left":snap_rl,"CRR":f"{snap_crr:.2f}","RRR":f"{snap_rrr:.2f}",
                f"{chasing[:12]} Win%":f"{sp[1]*100:.1f}%",
                f"{defending[:12]} Win%":f"{sp[0]*100:.1f}%"})
        except: pass
    if snapshots:
        st.dataframe(pd.DataFrame(snapshots),use_container_width=True,hide_index=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — SEASON DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "season":
    df = get_matches()
    page_hero("📊","Season Dashboard","IPL season-by-season performance breakdown")

    section_hdr("All-Time Overview","🌏")
    wins=df[df["result"]=="win"]
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total Matches",f"{len(df):,}")
    m2.metric("Seasons",df["season"].nunique())
    m3.metric("Teams Ever",pd.concat([df["team1"],df["team2"]]).nunique())
    m4.metric("Cities",df["city"].nunique())
    m5.metric("Avg Win Margin (runs)",f"{wins['win_by_runs'].mean():.0f}")

    section_hdr("Season-by-Season Summary","📅")
    ss = season_summary()
    st.dataframe(ss.rename(columns={"season":"Season","total_matches":"Matches",
        "cities_used":"Cities","top_team":"Season Leader","season_wins":"Wins"}),
        use_container_width=True, hide_index=True)

    section_hdr("Matches Per Season","📈")
    season_counts = df.groupby("season").size().reset_index(name="Matches")
    st.bar_chart(season_counts.set_index("season")["Matches"],height=280,color="#6366f1")

    section_hdr("Season Win Leaders","🥇")
    wins_by_season = wins.groupby(["season","match_winner"]).size().reset_index(name="wins")
    top3 = wins_by_season.sort_values("wins",ascending=False).groupby("season").head(3)
    season_sel = st.selectbox("Pick a season",sorted(df["season"].unique().tolist(),reverse=True))
    s_data = top3[top3["season"]==season_sel].sort_values("wins",ascending=False)
    for i,(_,row) in enumerate(s_data.iterrows()):
        medal=["🥇","🥈","🥉"][i] if i<3 else ""
        col1,col2=st.columns([6,1])
        col1.markdown(f"<p style='margin:.2rem 0;font-size:.92rem;font-weight:600;color:#f1f5f9!important;'>{medal} {row['match_winner']}</p>",unsafe_allow_html=True)
        col2.markdown(f"<p style='text-align:right;color:#a78bfa!important;font-weight:700;'>{int(row['wins'])} wins</p>",unsafe_allow_html=True)
        st.progress(int(row['wins']/14*100))

    section_hdr("Win Method Distribution","🎯")
    wbr_cnt=wins[wins["win_by_runs"]>0].groupby("season").size()
    wbw_cnt=wins[wins["win_by_wickets"]>0].groupby("season").size()
    method_df=pd.DataFrame({"Wins by Runs":wbr_cnt,"Wins by Wickets":wbw_cnt}).fillna(0).astype(int)
    st.bar_chart(method_df,height=260,color=["#6366f1","#10b981"])


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — TEAM STATS
# ══════════════════════════════════════════════════════════════
elif page == "teams":
    page_hero("🏆","Team Statistics","All-time franchise performance since IPL 2008")
    df    = get_matches()
    teams = get_teams()
    stats = team_all_time_stats()
    logo_map = dict(zip(teams["team_name"],teams["image_url"]))

    section_hdr("All-Time Leaderboard","🏅")
    disp=stats.copy()
    disp.index=range(1,len(disp)+1)
    st.dataframe(
        disp[["team","played","wins","losses","win_pct","toss_match_wins"]].rename(
            columns={"team":"Team","played":"Played","wins":"Wins",
                     "losses":"Losses","win_pct":"Win %","toss_match_wins":"Toss→Win"}),
        use_container_width=True)

    section_hdr("Win % Bar Chart","📊")
    chart_df=stats.set_index("team")[["win_pct"]].rename(columns={"win_pct":"Win %"})
    st.bar_chart(chart_df,height=300,color="#6366f1")

    section_hdr("Deep Dive: Single Team","🔍")
    wins=df[df["result"]=="win"]
    team_sel = st.selectbox("Select a team", sorted(stats["team"].tolist()))
    t_wins = wins[wins["match_winner"]==team_sel]
    t_all  = df[(df["team1"]==team_sel)|(df["team2"]==team_sel)]

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Matches",f"{len(t_all):,}")
    col2.metric("Wins",f"{len(t_wins):,}")
    col3.metric("Win %",f"{len(t_wins)/len(t_all)*100:.1f}%")
    col4.metric("Best Season",
        wins[wins["match_winner"]==team_sel].groupby("season").size().idxmax()
        if len(t_wins)>0 else "—")

    tab1,tab2,tab3 = st.tabs(["📅 Season-by-Season","🏙️ City Performance","📋 Recent Matches"])
    with tab1:
        sea_w=t_wins.groupby("season").size().reset_index(name="wins")
        sea_p=t_all.groupby("season").size().reset_index(name="played")
        sea_m=sea_p.merge(sea_w,on="season",how="left").fillna(0)
        sea_m["wins"]=sea_m["wins"].astype(int)
        sea_m["win_pct"]=(sea_m["wins"]/sea_m["played"]*100).round(1)
        st.dataframe(sea_m.rename(columns={"season":"Season","played":"Played","wins":"Wins","win_pct":"Win%"}),
            use_container_width=True,hide_index=True)
    with tab2:
        city_w=t_wins.groupby("city").size().reset_index(name="wins").sort_values("wins",ascending=False)
        st.dataframe(city_w.rename(columns={"city":"City","wins":"Wins"}),
            use_container_width=True,hide_index=True)
    with tab3:
        recent=t_wins.sort_values("match_date",ascending=False).head(15)
        st.dataframe(recent[["match_date","city","team1","team2","match_winner","win_by_runs","win_by_wickets"]],
            use_container_width=True,hide_index=True)

    section_hdr("Win Margin Distribution","📏")
    wm = win_margin_dist()
    team_wm = wm[wm["match_winner"]==team_sel]
    runs_only = team_wm[team_wm["margin_type"]=="runs"]["margin"]
    wkts_only = team_wm[team_wm["margin_type"]=="wickets"]["margin"]
    c1,c2=st.columns(2)
    with c1:
        st.markdown(f"**Wins by runs** (n={len(runs_only)})")
        if len(runs_only):
            bins=pd.cut(runs_only,bins=[0,10,20,30,50,100],labels=["1-10","11-20","21-30","31-50","50+"])
            st.bar_chart(bins.value_counts().sort_index(),height=200,color="#6366f1")
    with c2:
        st.markdown(f"**Wins by wickets** (n={len(wkts_only)})")
        if len(wkts_only):
            st.bar_chart(wkts_only.value_counts().sort_index(),height=200,color="#10b981")


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — HEAD TO HEAD
# ══════════════════════════════════════════════════════════════
elif page == "h2h":
    page_hero("⚔️","Head to Head","Detailed rivalry record between any two franchises")
    df    = get_matches()
    teams = get_teams()
    logo_map  = dict(zip(teams["team_name"],teams["image_url"]))
    short_map = dict(zip(teams["team_name"],teams["team_name_short"]))
    all_teams = sorted(pd.concat([df["team1"],df["team2"]]).dropna().unique().tolist())

    c1,c2 = st.columns(2)
    with c1: t1 = st.selectbox("Team 1",all_teams, index=all_teams.index("Mumbai Indians") if "Mumbai Indians" in all_teams else 0)
    with c2: t2 = st.selectbox("Team 2",[t for t in all_teams if t!=t1], index=[t for t in all_teams if t!=t1].index("Chennai Super Kings") if "Chennai Super Kings" in all_teams else 0)

    h = head_to_head(t1,t2)

    section_hdr("Head-to-Head Record","📊")
    if h["total"]==0:
        st.info(f"No matches found between {t1} and {t2}."); st.stop()

    hL,hM,hR = st.columns([5,2,5])
    t1pct = h["t1_wins"]/h["total"]*100 if h["total"]>0 else 0
    t2pct = h["t2_wins"]/h["total"]*100 if h["total"]>0 else 0
    with hL:
        st.markdown(team_prob_card(t1,t1pct,logo_map.get(t1),h["t1_wins"]>h["t2_wins"],short_map.get(t1,t1[:3])),unsafe_allow_html=True)
        st.progress(int(t1pct))
    with hM:
        st.markdown(f"""
        <div style="text-align:center;padding:1rem .5rem;">
          <div style="font-size:.75rem;color:#94a3b8!important;font-weight:600;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.5rem;">Total Meetings</div>
          <div style="font-size:2.8rem;font-weight:900;color:#a78bfa!important;">{h['total']}</div>
        </div>""",unsafe_allow_html=True)
    with hR:
        st.markdown(team_prob_card(t2,t2pct,logo_map.get(t2),h["t2_wins"]>h["t1_wins"],short_map.get(t2,t2[:3])),unsafe_allow_html=True)
        st.progress(int(t2pct))

    section_hdr("Match History","📋")
    h_df = h["matches"].copy()
    h_df["match_date"] = pd.to_datetime(h_df["match_date"],errors="coerce").dt.strftime("%d %b %Y")
    h_df["result"] = h_df.apply(
        lambda r: f"✅ {r['match_winner']}" + (f" by {int(r['win_by_runs'])} runs" if pd.notna(r['win_by_runs']) and r['win_by_runs']>0 else f" by {int(r['win_by_wickets'])} wkts" if pd.notna(r['win_by_wickets']) else ""),axis=1)
    st.dataframe(h_df[["match_date","season","city","result"]].rename(
        columns={"match_date":"Date","season":"Season","city":"City","result":"Result"}),
        use_container_width=True,hide_index=True)

    section_hdr("Season-Wise Dominance","📅")
    by_season = h["by_season"]
    if len(by_season):
        agg = []
        for _,row in by_season.iterrows():
            winners=row["match_winner"]
            if isinstance(winners,list):
                t1c=winners.count(t1); t2c=winners.count(t2)
            else:
                t1c=1 if winners==t1 else 0; t2c=1 if winners==t2 else 0
            agg.append({"Season":row["season"],f"{t1[:12]}":t1c,f"{t2[:12]}":t2c})
        agg_df=pd.DataFrame(agg).sort_values("Season")
        st.dataframe(agg_df,use_container_width=True,hide_index=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — VENUE ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "venue":
    page_hero("🏟️","Venue Analysis","Chase vs defend success rates across all IPL grounds")
    vs = venue_stats()
    cs = city_performance()

    section_hdr("Overview Metrics","📊")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Venues",len(vs))
    c2.metric("Cities Used",len(cs))
    c3.metric("Avg Chase Win%",f"{vs['chase_pct'].mean():.1f}%")
    c4.metric("Best Chase City",cs.sort_values("chase_pct",ascending=False).iloc[0]["city"])

    section_hdr("Chase vs Defend by Venue","🏟️")
    top_n = st.slider("Show top N venues by match count",5,30,15)
    top_v = vs.head(top_n)
    chart_df=top_v.set_index("venue")[["chase_wins","defend_wins"]].rename(
        columns={"chase_wins":"Chase Wins","defend_wins":"Defend Wins"})
    st.bar_chart(chart_df,height=320,color=["#10b981","#6366f1"])

    section_hdr("Venue Details Table","📋")
    st.dataframe(vs.head(30).rename(columns={
        "venue":"Venue","city":"City","total":"Matches",
        "chase_wins":"Chase Wins","defend_wins":"Defend Wins","chase_pct":"Chase Win%"}),
        use_container_width=True,hide_index=True)

    section_hdr("City-Level Chase Analysis","🏙️")
    st.bar_chart(cs.set_index("city")[["chase_pct"]].rename(columns={"chase_pct":"Chase Win%"}).head(15),
        height=260,color="#10b981")

    section_hdr("Best Chasing Cities","✅")
    best_chase = cs.sort_values("chase_pct",ascending=False).head(10)
    for _,row in best_chase.iterrows():
        col1,col2,col3=st.columns([4,2,1])
        col1.markdown(f"<p style='margin:.15rem 0;font-weight:600;color:#f1f5f9!important;'>🏙️ {row['city']}</p>",unsafe_allow_html=True)
        col2.markdown(f"<p style='color:#94a3b8!important;font-size:.82rem;'>{row['chase_wins']} of {row['total']} chases won</p>",unsafe_allow_html=True)
        col3.markdown(f"<p style='color:#10b981!important;font-weight:700;'>{row['chase_pct']}%</p>",unsafe_allow_html=True)
        st.progress(int(row["chase_pct"]))


# ══════════════════════════════════════════════════════════════
#  PAGE 6 — TOSS INTELLIGENCE
# ══════════════════════════════════════════════════════════════
elif page == "toss":
    page_hero("🎲","Toss Intelligence","How the coin flip affects match outcomes across cities & seasons")
    df   = get_matches()
    wins = df[df["result"]=="win"]

    section_hdr("Overall Toss Statistics","📊")
    total_matches   = len(wins)
    toss_match_wins = wins[wins["toss_winner"]==wins["match_winner"]]
    toss_pct        = len(toss_match_wins)/total_matches*100

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Toss Winner Wins",f"{toss_pct:.1f}%")
    m2.metric("Toss Winner Loses",f"{100-toss_pct:.1f}%")
    field_wins = wins[(wins["toss_decision"]=="field")&(wins["toss_winner"]==wins["match_winner"])]
    bat_wins   = wins[(wins["toss_decision"]=="bat")&(wins["toss_winner"]==wins["match_winner"])]
    field_total= wins[wins["toss_decision"]=="field"]
    bat_total  = wins[wins["toss_decision"]=="bat"]
    m3.metric("Chose Field & Won",f"{len(field_wins)/len(field_total)*100:.1f}%")
    m4.metric("Chose Bat & Won",f"{len(bat_wins)/len(bat_total)*100:.1f}%" if len(bat_total)>0 else "—")

    info_box("💡 Historically, teams that choose to <strong>field first</strong> have a slightly higher win rate — "
             "suggesting dew factor and pitch knowledge advantage in second innings.", "#10b981")

    section_hdr("Toss Decision Breakdown by Season","📅")
    dec_sea = df.groupby(["season","toss_decision"]).size().unstack(fill_value=0)
    dec_sea.columns = [f"Chose {c.capitalize()}" for c in dec_sea.columns]
    st.bar_chart(dec_sea,height=280,color=["#6366f1","#10b981"])

    section_hdr("Team Toss Win Rates","🏅")
    team_toss_total = df.groupby("toss_winner").size().reset_index(name="toss_total")
    team_toss_win   = wins[wins["toss_winner"]==wins["match_winner"]].groupby("match_winner").size().reset_index(name="toss_wins")
    toss_perf = team_toss_total.merge(team_toss_win,left_on="toss_winner",right_on="match_winner",how="left")
    toss_perf["toss_wins"]=toss_perf["toss_wins"].fillna(0).astype(int)
    toss_perf["toss_win_rate"]=(toss_perf["toss_wins"]/toss_perf["toss_total"]*100).round(1)
    st.dataframe(toss_perf[["toss_winner","toss_total","toss_wins","toss_win_rate"]].rename(
        columns={"toss_winner":"Team","toss_total":"Tosses","toss_wins":"Toss→Win","toss_win_rate":"Toss Win%"})
        .sort_values("Toss Win%",ascending=False),use_container_width=True,hide_index=True)

    section_hdr("City Toss Advantage","🏙️")
    city_toss = df.groupby("city").apply(
        lambda g: pd.Series({
            "matches": len(g),
            "toss_win_rate": (
                (g[g["result"]=="win"].apply(lambda r: 1 if r["toss_winner"]==r["match_winner"] else 0, axis=1).sum())
                / len(g) * 100
            ) if len(g)>0 else 0
        })
    ).reset_index().sort_values("toss_win_rate",ascending=False)
    city_toss=city_toss[city_toss["matches"]>=10]
    st.dataframe(city_toss.rename(columns={"city":"City","matches":"Matches","toss_win_rate":"Toss Win%"})
        .head(20),use_container_width=True,hide_index=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 7 — PLAYER PROFILES
# ══════════════════════════════════════════════════════════════
elif page == "players":
    page_hero("👤","Player Profiles","Search & explore 772 IPL player records")
    players = get_players()
    pom     = player_of_match_leaders()

    section_hdr("Dataset Overview","📋")
    p1,p2,p3,p4 = st.columns(4)
    p1.metric("Total Players",f"{len(players):,}")
    p2.metric("Batting Styles",players["bat_style"].nunique())
    p3.metric("Bowling Styles",players["bowl_style"].nunique())
    p4.metric("Wicketkeepers",int((players["field_pos"]=="Wicketkeeper").sum()))

    section_hdr("Player of the Match Leaders","🌟")
    top_pom = pom[pom["player_name"].notna()].head(10)
    for i,(_,row) in enumerate(top_pom.iterrows()):
        cols=st.columns([.5,4,1.5])
        with cols[0]: st.markdown(f"<p style='font-size:1.1rem;margin:.1rem 0;color:#a78bfa!important;font-weight:700;'>#{i+1}</p>",unsafe_allow_html=True)
        with cols[1]:
            name = row.get("player_full_name") or row.get("player_name","Unknown")
            bat  = row.get("bat_style","") or ""; bowl = row.get("bowl_style","") or ""
            st.markdown(f"<p style='margin:.1rem 0;font-weight:700;color:#f1f5f9!important;'>{name}</p>",unsafe_allow_html=True)
            st.markdown(" ".join([stat_pill(bat,"","#6366f1") if bat else "",
                                   stat_pill(bowl,"","#10b981") if bowl else ""]),unsafe_allow_html=True)
        with cols[2]:
            cnt=int(row["pom_count"])
            st.markdown(f"<p style='text-align:right;font-size:1.15rem;font-weight:800;color:#f59e0b!important;margin:.1rem 0;'>🏅 {cnt}</p>",unsafe_allow_html=True)
        st.progress(int(cnt/top_pom["pom_count"].max()*100))

    section_hdr("Search & Filter Players","🔍")
    sf1,sf2,sf3=st.columns(3)
    with sf1: search=st.text_input("Search by name",placeholder="e.g. Rohit, Kohli")
    with sf2:
        bat_opts=["All"]+sorted(players["bat_style"].dropna().unique().tolist())
        sel_bat=st.selectbox("Batting Style",bat_opts)
    with sf3:
        bowl_opts=["All"]+sorted(players["bowl_style"].dropna().unique().tolist())
        sel_bowl=st.selectbox("Bowling Style",bowl_opts)

    fp=players.copy()
    if search: fp=fp[fp["player_name"].str.contains(search,case=False,na=False)|fp["player_full_name"].str.contains(search,case=False,na=False)]
    if sel_bat!="All": fp=fp[fp["bat_style"]==sel_bat]
    if sel_bowl!="All": fp=fp[fp["bowl_style"]==sel_bowl]

    st.markdown(f"<p style='color:#94a3b8!important;font-size:.82rem;'><strong style='color:#a78bfa!important;'>{len(fp):,}</strong> players found</p>",unsafe_allow_html=True)
    show_c=["player_id","player_name","player_full_name","bat_style","bowl_style","field_pos"]
    st.dataframe(fp[[c for c in show_c if c in fp.columns]].reset_index(drop=True),
        use_container_width=True,height=380)

    section_hdr("Style Distribution","📊")
    t1,t2=st.tabs(["🏏 Batting Styles","🎳 Bowling Styles"])
    with t1:
        bc=players["bat_style"].value_counts().reset_index(); bc.columns=["Style","Players"]
        st.bar_chart(bc.set_index("Style"),height=200,color="#6366f1")
    with t2:
        bwc=players["bowl_style"].value_counts().reset_index(); bwc.columns=["Style","Players"]
        st.bar_chart(bwc.set_index("Style").head(12),height=240,color="#10b981")


# ══════════════════════════════════════════════════════════════
#  PAGE 8 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "data":
    page_hero("🗂️","Data Explorer","Inspect all registered datasets — add any new CSV in 3 steps")

    section_hdr("Registered Datasets","📦")
    info_df=registry_info()
    st.dataframe(info_df,use_container_width=True,hide_index=True)

    section_hdr("➕ How to Add a New CSV","📝")
    st.markdown("""
    <div style="background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.25);
        border-radius:14px;padding:1.2rem 1.5rem;font-size:.87rem;line-height:1.9;">
      <strong style="color:#a78bfa!important;">3-step process — no code changes needed anywhere else:</strong><br/>
      <strong style="color:#10b981!important;">Step 1 —</strong> Drop <code>.csv</code> into the <code>data/</code> folder<br/>
      <strong style="color:#10b981!important;">Step 2 —</strong> Add one entry to <code>data_loader.py → DATA_REGISTRY</code>:<br/>
      <pre style="background:rgba(0,0,0,.3);border-radius:8px;padding:.7rem;color:#e2e8f0!important;font-size:.8rem;margin:.4rem 0;">"my_key": {{
    "filename"   : "my_file.csv",
    "encoding"   : "utf-8",
    "description": "What this dataset contains",
}}</pre>
      <strong style="color:#10b981!important;">Step 3 —</strong> Use it anywhere in the app:<br/>
      <pre style="background:rgba(0,0,0,.3);border-radius:8px;padding:.7rem;color:#e2e8f0!important;font-size:.8rem;margin:.4rem 0;">from data_loader import load_data
df = load_data("my_key")  # auto-cached + auto-cleaned ✅</pre>
      Automatic Streamlit caching, whitespace cleaning, and error messages included!
    </div>""",unsafe_allow_html=True)

    section_hdr("Live Dataset Preview","👁️")
    ds_choice=st.selectbox("Choose dataset",list(DATA_REGISTRY))
    try:
        pv=load_data(ds_choice)
        st.markdown(f"<p style='color:#94a3b8!important;font-size:.8rem;'>Shape: "
                    f"<strong style='color:#a78bfa!important;'>{pv.shape[0]:,} rows × {pv.shape[1]} cols</strong></p>",
                    unsafe_allow_html=True)
        t1,t2,t3=st.tabs(["📄 Preview","ℹ️ Schema","📊 Stats"])
        with t1: st.dataframe(pv.head(50),use_container_width=True,height=380)
        with t2:
            schema=pd.DataFrame({"Column":pv.columns,"Type":pv.dtypes.astype(str).values,
                "Non-Null":pv.notna().sum().values,"Null":pv.isna().sum().values,
                "Unique":[pv[c].nunique() for c in pv.columns]})
            st.dataframe(schema,use_container_width=True,hide_index=True)
        with t3:
            num=pv.select_dtypes(include="number")
            if not num.empty: st.dataframe(num.describe().round(2),use_container_width=True)
            else: st.info("No numeric columns in this dataset.")
    except Exception as e:
        st.error(f"Could not load '{ds_choice}': {e}")
