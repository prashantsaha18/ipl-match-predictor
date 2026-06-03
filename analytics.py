"""
analytics.py — Pre-computed analytics from match + player data
"""
import pandas as pd
import numpy as np
import streamlit as st
from data_loader import load_data


@st.cache_data(show_spinner=False)
def get_matches() -> pd.DataFrame:
    df = load_data("matches")
    df["match_date"] = pd.to_datetime(df["match_date"], dayfirst=True, errors="coerce")
    df["year"] = df["match_date"].dt.year
    return df


@st.cache_data(show_spinner=False)
def get_teams():
    return load_data("teams")


@st.cache_data(show_spinner=False)
def get_players():
    return load_data("players")


@st.cache_data(show_spinner=False)
def get_ui_lists():
    df      = get_matches()
    teams   = get_teams()
    cities  = sorted(df["city"].dropna().unique().tolist())
    all_t   = sorted(pd.concat([df["team1"], df["team2"]]).dropna().unique().tolist())
    logo    = dict(zip(teams["team_name"], teams["image_url"]))
    short   = dict(zip(teams["team_name"], teams["team_name_short"]))
    return cities, all_t, logo, short


@st.cache_data(show_spinner=False)
def team_all_time_stats() -> pd.DataFrame:
    df   = get_matches()
    wins = df[df["result"] == "win"]
    played = pd.concat([df["team1"], df["team2"]]).value_counts().reset_index()
    played.columns = ["team", "played"]
    won = wins["match_winner"].value_counts().reset_index()
    won.columns = ["team", "wins"]
    stats = played.merge(won, on="team", how="left")
    stats["wins"]      = stats["wins"].fillna(0).astype(int)
    stats["losses"]    = stats["played"] - stats["wins"]
    stats["win_pct"]   = (stats["wins"] / stats["played"] * 100).round(1)
    # toss advantage
    toss_wins = wins[wins["toss_winner"] == wins["match_winner"]]
    toss_cnt  = toss_wins["match_winner"].value_counts().reset_index()
    toss_cnt.columns = ["team", "toss_match_wins"]
    stats = stats.merge(toss_cnt, on="team", how="left")
    stats["toss_match_wins"] = stats["toss_match_wins"].fillna(0).astype(int)
    return stats.sort_values("win_pct", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def season_summary() -> pd.DataFrame:
    df   = get_matches()
    wins = df[df["result"] == "win"]
    per_season = df.groupby("season").agg(
        total_matches=("match_id", "count"),
        cities_used=("city", "nunique"),
    ).reset_index()
    season_wins = wins.groupby(["season", "match_winner"]).size().reset_index(name="w")
    champions   = season_wins.sort_values("w", ascending=False).groupby("season").first().reset_index()
    champions.rename(columns={"match_winner": "top_team", "w": "season_wins"}, inplace=True)
    return per_season.merge(champions, on="season", how="left").sort_values("season")


@st.cache_data(show_spinner=False)
def head_to_head(t1: str, t2: str) -> dict:
    df   = get_matches()
    wins = df[df["result"] == "win"]
    h2h  = wins[
        ((wins["team1"] == t1) & (wins["team2"] == t2)) |
        ((wins["team1"] == t2) & (wins["team2"] == t1))
    ].copy()
    t1w   = (h2h["match_winner"] == t1).sum()
    t2w   = (h2h["match_winner"] == t2).sum()
    total = len(h2h)
    by_season = h2h.groupby("season")["match_winner"].apply(list).reset_index()
    return {
        "t1": t1, "t2": t2,
        "t1_wins": int(t1w), "t2_wins": int(t2w), "total": total,
        "matches": h2h[["season", "match_date", "city", "venue",
                         "match_winner", "win_by_runs", "win_by_wickets"]].sort_values("match_date", ascending=False),
        "by_season": by_season,
    }


@st.cache_data(show_spinner=False)
def venue_stats() -> pd.DataFrame:
    df   = get_matches()
    wins = df[df["result"] == "win"]
    vt   = df.groupby("venue").agg(
        total=("match_id", "count"),
        city=("city", "first"),
    ).reset_index()
    # chasing wins = win_by_wickets
    chase_w = wins[wins["win_by_wickets"].notna() & (wins["win_by_wickets"] > 0)]
    cv = chase_w["venue"].value_counts().reset_index()
    cv.columns = ["venue", "chase_wins"]
    vt = vt.merge(cv, on="venue", how="left")
    vt["chase_wins"]  = vt["chase_wins"].fillna(0).astype(int)
    vt["defend_wins"] = vt["total"] - vt["chase_wins"]
    vt["chase_pct"]   = (vt["chase_wins"] / vt["total"] * 100).round(1)
    return vt.sort_values("total", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def toss_analysis() -> pd.DataFrame:
    df   = get_matches()
    wins = df[df["result"] == "win"]
    total_toss = df.groupby("toss_decision").size().reset_index(name="total_toss")
    toss_won   = wins[wins["toss_winner"] == wins["match_winner"]]
    toss_dec   = toss_won.groupby("toss_decision").size().reset_index(name="toss_and_match_wins")
    merged = total_toss.merge(toss_dec, on="toss_decision", how="left")
    merged["win_pct"] = (merged["toss_and_match_wins"] / merged["total_toss"] * 100).round(1)
    return merged


@st.cache_data(show_spinner=False)
def win_margin_dist() -> pd.DataFrame:
    df   = get_matches()
    wins = df[df["result"] == "win"]
    by_runs = wins[wins["win_by_runs"] > 0][["match_winner","win_by_runs","season","city"]].copy()
    by_runs["margin_type"] = "runs"
    by_runs.rename(columns={"win_by_runs": "margin"}, inplace=True)
    by_wkt = wins[wins["win_by_wickets"] > 0][["match_winner","win_by_wickets","season","city"]].copy()
    by_wkt["margin_type"] = "wickets"
    by_wkt.rename(columns={"win_by_wickets": "margin"}, inplace=True)
    return pd.concat([by_runs, by_wkt], ignore_index=True)


@st.cache_data(show_spinner=False)
def player_of_match_leaders() -> pd.DataFrame:
    df      = get_matches()
    players = get_players()
    wins    = df[df["result"] == "win"].copy()
    wins["player_of_match"] = pd.to_numeric(wins["player_of_match"], errors="coerce")
    pom = wins["player_of_match"].value_counts().reset_index()
    pom.columns = ["player_id", "pom_count"]
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce")
    merged = pom.merge(players[["player_id","player_name","player_full_name",
                                 "bat_style","bowl_style","player_image"]],
                       on="player_id", how="left")
    return merged.head(25)


@st.cache_data(show_spinner=False)
def city_performance() -> pd.DataFrame:
    df   = get_matches()
    wins = df[df["result"] == "win"]
    city_total = df.groupby("city").size().reset_index(name="total")
    chase_wins = wins[wins["win_by_wickets"] > 0]["city"].value_counts().reset_index()
    chase_wins.columns = ["city", "chase_wins"]
    ct = city_total.merge(chase_wins, on="city", how="left")
    ct["chase_wins"]  = ct["chase_wins"].fillna(0).astype(int)
    ct["defend_wins"] = ct["total"] - ct["chase_wins"]
    ct["chase_pct"]   = (ct["chase_wins"] / ct["total"] * 100).round(1)
    return ct.sort_values("total", ascending=False).reset_index(drop=True)
