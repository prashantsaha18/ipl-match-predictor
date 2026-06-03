"""
ui_components.py — Shared HTML/CSS building blocks
"""
import streamlit as st

GLOBAL_CSS = """
<style>
/* ── App Background ─────────────────────────────────── */
.stApp,.main{background:linear-gradient(135deg,#0a0f1e 0%,#111827 45%,#0d1f12 100%)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.4rem 2rem 3rem!important;}

/* ── Typography ─────────────────────────────────────── */
h1,h2,h3,p,label,span,div,li{font-family:'Inter','Segoe UI',sans-serif!important;color:#f1f5f9!important;}

/* ── Sidebar ────────────────────────────────────────── */
[data-testid="stSidebar"]{
  background:rgba(8,12,24,.98)!important;
  border-right:1px solid rgba(255,255,255,.07)!important;
}
[data-testid="stSidebar"] .block-container{padding:1.2rem .8rem!important;}

/* ── Select / Number / Text inputs ─────────────────── */
.stSelectbox>div>div,
.stNumberInput>div>div>input,
.stTextInput>div>div>input{
  background:rgba(255,255,255,.06)!important;
  border:1px solid rgba(255,255,255,.13)!important;
  border-radius:10px!important;color:#f1f5f9!important;
}

/* ── Buttons ────────────────────────────────────────── */
.stButton>button{
  background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;
  border:none!important;border-radius:12px!important;
  color:white!important;font-weight:700!important;
  font-size:.98rem!important;padding:.65rem 1.6rem!important;
  box-shadow:0 4px 18px rgba(99,102,241,.4)!important;
  transition:all .2s!important;width:100%!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;
  box-shadow:0 8px 26px rgba(99,102,241,.55)!important;}

/* ── Progress ───────────────────────────────────────── */
.stProgress>div>div{background:linear-gradient(90deg,#6366f1,#10b981)!important;border-radius:99px!important;}
.stProgress{background:rgba(255,255,255,.07)!important;border-radius:99px!important;}

/* ── Metrics ────────────────────────────────────────── */
[data-testid="stMetric"]{
  background:rgba(255,255,255,.04)!important;
  border:1px solid rgba(255,255,255,.08)!important;
  border-radius:14px!important;padding:.9rem!important;
}
[data-testid="stMetricValue"]{color:#a78bfa!important;font-weight:800!important;}
[data-testid="stMetricLabel"]{color:#94a3b8!important;}
[data-testid="stMetricDelta"]{font-size:.8rem!important;}

/* ── Tabs ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,.03)!important;border-radius:12px!important;gap:4px!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;border-radius:8px!important;color:#94a3b8!important;font-weight:600!important;}
.stTabs [aria-selected="true"]{background:rgba(99,102,241,.25)!important;color:#a78bfa!important;}

/* ── Dataframe ──────────────────────────────────────── */
[data-testid="stDataFrame"]{border-radius:12px!important;overflow:hidden;}
.stDataFrame th{background:rgba(99,102,241,.18)!important;}

/* ── Expander ───────────────────────────────────────── */
.streamlit-expanderHeader{background:rgba(255,255,255,.04)!important;border-radius:10px!important;}

hr{border-color:rgba(255,255,255,.07)!important;}
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_hero(emoji: str, title: str, subtitle: str):
    st.markdown(f"""
    <div style="text-align:center;padding:2rem 1rem 1.5rem;
        background:rgba(255,255,255,.02);border-radius:20px;
        border:1px solid rgba(255,255,255,.06);margin-bottom:1.8rem;">
      <div style="font-size:3.2rem;margin-bottom:.3rem;">{emoji}</div>
      <h1 style="font-size:2.2rem!important;font-weight:800!important;margin:0!important;
          background:linear-gradient(135deg,#6366f1,#a78bfa,#10b981);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{title}</h1>
      <p style="color:#94a3b8!important;font-size:.95rem;margin-top:.45rem;">{subtitle}</p>
    </div>""", unsafe_allow_html=True)


def section_hdr(text: str, icon: str = ""):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:.55rem;
        margin:1.4rem 0 .75rem;padding-bottom:.45rem;
        border-bottom:1px solid rgba(255,255,255,.08);">
      <span style="font-size:1.15rem;">{icon}</span>
      <span style="font-size:1rem;font-weight:700;color:#f1f5f9!important;">{text}</span>
    </div>""", unsafe_allow_html=True)


def glass_card(label: str, value: str, icon: str = "", color: str = "#6366f1") -> str:
    return f"""
    <div style="background:rgba(255,255,255,.04);backdrop-filter:blur(12px);
        border-radius:16px;border:1px solid rgba(255,255,255,.08);
        padding:1rem 1.2rem;margin:.2rem 0;">
      <div style="font-size:.68rem;font-weight:600;letter-spacing:.08em;
          color:#94a3b8!important;text-transform:uppercase;margin-bottom:.3rem;">{icon} {label}</div>
      <div style="font-size:1.85rem;font-weight:800;color:{color}!important;line-height:1.1;">{value}</div>
    </div>"""


def team_prob_card(name: str, prob: float, logo_url, leading: bool, short: str = "") -> str:
    border = "#10b981" if leading else "#6366f1"
    glow   = "0 0 26px rgba(16,185,129,.35)" if leading else "none"
    badge  = ('<span style="background:#10b981;color:#fff;font-size:.6rem;padding:2px 7px;'
              'border-radius:99px;font-weight:700;margin-left:6px;">▲ LEADING</span>') if leading else ""
    img    = (f'<img src="{logo_url}" style="width:70px;height:70px;object-fit:contain;'
              f'margin-bottom:.5rem;filter:drop-shadow(0 2px 8px rgba(0,0,0,.5));"/>'
              if logo_url else
              f'<div style="width:64px;height:64px;border-radius:50%;'
              f'background:linear-gradient(135deg,{border},#1e1b4b);'
              f'display:flex;align-items:center;justify-content:center;'
              f'font-size:1.3rem;color:#fff!important;margin:0 auto .5rem;">'
              f'{short[:2].upper()}</div>')
    pct_color = "#10b981" if leading else "#f1f5f9"
    return f"""
    <div style="background:rgba(255,255,255,.04);backdrop-filter:blur(14px);
        border-radius:20px;border:2px solid {border};padding:1.6rem 1rem;
        text-align:center;box-shadow:{glow};">
      {img}
      <div style="font-size:.95rem;font-weight:700;color:#f1f5f9!important;">{name}{badge}</div>
      <div style="font-size:3rem;font-weight:900;color:{pct_color}!important;line-height:1.1;">{prob:.1f}%</div>
      <div style="color:#94a3b8!important;font-size:.75rem;">Win Probability</div>
    </div>"""


def dominance_banner(chasing: str, defending: str, cp: float, dp: float) -> str:
    if   cp >= 70: c, msg = "#10b981", f"🟢  {chasing} are DOMINANT — Chase is ON!"
    elif dp >= 70: c, msg = "#ef4444", f"🔴  {defending} are in FULL CONTROL!"
    elif cp >= 55: c, msg = "#10b981", f"⚡  {chasing} have the EDGE"
    elif dp >= 55: c, msg = "#f59e0b", f"⚡  {defending} hold a slight ADVANTAGE"
    else:          c, msg = "#6366f1", "⚔️  Dead Heat — Could go EITHER WAY!"
    return f"""<div style="margin-top:1.4rem;padding:1.2rem 2rem;
        background:linear-gradient(135deg,{c}22,{c}0d);border:1.5px solid {c};
        border-radius:16px;text-align:center;box-shadow:0 0 28px {c}28;">
      <span style="font-size:1.3rem;font-weight:800;color:{c}!important;">{msg}</span>
    </div>"""


def stat_pill(label: str, value: str, color: str = "#6366f1") -> str:
    return (f'<span style="background:{color}22;border:1px solid {color}55;'
            f'border-radius:99px;padding:3px 12px;font-size:.78rem;font-weight:600;'
            f'color:{color}!important;margin:2px;">{label}: {value}</span>')


def info_box(text: str, color: str = "#6366f1"):
    st.markdown(
        f'<div style="background:{color}0f;border:1px solid {color}33;'
        f'border-radius:12px;padding:.9rem 1.2rem;font-size:.88rem;'
        f'color:#e2e8f0!important;margin:.5rem 0;">{text}</div>',
        unsafe_allow_html=True
    )
