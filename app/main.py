import streamlit as st
import pandas as pd
import os
import sys
import requests

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_and_preprocess_data
from src.content_engine import ContentBasedRecommender
from src.collaborative_engine import CollaborativeRecommender
from src.hybrid_engine import HybridRecommender
from src.evaluation import run_full_evaluation

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Genre colour map ─────────────────────────────────────────────────────────
GENRE_COLORS = {
    "Action":     "#ef4444", "Adventure":  "#f97316", "Animation": "#eab308",
    "Children":   "#84cc16", "Comedy":     "#22c55e", "Crime":     "#14b8a6",
    "Documentary":"#06b6d4", "Drama":      "#3b82f6", "Fantasy":   "#8b5cf6",
    "Horror":     "#dc2626", "Musical":    "#ec4899", "Mystery":   "#6366f1",
    "Romance":    "#f43f5e", "Sci-Fi":     "#0ea5e9", "Thriller":  "#f59e0b",
    "War":        "#78716c", "Western":    "#d97706",
}

def genre_badge(genre):
    color = GENRE_COLORS.get(genre.strip(), "#6b7280")
    return f'<span style="background:{color}22;color:{color};border:1px solid {color}55;padding:2px 9px;border-radius:50px;font-size:0.7rem;font-weight:600;margin:2px;display:inline-block;">{genre.strip()}</span>'

def genre_badges(genres_str):
    return "".join(genre_badge(g) for g in str(genres_str).replace("|", ",").split(","))

def score_bar(score, max_score=1.0, color="#a78bfa"):
    pct = min(100, max(0, (score / max_score) * 100)) if max_score else 0
    return f"""
    <div style="background:rgba(255,255,255,0.07);border-radius:50px;height:5px;width:100%;margin-top:6px;">
        <div style="background:{color};height:5px;border-radius:50px;width:{pct:.1f}%;transition:width 0.6s ease;"></div>
    </div>"""

# ─── Master CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Space Grotesk', 'Inter', sans-serif; }

/* ── Keyframe Animations ── */
@keyframes pulse-glow {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50%       { opacity: 1;   transform: scale(1.05); }
}
@keyframes gradient-shift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes shimmer {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-8px); }
}
@keyframes border-glow {
    0%, 100% { border-color: rgba(167,139,250,0.2); box-shadow: 0 0 0px rgba(167,139,250,0); }
    50%       { border-color: rgba(167,139,250,0.6); box-shadow: 0 0 20px rgba(167,139,250,0.15); }
}
@keyframes fade-up {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes spin-slow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

/* ── Cinematic Background ── */
.stApp {
    background: #060610;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 60% at 15% 5%,  rgba(124,58,237,0.22) 0%, transparent 55%),
        radial-gradient(ellipse 50% 50% at 85% 95%, rgba(59,130,246,0.18) 0%, transparent 55%),
        radial-gradient(ellipse 40% 30% at 60% 50%, rgba(236,72,153,0.06) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}
.stApp > * { position: relative; z-index: 1; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #7c3aed, #3b82f6);
    border-radius: 10px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(8,6,20,0.98) 0%, rgba(12,8,28,0.98) 100%) !important;
    border-right: 1px solid rgba(124,58,237,0.15) !important;
    backdrop-filter: blur(30px);
    box-shadow: 4px 0 30px rgba(0,0,0,0.5);
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] .stSlider > label {
    color: rgba(167,139,250,0.6) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

/* ── Sidebar selectbox / radio styling ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(167,139,250,0.2) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
    border-color: rgba(167,139,250,0.5) !important;
}

/* ── Radio buttons ── */
[data-testid="stSidebar"] [data-baseweb="radio"] label {
    padding: 8px 12px !important;
    border-radius: 8px !important;
    transition: background 0.2s ease !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
    background: rgba(167,139,250,0.08) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
[data-testid="stDecoration"] { display: none; }

/* ── Buttons ── */
.stButton > button {
    position: relative !important;
    background: linear-gradient(135deg, #7c3aed, #4f46e5, #2563eb) !important;
    background-size: 200% 200% !important;
    animation: gradient-shift 4s ease infinite !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 0.7rem 1.4rem !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    transition: all 0.25s ease !important;
    overflow: hidden !important;
}
.stButton > button::after {
    content: '' !important;
    position: absolute !important;
    top: 0; left: -100% !important;
    width: 100%; height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent) !important;
    animation: shimmer 2.5s infinite !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 35px rgba(124,58,237,0.6), 0 0 0 1px rgba(167,139,250,0.3) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #a78bfa !important; }

/* ── HR ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(167,139,250,0.3), transparent) !important;
    margin: 1.2rem 0 !important;
}

/* ── Slider thumb ── */
[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    box-shadow: 0 0 10px rgba(124,58,237,0.5) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Data & Models ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    return load_and_preprocess_data(
        os.path.join(base, "..", "data", "raw", "movies.csv"),
        os.path.join(base, "..", "data", "raw", "ratings.csv"),
    )

@st.cache_resource
def train_models(movies, train):
    cb = ContentBasedRecommender(); cb.fit(movies)
    cf = CollaborativeRecommender(factors=50); cf.fit(train)
    hy = HybridRecommender(cb, cf, content_weight=0.5)
    return cb, cf, hy

@st.cache_data
def evaluate_models(_model, test):
    return run_full_evaluation(_model, test)

@st.cache_data(ttl=86400)
def get_movie_poster_v6(title: str, api_key: str) -> str:
    """Fetch movie poster and return as base64 data URI for inline embedding."""
    if not api_key:
        return ""
    try:
        import base64
        import re
        # Remove everything after the first parenthesis (strips years AND foreign/alternate titles)
        clean = title.split("(")[0].strip()
        # Fix "Movie, The" or "Movie, A" formatting from MovieLens
        match = re.search(r'^(.*),\s*(The|A|An)$', clean, flags=re.IGNORECASE)
        if match:
            clean = f"{match.group(2)} {match.group(1)}"
            
        # Step 1: get the poster URL from OMDb via exact match
        r = requests.get("http://www.omdbapi.com/", params={"t": clean, "apikey": api_key}, timeout=4)
        data = r.json()
        poster_url = data.get("Poster", "N/A")
        
        # Fallback to search match if exact match fails
        if data.get("Response") != "True" or poster_url == "N/A":
            simpler = clean.split(":")[0].strip()
            # OMDb often uses roman numerals instead of "2" or "Part II"
            simpler = simpler.replace("Part II", "II").replace("Part III", "III").replace("Part 2", "II")
            simpler = re.sub(r' 2$', ' II', simpler)
            simpler = re.sub(r' 3$', ' III', simpler)
            # Remove trailing 4-digit years without parentheses (e.g. "Fullmetal Alchemist 2018")
            simpler = re.sub(r'\s+\d{4}$', '', simpler).strip()
            
            r_search = requests.get("http://www.omdbapi.com/", params={"s": simpler, "apikey": api_key}, timeout=4)
            data_search = r_search.json()
            if data_search.get("Response") == "True" and data_search.get("Search"):
                poster_url = data_search["Search"][0].get("Poster", "N/A")
                
        if not poster_url or poster_url == "N/A":
            return ""
        # Step 2: download the image bytes and encode as base64
        img_r = requests.get(poster_url, timeout=6)
        if img_r.status_code == 200:
            b64 = base64.b64encode(img_r.content).decode()
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        pass
    return ""

with st.spinner("🔄  Loading data & training models…"):
    try:
        movies, ratings, train, test = load_data()
        content_model, collab_model, hybrid_model = train_models(movies, train)
    except Exception as e:
        st.error(f"**Data load error.** Make sure `movies.csv` and `ratings.csv` are in `data/raw/`.\n\n`{e}`")
        st.stop()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.8rem 0 1.2rem; position:relative;">
        <!-- Glowing ring -->
        <div style="
            width:72px; height:72px; margin:0 auto 0.8rem;
            border-radius:50%;
            background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(59,130,246,0.3));
            border: 2px solid rgba(167,139,250,0.5);
            box-shadow: 0 0 25px rgba(124,58,237,0.4), 0 0 60px rgba(124,58,237,0.15), inset 0 0 20px rgba(167,139,250,0.1);
            display:flex; align-items:center; justify-content:center;
            font-size:2rem;
            animation: pulse-glow 3s ease-in-out infinite;
        ">🎬</div>
        <div style="
            font-size:1.6rem; font-weight:800;
            background: linear-gradient(90deg, #a78bfa, #f472b6, #60a5fa);
            background-size: 200% 200%;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            animation: gradient-shift 5s ease infinite;
            letter-spacing: -0.5px;
        ">CineMatch</div>
        <div style="
            font-size:0.65rem; color:rgba(167,139,250,0.5);
            letter-spacing:3px; margin-top:4px; font-weight:600;
            text-transform:uppercase;
        ">SMART RECOMMENDER</div>
        <div style="margin-top:12px; display:flex; justify-content:center; gap:4px;">
            <div style="width:4px;height:4px;border-radius:50%;background:#a78bfa;opacity:0.6;"></div>
            <div style="width:18px;height:4px;border-radius:2px;background:linear-gradient(90deg,#a78bfa,#60a5fa);"></div>
            <div style="width:4px;height:4px;border-radius:50%;background:#60a5fa;opacity:0.6;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Strategy Selection first
    st.markdown('<div style="font-size:0.65rem;color:rgba(167,139,250,0.6);letter-spacing:2px;text-transform:uppercase;font-weight:600;margin-bottom:0.5rem;">🔀 Strategy</div>', unsafe_allow_html=True)
    strategy = st.radio("Strategy", ["Hybrid", "Content-Based", "Collaborative Filtering"], label_visibility="collapsed")
    
    st.markdown("---")

    # Dynamic Inputs
    selected_user = None
    selected_movie = None
    
    if strategy in ["Hybrid", "Collaborative Filtering"]:
        st.markdown('<div style="font-size:0.65rem;color:rgba(167,139,250,0.6);letter-spacing:2px;text-transform:uppercase;font-weight:600;margin-bottom:0.5rem;">👤 User Profile</div>', unsafe_allow_html=True)
        selected_user = st.selectbox("User ID", sorted(ratings['userId'].unique()), label_visibility="collapsed")

    if strategy in ["Hybrid", "Content-Based"]:
        st.markdown('<div style="font-size:0.65rem;color:rgba(167,139,250,0.6);letter-spacing:2px;text-transform:uppercase;font-weight:600;margin:0.8rem 0 0.5rem;">🎥 Seed Movie</div>', unsafe_allow_html=True)
        movie_titles   = sorted(movies['title'].dropna().unique())
        selected_movie = st.selectbox("Movie", movie_titles, label_visibility="collapsed")

    st.markdown('<div style="font-size:0.65rem;color:rgba(167,139,250,0.6);letter-spacing:2px;text-transform:uppercase;font-weight:600;margin:0.8rem 0 0.5rem;">🎯 Top N</div>', unsafe_allow_html=True)
    top_n = st.slider("Top N", 5, 20, 10, label_visibility="collapsed")

    st.markdown("---")
    run_btn  = st.button("✨  Get Recommendations")
    eval_btn = st.button("📊  Evaluate Models")

    # Hardcoded OMDb API Key (Using fresh key to bypass previous rate limit)
    omdb_key = "b6003d8a"

    # ── Quick stats ──
    st.markdown("---")
    st.markdown('<div style="font-size:0.65rem;color:rgba(167,139,250,0.6);letter-spacing:2px;text-transform:uppercase;font-weight:600;margin-bottom:0.8rem;">📁 Dataset Stats</div>', unsafe_allow_html=True)
    stats = [("Movies", f"{movies.shape[0]:,}"), ("Users", f"{ratings['userId'].nunique():,}"),
             ("Ratings", f"{len(ratings):,}"), ("Genres", str(len(set(g for gl in movies['genre_list'] for g in gl))))]
    for label, val in stats:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:0.35rem 0;border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="color:rgba(255,255,255,0.4);font-size:0.8rem;">{label}</span>
            <span style="color:#a78bfa;font-weight:700;font-size:0.82rem;">{val}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="padding: 2.5rem 0 2rem; margin-bottom: 2rem; position:relative;">
    <!-- Neon accent line -->
    <div style="
        position:absolute; top:0; left:0;
        width:60px; height:3px;
        background: linear-gradient(90deg, #7c3aed, #f472b6);
        border-radius:2px;
        box-shadow: 0 0 12px rgba(124,58,237,0.7);
    "></div>
    <h1 style="
        font-size:2.8rem; font-weight:800; margin: 1.2rem 0 0.5rem;
        background: linear-gradient(90deg, #ffffff 0%, #c4b5fd 40%, #93c5fd 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -1px; line-height: 1.1;
    ">Movie Recommendations</h1>
    <p style="color:rgba(255,255,255,0.3); margin: 0 0 1rem; font-size:0.9rem; letter-spacing:0.3px;">
        Discover your next favourite film with AI-powered hybrid intelligence.
    </p>
    <div style="display:flex; gap:8px; flex-wrap:wrap;">
        <span style="background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.25);color:rgba(167,139,250,0.8);padding:3px 12px;border-radius:50px;font-size:0.72rem;font-weight:600;letter-spacing:0.5px;">TF-IDF</span>
        <span style="background:rgba(96,165,250,0.1);border:1px solid rgba(96,165,250,0.25);color:rgba(96,165,250,0.8);padding:3px 12px;border-radius:50px;font-size:0.72rem;font-weight:600;letter-spacing:0.5px;">Cosine Similarity</span>
        <span style="background:rgba(244,114,182,0.1);border:1px solid rgba(244,114,182,0.25);color:rgba(244,114,182,0.8);padding:3px 12px;border-radius:50px;font-size:0.72rem;font-weight:600;letter-spacing:0.5px;">SVD Factorisation</span>
        <span style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.25);color:rgba(34,197,94,0.8);padding:3px 12px;border-radius:50px;font-size:0.72rem;font-weight:600;letter-spacing:0.5px;">Hybrid Weighting</span>
    </div>
    <div style="margin-top:1.5rem;height:1px;background:linear-gradient(90deg,rgba(124,58,237,0.4),rgba(59,130,246,0.2),transparent);"></div>
</div>
""", unsafe_allow_html=True)

col_main, col_info = st.columns([3, 1], gap="large")

def render_grid(recs, score_col, score_label, accent, max_score=1.0, api_key=""):
    """Render recommendations as a Netflix-style poster card grid."""
    if recs is None or (hasattr(recs, 'empty') and recs.empty):
        st.warning("⚠️  No recommendations found. Try a different movie or user.")
        return

    rows_list = list(recs.iterrows())
    cols_per_row = 5
    for row_start in range(0, len(rows_list), cols_per_row):
        row_slice = rows_list[row_start: row_start + cols_per_row]
        cols = st.columns(len(row_slice), gap="small")
        for col, (rank_offset, (_, row)) in zip(cols, enumerate(row_slice)):
            rank = row_start + rank_offset + 1
            score = float(row.get(score_col, 0))
            pct = min(100, max(0, (score / max_score) * 100)) if max_score else 0
            genres_clean = str(row.get('genres', '')).replace('|', ' · ')
            title = row['title']

            # Fetch poster
            poster_url = get_movie_poster_v6(title, api_key) if api_key else ""

            if poster_url:
                img_html = f'<img src="{poster_url}" style="width:100%;height:240px;object-fit:cover;display:block;border-radius:0;">'
            else:
                # Beautiful gradient placeholder
                color2 = accent + "55"
                first_letter = title[0].upper() if title else "?"
                img_html = f"""
                <div style="width:100%;height:240px;display:flex;align-items:center;justify-content:center;
                    background:linear-gradient(135deg,{accent}33,{color2},rgba(0,0,0,0.5));
                    font-size:3.5rem;font-weight:800;color:{accent};">
                    {first_letter}
                </div>"""

            with col:
                card = (
                    f'<div style="position:relative;border-radius:14px;overflow:hidden;background:#0d0d1a;'
                    f'border:1px solid rgba(255,255,255,0.07);box-shadow:0 4px 20px rgba(0,0,0,0.5);'
                    f'transition:all 0.3s ease;cursor:pointer;"'
                    f' onmouseover="this.style.transform=\'translateY(-6px)\';this.style.boxShadow=\'0 16px 40px rgba(0,0,0,0.7),0 0 0 1px {accent}55\';"'
                    f' onmouseout="this.style.transform=\'\';this.style.boxShadow=\'0 4px 20px rgba(0,0,0,0.5)\';">'
                    f'{img_html}'
                    f'<div style="position:absolute;top:10px;left:10px;background:rgba(0,0,0,0.75);backdrop-filter:blur(8px);'
                    f'border:1px solid {accent}66;color:{accent};font-size:0.72rem;font-weight:800;'
                    f'padding:3px 9px;border-radius:50px;letter-spacing:0.5px;">#{rank:02d}</div>'
                    f'<div style="position:absolute;top:10px;right:10px;background:rgba(0,0,0,0.75);backdrop-filter:blur(8px);'
                    f'border:1px solid {accent}66;color:{accent};font-size:0.72rem;font-weight:800;'
                    f'padding:3px 9px;border-radius:50px;">{score:.2f}</div>'
                    f'<div style="padding:0.9rem 0.8rem 0.8rem;">'
                    f'<div style="font-size:0.88rem;font-weight:700;color:#f1f5f9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:4px;">{title}</div>'
                    f'<div style="font-size:0.7rem;color:rgba(255,255,255,0.4);margin-bottom:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{genres_clean}</div>'
                    f'<div style="background:rgba(255,255,255,0.08);border-radius:50px;height:3px;width:100%;">'
                    f'<div style="background:{accent};height:3px;border-radius:50px;width:{pct:.1f}%;box-shadow:0 0 6px {accent};"></div>'
                    f'</div></div></div>'
                )
                st.markdown(card, unsafe_allow_html=True)


with col_main:
    if run_btn:
        badge_map = {
            "Content-Based": '<span style="background:rgba(167,139,250,0.2);color:#a78bfa;padding:3px 10px;border-radius:50px;font-size:0.8rem;vertical-align:middle;margin-left:8px;">Content</span>',
            "Collaborative Filtering": '<span style="background:rgba(96,165,250,0.2);color:#60a5fa;padding:3px 10px;border-radius:50px;font-size:0.8rem;vertical-align:middle;margin-left:8px;">Collaborative</span>',
            "Hybrid": '<span style="background:rgba(244,114,182,0.2);color:#f472b6;padding:3px 10px;border-radius:50px;font-size:0.8rem;vertical-align:middle;margin-left:8px;">Hybrid</span>',
        }
        
        st.markdown(f"""
        <div style="font-size:1.3rem;font-weight:700;color:#e2e8f0;margin-bottom:1.5rem;display:flex;align-items:center;">
            🍿 Top Recommendations {badge_map[strategy]}
        </div>
        """, unsafe_allow_html=True)

        if strategy == "Content-Based":
            with st.spinner("Computing content similarity…"):
                recs_cb = content_model.recommend(selected_movie, top_n=top_n)
            render_grid(recs_cb, "similarity", "Similarity", "#a78bfa", api_key=omdb_key)

        elif strategy == "Collaborative Filtering":
            with st.spinner("Running SVD predictions…"):
                recs_cf = collab_model.recommend(selected_user, movies, top_n=top_n)
            render_grid(recs_cf, "est_rating", "SVD Score", "#60a5fa", max_score=5.0, api_key=omdb_key)

        elif strategy == "Hybrid":
            with st.spinner("Running hybrid model…"):
                recs_hy = hybrid_model.recommend(selected_user, selected_movie, top_n=top_n)
            render_grid(recs_hy, "hybrid_score", "Hybrid Score", "#f472b6", api_key=omdb_key)

    else:
        # ── Landing splash ──
        st.markdown(f"""
        <div style="
            border: 1px dashed rgba(255,255,255,0.1);
            border-radius: 20px; padding: 5rem 3rem;
            text-align: center; margin-top: 1rem;
            background: rgba(255,255,255,0.015);
            ">
            <div style="font-size:4rem; margin-bottom:1.2rem;">🍿</div>
            <div style="font-size:1.15rem;font-weight:600;color:rgba(255,255,255,0.7);margin-bottom:0.5rem;">
                Ready to discover great movies?
            </div>
            <div style="color:rgba(255,255,255,0.3);font-size:0.9rem; max-width:360px;margin:0 auto;">
                Select your strategy and preferences from the sidebar,
                then click <strong style="color:#a78bfa;">✨ Get Recommendations</strong>.
            </div>
            <div style="margin-top:2.5rem; display:flex; gap:1rem; justify-content:center; flex-wrap:wrap;">
                {''.join(f'<span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:50px;padding:5px 16px;font-size:0.8rem;color:rgba(255,255,255,0.4);">{t}</span>' for t in ['TF-IDF Vectorisation','Cosine Similarity','SVD Factorisation','Weighted Hybrid'])}
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_info:
    st.markdown('<div style="font-size:1.3rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">🔬 Models</div>', unsafe_allow_html=True)
    model_info = [
        ("Content-Based",  "TF-IDF · Cosine Sim",      "background:rgba(167,139,250,0.2);color:#a78bfa;border:1px solid rgba(167,139,250,0.4);"),
        ("Collaborative",  "TruncatedSVD · 50 factors", "background:rgba(96,165,250,0.2);color:#60a5fa;border:1px solid rgba(96,165,250,0.4);"),
        ("Hybrid",         "Weighted Avg · α = 0.5",    "background:rgba(244,114,182,0.2);color:#f472b6;border:1px solid rgba(244,114,182,0.4);"),
    ]
    for name, detail, style in model_info:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:0.9rem 1.2rem;margin-bottom:0.5rem;">
            <span style="{style}padding:0.25rem 0.8rem;border-radius:50px;font-size:0.75rem;font-weight:600;display:inline-block;margin-bottom:0.4rem;">{name}</span>
            <div style="color:rgba(255,255,255,0.45);font-size:0.78rem;margin-top:0.25rem;">{detail}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Currently viewing ──
    if run_btn:
        st.markdown('<div style="font-size:1.3rem;font-weight:700;color:#e2e8f0;margin:1.5rem 0 1rem;">🎯 Current Query</div>', unsafe_allow_html=True)
        
        user_html = ""
        if selected_user:
            user_html = f'<div style="color:rgba(255,255,255,0.45); font-size:0.75rem; margin-bottom:0.3rem">USER</div><div style="color:#e2e8f0; font-weight:600; font-size:0.9rem">#{selected_user}</div>'
            
        movie_html = ""
        if selected_movie:
            m_top = "0.6rem" if selected_user else "0rem"
            movie_html = f'<div style="color:rgba(255,255,255,0.45); font-size:0.75rem; margin: {m_top} 0 0.3rem">SEED MOVIE</div><div style="color:#e2e8f0; font-weight:600; font-size:0.9rem">{selected_movie}</div>'
            
        st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:0.9rem 1.2rem;">{user_html}{movie_html}</div>""", unsafe_allow_html=True)

# ─── Evaluation Section ───────────────────────────────────────────────────────
if eval_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:1.2rem;font-weight:700;color:#f1f5f9;margin-bottom:1.2rem;
        display:flex;align-items:center;gap:0.5rem;">
        📊 Model Evaluation <span style="font-size:0.75rem;font-weight:500;color:rgba(255,255,255,0.35);margin-left:4px;">— Collaborative SVD on 20% test split</span>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Evaluating…"):
        metrics = evaluate_models(collab_model, test)

    items = [
        ("RMSE",       metrics["RMSE"],         "#ef4444", "Prediction error",    "Lower is better"),
        ("MAE",         metrics["MAE"],           "#f97316", "Absolute error",      "Lower is better"),
        ("Precision",   metrics["Precision@K"],   "#22c55e", "Relevant in top 10",  "Higher is better"),
        ("Recall",      metrics["Recall@K"],      "#3b82f6", "Coverage of relevant","Higher is better"),
        ("F1-Score",    metrics["F1-Score"],      "#a78bfa", "Harmonic mean",       "P-R balance"),
    ]

    cols = st.columns(5, gap="medium")
    for col, (name, val, color, subtitle, hint) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.03);
                border: 1px solid {color}33;
                border-radius: 16px; padding: 1.4rem 1rem;
                text-align: center;
                ">
                <div style="width:42px;height:4px;background:{color};border-radius:2px;margin:0 auto 1rem;"></div>
                <div style="font-size:2rem;font-weight:800;color:{color};line-height:1;">{val:.4f}</div>
                <div style="font-size:0.82rem;font-weight:700;color:#e2e8f0;margin:0.5rem 0 0.2rem;">{name}</div>
                <div style="font-size:0.7rem;color:rgba(255,255,255,0.3);">{subtitle}</div>
                <div style="margin-top:0.7rem;font-size:0.65rem;background:{color}22;color:{color};
                    border-radius:50px;padding:2px 10px;display:inline-block;font-weight:600;">
                    {hint}
                </div>
            </div>
            """, unsafe_allow_html=True)
