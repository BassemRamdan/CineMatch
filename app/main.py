import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import requests
import base64
import re
import textwrap

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_and_preprocess_data
from src.content_engine import ContentBasedRecommender
from src.collaborative_engine import CollaborativeRecommender
from src.hybrid_engine import HybridRecommender
from src.evaluation import run_full_evaluation

st.set_page_config(
    page_title="CineMatch Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── 1. UI Components & CSS ──────────────────────────────────────────────────
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
    if not genres_str or str(genres_str) == "nan":
        return ""
    return "".join(genre_badge(g) for g in str(genres_str).replace("|", ",").split(","))

def apply_custom_css():
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
    
    /* Input Styling */
    .stSelectbox > div > div, .stTextInput > div > div, .stMultiSelect > div > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(167,139,250,0.2) !important;
        border-radius: 10px !important;
        transition: border-color 0.2s ease !important;
    }
    .stSelectbox > div > div:hover, .stTextInput > div > div:hover {
        border-color: rgba(167,139,250,0.5) !important;
    }
    label {
        color: rgba(167,139,250,0.6) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer { visibility: hidden; }
    header { background: transparent !important; }

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

    /* ── Nav Buttons in Sidebar ── */
    .nav-btn .stButton > button {
        background: transparent !important;
        border: 1px solid rgba(167,139,250,0.2) !important;
        box-shadow: none !important;
        color: rgba(255,255,255,0.7) !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0.8rem 1rem !important;
    }
    .nav-btn .stButton > button:hover {
        background: rgba(167,139,250,0.1) !important;
        border-color: rgba(167,139,250,0.5) !important;
        color: white !important;
        transform: translateY(0) !important;
    }
    .nav-btn-active .stButton > button {
        background: linear-gradient(90deg, rgba(124,58,237,0.2), transparent) !important;
        border-left: 3px solid #a78bfa !important;
        border-top: none !important; border-right: none !important; border-bottom: none !important;
        color: #a78bfa !important;
        border-radius: 0 12px 12px 0 !important;
        box-shadow: none !important;
    }

    /* ── Progress & Spinner ── */
    .stSpinner > div { border-top-color: #a78bfa !important; }
    .stProgress > div > div > div > div { background-color: #a78bfa !important; }

    /* ── HR ── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(167,139,250,0.3), transparent) !important;
        margin: 1.2rem 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_poster(title: str, api_key: str = "thewdb") -> str:
    if not api_key:
        return ""
    try:
        clean = title.split("(")[0].strip()
        match = re.search(r'^(.*),\s*(The|A|An)$', clean, flags=re.IGNORECASE)
        if match: clean = f"{match.group(2)} {match.group(1)}"
            
        r = requests.get("http://www.omdbapi.com/", params={"t": clean, "apikey": api_key}, timeout=4)
        data = r.json()
        poster_url = data.get("Poster", "N/A")
        
        if data.get("Response") != "True" or poster_url == "N/A":
            simpler = clean.split(":")[0].strip()
            simpler = simpler.replace("Part II", "II").replace("Part III", "III").replace("Part 2", "II")
            simpler = re.sub(r' 2$', ' II', simpler)
            simpler = re.sub(r' 3$', ' III', simpler)
            simpler = re.sub(r'\s+\d{4}$', '', simpler).strip()
            
            r_search = requests.get("http://www.omdbapi.com/", params={"s": simpler, "apikey": api_key}, timeout=4)
            data_search = r_search.json()
            if data_search.get("Response") == "True" and data_search.get("Search"):
                poster_url = data_search["Search"][0].get("Poster", "N/A")
                
        if not poster_url or poster_url == "N/A":
            return ""
        img_r = requests.get(poster_url, timeout=6)
        if img_r.status_code == 200:
            b64 = base64.b64encode(img_r.content).decode()
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        pass
    return ""

def render_movie_grid(recs, score_col, score_label, accent, max_score=1.0, api_key="thewdb", show_rank=True):
    if recs is None or (hasattr(recs, 'empty') and recs.empty):
        st.warning("⚠️  No movies found.")
        return

    rows_list = list(recs.iterrows())
    cols_per_row = 5
    for row_start in range(0, len(rows_list), cols_per_row):
        row_slice = rows_list[row_start: row_start + cols_per_row]
        cols = st.columns(len(row_slice), gap="small")
        for col, (rank_offset, (_, row)) in zip(cols, enumerate(row_slice)):
            rank = row_start + rank_offset + 1
            score = float(row.get(score_col, 0)) if score_col in row else 0
            pct = min(100, max(0, (score / max_score) * 100)) if max_score else 0
            genres_clean = str(row.get('genres', '')).replace('|', ' · ')
            title = row.get('title', 'Unknown')

            poster_url = get_movie_poster(title, api_key)

            if poster_url:
                img_html = f'<img src="{poster_url}" style="width:100%;aspect-ratio:2/3;object-fit:cover;display:block;border-radius:0;object-position:top;">'
            else:
                color2 = accent + "55"
                first_letter = title[0].upper() if title else "?"
                img_html = f"""<div style="width:100%;aspect-ratio:2/3;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,{accent}33,{color2},rgba(0,0,0,0.5));font-size:3.5rem;font-weight:800;color:{accent};">{first_letter}</div>"""

            with col:
                rank_badge = f'<div style="position:absolute;top:10px;left:10px;background:rgba(0,0,0,0.75);backdrop-filter:blur(8px);border:1px solid {accent}66;color:{accent};font-size:0.72rem;font-weight:800;padding:3px 9px;border-radius:50px;letter-spacing:0.5px;">#{rank:02d}</div>' if show_rank else ""
                score_badge = f'<div style="position:absolute;top:10px;right:10px;background:rgba(0,0,0,0.75);backdrop-filter:blur(8px);border:1px solid {accent}66;color:{accent};font-size:0.72rem;font-weight:800;padding:3px 9px;border-radius:50px;">{score:.2f}</div>' if score_col else ""
                score_bar = f'<div style="background:rgba(255,255,255,0.08);border-radius:50px;height:3px;width:100%;"><div style="background:{accent};height:3px;border-radius:50px;width:{pct:.1f}%;box-shadow:0 0 6px {accent};"></div></div>' if score_col else ""

                card = f"""<div style="position:relative;border-radius:14px;overflow:hidden;background:#0d0d1a;border:1px solid rgba(255,255,255,0.07);box-shadow:0 4px 20px rgba(0,0,0,0.5);transition:all 0.3s ease;cursor:pointer;margin-bottom:1rem;" onmouseover="this.style.transform='translateY(-6px)';this.style.boxShadow='0 16px 40px rgba(0,0,0,0.7),0 0 0 1px {accent}55';" onmouseout="this.style.transform='';this.style.boxShadow='0 4px 20px rgba(0,0,0,0.5)';">
{img_html}{rank_badge}{score_badge}
<div style="padding:0.9rem 0.8rem 0.8rem;">
<div style="font-size:0.88rem;font-weight:700;color:#f1f5f9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:4px;" title="{title}">{title}</div>
<div style="font-size:0.7rem;color:rgba(255,255,255,0.4);margin-bottom:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{genres_clean}</div>
{score_bar}
</div>
</div>"""
                st.markdown(card, unsafe_allow_html=True)

def page_header(title, subtitle, accent_color="#7c3aed"):
    st.markdown(f"""
    <div style="padding: 1.5rem 0 1rem; margin-bottom: 1.5rem; position:relative;">
        <div style="position:absolute; top:0; left:0; width:60px; height:3px; background: linear-gradient(90deg, {accent_color}, transparent); border-radius:2px;"></div>
        <h1 style="font-size:2.5rem; font-weight:800; margin: 0.5rem 0 0.5rem; background: linear-gradient(90deg, #ffffff 0%, #e2e8f0 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px; line-height: 1.1;">
            {title}
        </h1>
        <p style="color:rgba(255,255,255,0.4); margin: 0; font-size:0.95rem; letter-spacing:0.3px;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# Apply CSS immediately
apply_custom_css()

# ─── 2. Data & Models ────────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = "Home"

@st.cache_data
def load_data_v2():
    base = os.path.dirname(__file__)
    return load_and_preprocess_data(
        os.path.join(base, "..", "data", "raw", "movies.csv"),
        os.path.join(base, "..", "data", "raw", "ratings.csv"),
    )

@st.cache_resource
def train_models_v3(movies, train):
    cb = ContentBasedRecommender(); cb.fit(movies)
    cf = CollaborativeRecommender(factors=50); cf.fit(train)
    hy = HybridRecommender(cb, cf, content_weight=0.3)
    return cb, cf, hy

with st.spinner("🔄  Booting CineMatch AI Engine..."):
    try:
        movies, ratings, train, test = load_data_v2()
        content_model, collab_model, hybrid_model = train_models_v3(movies, train)
    except Exception as e:
        st.error(f"**Data load error.** Make sure data is in `data/raw/`.\n\n`{e}`")
        st.stop()

# ─── 3. Sidebar Navigation ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.8rem 0 1.2rem; position:relative;">
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
        <div style="font-size:0.65rem; color:rgba(167,139,250,0.5); letter-spacing:3px; margin-top:4px; font-weight:600; text-transform:uppercase;">SMART RECOMMENDER</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:0.5rem 0 1.5rem 0 !important;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;color:rgba(167,139,250,0.6);letter-spacing:2px;text-transform:uppercase;font-weight:600;margin-bottom:0.8rem;padding-left:1rem;">Menu</div>', unsafe_allow_html=True)
    
    nav_items = [
        ("Home", "🏠 Discover"),
        ("Profile", "👤 My Profile"),
        ("Content", "🎥 Find Similar Movies"),
        ("Hybrid", "🔮 AI Smart Match"),
        ("Analytics", "📊 System Analytics")
    ]
    
    for page_id, label in nav_items:
        active_class = "nav-btn-active" if st.session_state.page == page_id else "nav-btn"
        st.markdown(f'<div class="{active_class}">', unsafe_allow_html=True)
        if st.button(label, key=f"nav_{page_id}", use_container_width=True):
            st.session_state.page = page_id
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin:2rem 0 1.5rem 0 !important;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;color:rgba(167,139,250,0.6);letter-spacing:2px;text-transform:uppercase;font-weight:600;margin-bottom:0.8rem;padding-left:1rem;">Database</div>', unsafe_allow_html=True)
    stats = [("Movies", f"{movies.shape[0]:,}"), ("Users", f"{ratings['userId'].nunique():,}"), ("Ratings", f"{len(ratings):,}")]
    for label, val in stats:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:0.35rem 1rem;border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="color:rgba(255,255,255,0.4);font-size:0.8rem;">{label}</span>
            <span style="color:#a78bfa;font-weight:700;font-size:0.82rem;">{val}</span>
        </div>""", unsafe_allow_html=True)

# ─── 4. Page Rendering Functions ─────────────────────────────────────────────

def render_home():
    page_header("Welcome to CineMatch", "Your AI-powered cinematic journey begins here.", "#7c3aed")
    
    st.markdown("""
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:2rem;margin-bottom:2rem;">
        <h3 style="color:#e2e8f0;font-size:1.3rem;margin-bottom:0.8rem;font-weight:700;">Start Your Journey</h3>
        <p style="color:rgba(255,255,255,0.6);font-size:0.95rem;line-height:1.6;margin-bottom:1.5rem;max-width:800px;">
            CineMatch uses advanced Machine Learning to understand your taste. 
            Before we generate personalized recommendations, tell us a bit about what you like.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">1. Select Your Profile</div>', unsafe_allow_html=True)
        if 'active_user' not in st.session_state:
            st.session_state.active_user = sorted(ratings['userId'].unique())[0]
            
        user_id = st.selectbox("Who is watching?", sorted(ratings['userId'].unique()), index=sorted(ratings['userId'].unique()).index(st.session_state.active_user))
        if user_id != st.session_state.active_user:
            st.session_state.active_user = user_id
            st.rerun()
            
        user_ratings = ratings[ratings['userId'] == user_id]
        st.markdown(f"<div style='color:rgba(255,255,255,0.4);font-size:0.85rem;margin-top:0.5rem;'>Profile loaded with {len(user_ratings)} existing ratings.</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">2. Quick Preferences</div>', unsafe_allow_html=True)
        all_genres = sorted(list(set(g for gl in movies['genre_list'] for g in gl if g != "(no genres listed)")))
        selected_genres = st.multiselect("What are you in the mood for?", all_genres, default=st.session_state.get('current_mood', []))
        
        # When preferences change, update session state
        if selected_genres != st.session_state.get('current_mood', []):
            st.session_state.current_mood = selected_genres
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Calculate rating count dynamically
    movie_counts = ratings['movieId'].value_counts().reset_index()
    movie_counts.columns = ['movieId', 'rating_count']
    trending_movies = movies.merge(movie_counts, on='movieId', how='left')
    
    # Filter by mood if selected
    current_mood = st.session_state.get('current_mood', [])
    if current_mood:
        st.markdown(f'<div style="font-size:1.4rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem;">🔥 Top Picks for You</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.9rem;color:rgba(255,255,255,0.5);margin-bottom:1.5rem;">Filtered by: {", ".join(current_mood)}</div>', unsafe_allow_html=True)
        # Check if the movie's genre_list contains ANY of the selected mood genres
        mask = trending_movies['genre_list'].apply(lambda x: any(g in x for g in current_mood) if isinstance(x, (list, tuple)) else False)
        trending_movies = trending_movies[mask]
    else:
        st.markdown('<div style="font-size:1.4rem;font-weight:700;color:#e2e8f0;margin-bottom:1.5rem;">🔥 Trending Now</div>', unsafe_allow_html=True)
        
    trending = trending_movies.sort_values(by='rating_count', ascending=False).head(10)
    render_movie_grid(trending, score_col=None, score_label=None, accent="#f472b6", show_rank=False)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
    with col_btn2:
        if st.button("Continue to Profile ➔", key="home_continue"):
            st.session_state.page = "Profile"
            st.rerun()

def render_profile():
    user_id = st.session_state.get('active_user', sorted(ratings['userId'].unique())[0])
    page_header(f"Profile: User #{user_id}", "Your watching history and taste analysis.", "#3b82f6")
    
    user_r = ratings[ratings["userId"] == user_id].copy()
    user_r = user_r.merge(movies[["movieId","title","genres","year"]], on="movieId", how="left")
    user_r = user_r.sort_values("rating", ascending=False)
    
    if user_r.empty:
        st.info("No ratings found for this user.")
        return
        
    col1, col2, col3, col4 = st.columns(4)
    def stat_card(col, title, val, color):
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:1.2rem;text-align:center;">
                <div style="font-size:0.75rem;color:rgba(255,255,255,0.4);letter-spacing:1px;text-transform:uppercase;margin-bottom:0.5rem;">{title}</div>
                <div style="font-size:1.8rem;font-weight:800;color:{color};">{val}</div>
            </div>
            """, unsafe_allow_html=True)
            
    stat_card(col1, "Total Ratings", f"{len(user_r):,}", "#3b82f6")
    stat_card(col2, "Average Rating", f"{user_r['rating'].mean():.2f}", "#a78bfa")
    
    genre_counts = pd.Series([g for glist in user_r["genres"].dropna().str.split("|") for g in glist if g != "(no genres listed)"]).value_counts()
    top_genre = genre_counts.index[0] if len(genre_counts) > 0 else "N/A"
    stat_card(col3, "Top Genre", top_genre, "#f472b6")
    
    if not user_r['year'].isna().all():
        user_r['decade'] = user_r['year'] // 10 * 10
        era_stats = user_r.groupby('decade').agg(avg_rating=('rating', 'mean'), count=('rating', 'count'))
        valid_eras = era_stats[era_stats['count'] >= (3 if len(user_r) > 10 else 1)]
        fav_decade = valid_eras['avg_rating'].idxmax() if not valid_eras.empty else era_stats['avg_rating'].idxmax()
    else:
        fav_decade = "N/A"
        
    stat_card(col4, "Favorite Era", f"{int(fav_decade)}s" if fav_decade != "N/A" else "N/A", "#22c55e")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([2,1], gap="large")
    with col_l:
        st.markdown('<div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">⭐ Highest Rated Movies</div>', unsafe_allow_html=True)
        top_movies = user_r.head(10)
        render_movie_grid(top_movies, score_col="rating", score_label="Your Rating", accent="#f59e0b", max_score=5.0, show_rank=False)
        
    with col_r:
        st.markdown('<div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">📊 Taste Profile</div>', unsafe_allow_html=True)
        st.markdown('<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:1.5rem;">', unsafe_allow_html=True)
        for genre, count in genre_counts.head(8).items():
            pct = count / len(user_r) * 100
            st.markdown(f"""
            <div style="margin-bottom:0.8rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:rgba(255,255,255,0.7); margin-bottom:4px;">
                    <span>{genre}</span><span style="color:#3b82f6;font-weight:600;">{count}</span>
                </div>
                <div style="background:rgba(255,255,255,0.05); height:4px; border-radius:2px;">
                    <div style="width:{pct}%; height:4px; background:linear-gradient(90deg, #3b82f6, #a78bfa); border-radius:2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_content():
    page_header("Find Similar Movies", "Because you liked this movie...", "#22c55e")
    movie_titles = sorted(movies['title'].dropna().unique())
    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.markdown('<div style="font-size:0.9rem;font-weight:600;color:rgba(255,255,255,0.6);margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:1px;">Select a movie you like</div>', unsafe_allow_html=True)
        selected_movie = st.selectbox("Select a movie", movie_titles, label_visibility="collapsed", key="cb_movie")
    with col2:
        st.markdown('<div style="font-size:0.9rem;font-weight:600;color:rgba(255,255,255,0.6);margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:1px;">Results</div>', unsafe_allow_html=True)
        top_n = st.slider("Top N", 5, 20, 10, label_visibility="collapsed", key="cb_topn")
        
    st.markdown("<hr>", unsafe_allow_html=True)
    if selected_movie:
        with st.spinner("Finding similar movies..."):
            recs_cb = content_model.recommend(selected_movie, top_n=top_n)
            
        poster_url = get_movie_poster(selected_movie)
        
        col_poster, col_info = st.columns([1, 4])
        with col_poster:
            if poster_url:
                st.markdown(f'<img src="{poster_url}" style="width:100%;aspect-ratio:2/3;object-fit:cover;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.5);">', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="width:100%;aspect-ratio:2/3;display:flex;align-items:center;justify-content:center;background:#22c55e33;border-radius:12px;color:#22c55e;font-size:3rem;font-weight:800;">?</div>', unsafe_allow_html=True)
        with col_info:
            st.markdown(f'<div style="font-size:1.4rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem;margin-top:1rem;">Because you liked <br><span style="color:#22c55e;font-size:2.5rem;">{selected_movie}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color:rgba(255,255,255,0.6);font-size:1.1rem;">Here are {top_n} movies with highly similar genres...</div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        render_movie_grid(recs_cb, "similarity", "Similarity", "#22c55e", show_rank=True)

def render_hybrid():
    page_header("AI Smart Match", "The ultimate personalized movie mix.", "#f472b6")
    user_id = st.session_state.get('active_user', sorted(ratings['userId'].unique())[0])
    movie_titles = sorted(movies['title'].dropna().unique())
    
    user_r = ratings[ratings["userId"] == user_id].copy()
    user_r = user_r.merge(movies[["movieId","genres"]], on="movieId", how="left")
    genre_counts = pd.Series([g for glist in user_r["genres"].dropna().str.split("|") for g in glist if g != "(no genres listed)"]).value_counts()
    top_genres = ", ".join(genre_counts.head(3).index.tolist()) if len(genre_counts) > 0 else "N/A"
    
    st.markdown(f"""
    <div style="background:linear-gradient(90deg, rgba(244,114,182,0.1), transparent); border-left: 4px solid #f472b6; padding: 1rem 1.5rem; border-radius: 4px; margin-bottom: 1.5rem;">
        <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-bottom:0.3rem;">Personalized for User #{user_id}</div>
        <div style="font-size:0.85rem; color:rgba(255,255,255,0.6);">
            You have rated <strong style="color:#f472b6;">{len(user_r)}</strong> movies. 
            Your favorite genres appear to be <strong style="color:#f472b6;">{top_genres}</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.markdown('<div style="font-size:0.8rem;color:rgba(255,255,255,0.5);text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;">Choose a seed movie</div>', unsafe_allow_html=True)
        selected_movie = st.selectbox("Movie", movie_titles, label_visibility="collapsed", key="hy_movie")
    with col2:
        st.markdown('<div style="font-size:0.8rem;color:rgba(255,255,255,0.5);text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;">Results</div>', unsafe_allow_html=True)
        top_n = st.slider("Top N", 5, 20, 10, label_visibility="collapsed", key="hy_topn")
        
    st.markdown("<hr>", unsafe_allow_html=True)
    if selected_movie:
        with st.spinner("Mixing collaborative & content signals..."):
            recs_hy = hybrid_model.recommend(user_id, selected_movie, top_n=top_n)
            
        if recs_hy is not None and not recs_hy.empty:
            st.markdown(f'<div style="font-size:1.4rem;font-weight:700;color:#e2e8f0;margin-bottom:1.5rem;">The Perfect Blend</div>', unsafe_allow_html=True)
            top_rec = recs_hy.iloc[0]
            sim_score = top_rec.get('similarity', 0)
            cf_score = top_rec.get('est_rating', 0)
            cf_norm = min(1.0, max(0, cf_score / 5.0))
            hy_score = top_rec.get('hybrid_score', 0)
            
            html_block = textwrap.dedent(f"""
            <div style="background:rgba(0,0,0,0.3); border:1px solid rgba(244,114,182,0.3); border-radius:12px; padding:1.5rem; margin-bottom:2rem; display:flex; align-items:center; gap:2rem;">
                <div style="flex:1;">
                    <div style="font-size:0.8rem; color:#a78bfa; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.3rem;">Top Match Analysis</div>
                    <div style="font-size:1.2rem; font-weight:700; color:white; margin-bottom:1rem;">{top_rec['title']}</div>
                    <div style="margin-bottom:0.8rem;">
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:rgba(255,255,255,0.7); margin-bottom:4px;">
                            <span>Content Similarity (TF-IDF)</span><span style="color:#22c55e;">{sim_score:.2f}</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.05); height:6px; border-radius:3px;">
                            <div style="width:{sim_score*100}%; height:6px; background:#22c55e; border-radius:3px;"></div>
                        </div>
                    </div>
                    <div style="margin-bottom:0.8rem;">
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:rgba(255,255,255,0.7); margin-bottom:4px;">
                            <span>Collaborative Prediction (SVD)</span><span style="color:#3b82f6;">{cf_score:.2f} / 5.0</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.05); height:6px; border-radius:3px;">
                            <div style="width:{cf_norm*100}%; height:6px; background:#3b82f6; border-radius:3px;"></div>
                        </div>
                    </div>
                </div>
                <div style="text-align:center; padding-left:2rem; border-left:1px dashed rgba(255,255,255,0.1);">
                    <div style="font-size:0.8rem; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;">Final Hybrid Score</div>
                    <div style="font-size:3.5rem; font-weight:800; color:#f472b6; line-height:1; text-shadow: 0 0 20px rgba(244,114,182,0.4);">{hy_score:.2f}</div>
                    <div style="font-size:0.7rem; color:rgba(255,255,255,0.3); margin-top:0.5rem;">Weighted 70% CF / 30% Content</div>
                </div>
            </div>
            """)
            st.markdown(html_block, unsafe_allow_html=True)
            render_movie_grid(recs_hy, "hybrid_score", "Hybrid Score", "#f472b6", show_rank=True)

def render_analytics():
    page_header("System Analytics", "Model Performance & Evaluation Metrics", "#eab308")
    st.markdown("""
    <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1.2rem; display:flex;align-items:center;gap:0.5rem;">
        📊 Model Evaluation <span style="font-size:0.75rem;font-weight:500;color:rgba(255,255,255,0.35);margin-left:4px;">— Collaborative SVD on 20% test split</span>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Evaluating models... this takes a few seconds."):
        @st.cache_data
        def get_metrics(_model, _test):
            return run_full_evaluation(_model, _test)
        metrics = get_metrics(collab_model, test)

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
            
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">Architecture Summary</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    def arch_card(c, title, details, color):
        with c:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:1.2rem;height:100%;">
                <div style="color:{color};font-weight:700;margin-bottom:0.8rem;display:flex;align-items:center;gap:8px;">
                    <div style="width:8px;height:8px;border-radius:50%;background:{color};"></div>
                    {title}
                </div>
                <div style="color:rgba(255,255,255,0.6);font-size:0.85rem;line-height:1.6;">{details}</div>
            </div>
            """, unsafe_allow_html=True)
            
    arch_card(col1, "Content-Based", "TF-IDF Vectorization on movie genres. Pairwise Cosine Similarity computed to find exact genre matches in O(1) retrieval time.", "#22c55e")
    arch_card(col2, "Collaborative SVD", "Surprise SVD Matrix Factorization with 50 latent factors. Trained on 80% split using Stochastic Gradient Descent.", "#3b82f6")
    arch_card(col3, "Hybrid Engine", "Weighted combination: 70% Collaborative Filtering, 30% Content-Based. Optimized to maximize Precision@10.", "#f472b6")

# ─── 5. Route the Application ────────────────────────────────────────────────
if st.session_state.page == "Home":
    render_home()
elif st.session_state.page == "Profile":
    render_profile()
elif st.session_state.page == "Content":
    render_content()
elif st.session_state.page == "Hybrid":
    render_hybrid()
elif st.session_state.page == "Analytics":
    render_analytics()
