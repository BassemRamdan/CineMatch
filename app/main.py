import streamlit as st
import pandas as pd
import os
import sys

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_and_preprocess_data
from src.content_engine import ContentBasedRecommender
from src.collaborative_engine import CollaborativeRecommender
from src.hybrid_engine import HybridRecommender
from src.evaluation import run_full_evaluation

st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
    text-align: center;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #f472b6, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
}
.hero p {
    color: rgba(255,255,255,0.6);
    font-size: 1.1rem;
    margin: 0;
}

/* Cards */
.card {
    background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.card:hover {
    border-color: rgba(167,139,250,0.5);
    transform: translateY(-2px);
}

/* Movie recommendation card */
.movie-card {
    background: linear-gradient(135deg, rgba(167,139,250,0.1), rgba(96,165,250,0.05));
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: all 0.25s ease;
}
.movie-card:hover {
    background: linear-gradient(135deg, rgba(167,139,250,0.18), rgba(96,165,250,0.1));
    border-color: rgba(167,139,250,0.5);
    transform: translateX(4px);
}
.movie-rank {
    font-size: 1.5rem;
    font-weight: 700;
    color: rgba(167,139,250,0.6);
    min-width: 2.5rem;
    text-align: center;
}
.movie-title {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0 0 0.25rem 0;
}
.movie-genres {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.45);
    margin: 0;
}
.movie-score {
    margin-left: auto;
    text-align: right;
}
.score-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
}
.score-label {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.4);
    display: block;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    backdrop-filter: blur(10px);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: block;
}
.metric-name {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    margin-top: 0.25rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* Badge pills for strategy */
.badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-right: 0.4rem;
}
.badge-content  { background: rgba(167,139,250,0.2); color: #a78bfa; border: 1px solid rgba(167,139,250,0.4); }
.badge-collab   { background: rgba(96,165,250,0.2);  color: #60a5fa; border: 1px solid rgba(96,165,250,0.4); }
.badge-hybrid   { background: rgba(244,114,182,0.2); color: #f472b6; border: 1px solid rgba(244,114,182,0.4); }

/* Section headers */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,12,41,0.95), rgba(36,36,62,0.95));
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSlider label {
    color: rgba(255,255,255,0.7) !important;
    font-weight: 500;
}

/* Button override */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #4338ca) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}

/* Override the default streamlit table */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #a78bfa !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 1.5rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Data & Model Loading ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    movies_path  = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "movies.csv")
    ratings_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ratings.csv")
    return load_and_preprocess_data(movies_path, ratings_path)

@st.cache_resource
def train_models(movies, train):
    content_model = ContentBasedRecommender()
    content_model.fit(movies)
    collab_model  = CollaborativeRecommender(factors=50)
    collab_model.fit(train)
    hybrid_model  = HybridRecommender(content_model, collab_model, content_weight=0.5)
    return content_model, collab_model, hybrid_model

@st.cache_data
def evaluate_models(_collab_model, test):
    return run_full_evaluation(_collab_model, test)

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎬 CineMatch</h1>
  <p>Hybrid Movie Recommendation System · Content-Based · Collaborative Filtering · SVD</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Data ───────────────────────────────────────────────────────────────
with st.spinner("Loading & preprocessing MovieLens dataset..."):
    try:
        movies, ratings, train, test = load_data()
    except Exception as e:
        st.error(f"❌ Could not load data. Ensure `movies.csv` and `ratings.csv` are in `data/raw/`.\n\n`{e}`")
        st.stop()

with st.spinner("Training content-based & collaborative models..."):
    content_model, collab_model, hybrid_model = train_models(movies, train)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Preferences")
    st.markdown("---")

    all_user_ids   = sorted(ratings['userId'].unique())
    selected_user  = st.selectbox("👤 User ID", all_user_ids, help="Select the user to get recommendations for.")

    movie_titles   = sorted(movies['title'].dropna().unique())
    selected_movie = st.selectbox("🎥 Seed Movie", movie_titles, help="Select a movie you like.")

    strategy = st.radio(
        "🔀 Strategy",
        ["Content-Based", "Collaborative Filtering", "Hybrid"],
        help="Choose how recommendations are generated."
    )

    top_n = st.slider("🎯 Top N Results", 5, 20, 10)

    st.markdown("---")
    run_btn = st.button("✨ Get Recommendations")

    st.markdown("---")
    eval_btn = st.button("📊 Evaluate Model")

# ─── Strategy badge map ──────────────────────────────────────────────────────
badge_map = {
    "Content-Based":          '<span class="badge badge-content">Content-Based</span>',
    "Collaborative Filtering": '<span class="badge badge-collab">Collaborative SVD</span>',
    "Hybrid":                 '<span class="badge badge-hybrid">Hybrid · Weighted Avg</span>',
}

# ─── Main Area ───────────────────────────────────────────────────────────────
col_main, col_info = st.columns([3, 1], gap="large")

with col_main:
    # ── Recommendations ──────────────────────────────────────────────────────
    if run_btn:
        st.markdown(
            f'<div class="section-title">🍿 Recommendations &nbsp; {badge_map[strategy]}</div>',
            unsafe_allow_html=True
        )

        with st.spinner("Finding your perfect movies..."):
            recs = None
            err  = None

            if strategy == "Content-Based":
                recs = content_model.recommend(selected_movie, top_n=top_n)

            elif strategy == "Collaborative Filtering":
                recs = collab_model.recommend(selected_user, movies, top_n=top_n)

            elif strategy == "Hybrid":
                recs = hybrid_model.recommend(selected_user, selected_movie, top_n=top_n)

        if recs is None or (hasattr(recs, 'empty') and recs.empty):
            st.warning("⚠️ No recommendations found. Try a different movie or user.")
        else:
            # Choose which column to display as the score
            if strategy == "Content-Based":
                score_col, score_label = "similarity", "Similarity"
            elif strategy == "Collaborative Filtering":
                score_col, score_label = "est_rating", "SVD Score"
            else:
                score_col, score_label = "hybrid_score", "Hybrid Score"

            for rank, (_, row) in enumerate(recs.iterrows(), start=1):
                score_val = row.get(score_col, 0)
                genres_clean = str(row.get('genres', '')).replace("|", " · ")
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-rank">#{rank}</div>
                    <div>
                        <p class="movie-title">{row['title']}</p>
                        <p class="movie-genres">{genres_clean}</p>
                    </div>
                    <div class="movie-score">
                        <span class="score-value">{score_val:.3f}</span>
                        <span class="score-label">{score_label}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        # Placeholder when no recommendations yet
        st.markdown("""
        <div class="card" style="text-align:center; padding: 3rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎥</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 1rem;">
                Select your preferences on the left and click<br>
                <strong style="color: rgba(167,139,250,0.8);">✨ Get Recommendations</strong> to discover movies.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Evaluation ───────────────────────────────────────────────────────────
    if eval_btn:
        st.markdown('<div class="section-title">📊 Collaborative Model Evaluation</div>', unsafe_allow_html=True)
        with st.spinner("Evaluating on test set..."):
            metrics = evaluate_models(collab_model, test)

        metric_items = [
            ("RMSE",          metrics["RMSE"],          "Lower is better"),
            ("MAE",           metrics["MAE"],            "Lower is better"),
            ("Precision@10",  metrics["Precision@K"],    "Higher is better"),
            ("Recall@10",     metrics["Recall@K"],       "Higher is better"),
            ("F1-Score",      metrics["F1-Score"],       "Harmonic mean"),
        ]

        cols = st.columns(5)
        for col, (name, val, hint) in zip(cols, metric_items):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-value">{val:.4f}</span>
                    <div class="metric-name">{name}</div>
                    <div style="font-size:0.68rem; color:rgba(255,255,255,0.3); margin-top:0.2rem">{hint}</div>
                </div>
                """, unsafe_allow_html=True)

with col_info:
    # ── Dataset Stats ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📁 Dataset</div>', unsafe_allow_html=True)
    stats = [
        ("🎬 Movies",  f"{movies.shape[0]:,}"),
        ("👥 Users",   f"{ratings['userId'].nunique():,}"),
        ("⭐ Ratings", f"{len(ratings):,}"),
        ("📚 Genres",  str(len(set(g for gl in movies['genre_list'] for g in gl)))),
        ("🚆 Train",   f"{len(train):,}"),
        ("🧪 Test",    f"{len(test):,}"),
    ]
    for label, value in stats:
        st.markdown(f"""
        <div class="card" style="padding: 0.8rem 1.2rem; margin-bottom: 0.5rem; display:flex; justify-content:space-between; align-items:center">
            <span style="color:rgba(255,255,255,0.55); font-size:0.85rem">{label}</span>
            <span style="color:#a78bfa; font-weight:700; font-size:0.95rem">{value}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔬 Models</div>', unsafe_allow_html=True)
    model_info = [
        ("Content-Based",  "TF-IDF · Cosine Sim",      "badge-content"),
        ("Collaborative",  "TruncatedSVD · 50 factors", "badge-collab"),
        ("Hybrid",         "Weighted Avg · α = 0.5",    "badge-hybrid"),
    ]
    for name, detail, badge_cls in model_info:
        st.markdown(f"""
        <div class="card" style="padding: 0.9rem 1.2rem; margin-bottom: 0.5rem;">
            <span class="badge {badge_cls}" style="margin-bottom:0.4rem;display:inline-block">{name}</span>
            <div style="color:rgba(255,255,255,0.45); font-size:0.78rem; margin-top:0.25rem">{detail}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Currently viewing ────────────────────────────────────────────────────
    if run_btn:
        st.markdown('<div class="section-title">🎯 Current Query</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card" style="padding: 0.9rem 1.2rem;">
            <div style="color:rgba(255,255,255,0.45); font-size:0.75rem; margin-bottom:0.3rem">USER</div>
            <div style="color:#e2e8f0; font-weight:600; font-size:0.9rem">#{selected_user}</div>
            <div style="color:rgba(255,255,255,0.45); font-size:0.75rem; margin: 0.6rem 0 0.3rem">SEED MOVIE</div>
            <div style="color:#e2e8f0; font-weight:600; font-size:0.9rem">{selected_movie}</div>
        </div>
        """, unsafe_allow_html=True)
