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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* ── App background ── */
.stApp {
    background: #080812;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(124,58,237,0.25) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(59,130,246,0.18) 0%, transparent 60%);
    min-height: 100vh;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.3); border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10,10,25,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] .stSlider > label { color: rgba(255,255,255,0.5) !important; font-size:0.78rem !important; font-weight:500 !important; letter-spacing:0.8px !important; text-transform:uppercase !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    font-size: 0.88rem !important; padding: 0.65rem 1.4rem !important;
    width: 100% !important; letter-spacing: 0.3px !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.55) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #a78bfa !important; }

/* ── HR ── */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.2rem 0 !important; }
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
    <div style="text-align:center; padding: 1.5rem 0 1rem;">
        <div style="font-size:2.4rem; margin-bottom:0.3rem;">🎬</div>
        <div style="font-size:1.4rem; font-weight:800; background:linear-gradient(90deg,#a78bfa,#60a5fa);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">CineMatch</div>
        <div style="font-size:0.72rem; color:rgba(255,255,255,0.35); letter-spacing:1px; margin-top:3px;">
            SMART RECOMMENDER
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Strategy Selection first
    st.markdown('<div style="font-size:0.72rem;color:rgba(255,255,255,0.35);letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem;">🔀 Strategy</div>', unsafe_allow_html=True)
    strategy = st.radio("Strategy", ["Hybrid", "Content-Based", "Collaborative Filtering"], label_visibility="collapsed")
    
    st.markdown("---")

    # Dynamic Inputs
    selected_user = None
    selected_movie = None
    
    if strategy in ["Hybrid", "Collaborative Filtering"]:
        st.markdown('<div style="font-size:0.72rem;color:rgba(255,255,255,0.35);letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.5rem;">👤 User Profile</div>', unsafe_allow_html=True)
        selected_user = st.selectbox("User ID", sorted(ratings['userId'].unique()), label_visibility="collapsed")

    if strategy in ["Hybrid", "Content-Based"]:
        st.markdown('<div style="font-size:0.72rem;color:rgba(255,255,255,0.35);letter-spacing:0.8px;text-transform:uppercase;margin:0.8rem 0 0.5rem;">🎥 Seed Movie</div>', unsafe_allow_html=True)
        movie_titles   = sorted(movies['title'].dropna().unique())
        selected_movie = st.selectbox("Movie", movie_titles, label_visibility="collapsed")

    st.markdown('<div style="font-size:0.72rem;color:rgba(255,255,255,0.35);letter-spacing:0.8px;text-transform:uppercase;margin:0.8rem 0 0.5rem;">🎯 Top N</div>', unsafe_allow_html=True)
    top_n = st.slider("Top N", 5, 20, 10, label_visibility="collapsed")

    st.markdown("---")
    run_btn  = st.button("✨  Get Recommendations")
    eval_btn = st.button("📊  Evaluate Models")

    # ── Quick stats ──
    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem;color:rgba(255,255,255,0.35);letter-spacing:0.8px;text-transform:uppercase;margin-bottom:0.8rem;">📁 Dataset Stats</div>', unsafe_allow_html=True)
    stats = [("Movies", f"{movies.shape[0]:,}"), ("Users", f"{ratings['userId'].nunique():,}"),
             ("Ratings", f"{len(ratings):,}"), ("Genres", str(len(set(g for gl in movies['genre_list'] for g in gl))))]
    for label, val in stats:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:0.35rem 0;border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="color:rgba(255,255,255,0.4);font-size:0.8rem;">{label}</span>
            <span style="color:#a78bfa;font-weight:700;font-size:0.82rem;">{val}</span>
        </div>""", unsafe_allow_html=True)

# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2.5rem 0 2rem; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 2rem;">
    <h1 style="font-size:2.6rem;font-weight:800;margin:0 0 0.4rem;
        background:linear-gradient(90deg,#fff 0%,rgba(255,255,255,0.7) 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Movie Recommendations
    </h1>
    <p style="color:rgba(255,255,255,0.35);margin:0;font-size:0.95rem;">
        Powered by TF-IDF · Cosine Similarity · Singular Value Decomposition
    </p>
</div>
""", unsafe_allow_html=True)

col_main, col_info = st.columns([3, 1], gap="large")

def render_cards(recs, score_col, score_label, accent, max_score=1.0):
    if recs is None or (hasattr(recs, 'empty') and recs.empty):
        st.warning("⚠️  No recommendations found.")
        return
    for rank, (_, row) in enumerate(recs.iterrows(), 1):
        score = float(row.get(score_col, 0))
        genres_html = genre_badges(row.get('genres', ''))
        bar = score_bar(score, max_score, accent)
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 16px; padding: 1.1rem 1.4rem;
            margin-bottom: 0.7rem;
            display: flex; align-items: center; gap: 1.2rem;
            transition: all 0.25s ease;
            ">
            <!-- Rank -->
            <div style="min-width:2.4rem; text-align:center;">
                <span style="font-size:1.4rem;font-weight:800;
                    background:linear-gradient(135deg,{accent},{accent}88);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                    {rank:02d}
                </span>
            </div>
            <!-- Separator -->
            <div style="width:2px;height:48px;background:linear-gradient({accent},{accent}22);border-radius:2px;flex-shrink:0;"></div>
            <!-- Info -->
            <div style="flex:1; min-width:0;">
                <div style="font-size:1rem;font-weight:700;color:#f1f5f9;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:4px;">
                    {row['title']}
                </div>
                <div>{genres_html}</div>
                {bar}
            </div>
            <!-- Score -->
            <div style="text-align:right; flex-shrink:0; padding-left:1rem;">
                <div style="font-size:1.35rem;font-weight:800;color:{accent};">{score:.3f}</div>
                <div style="font-size:0.68rem;color:rgba(255,255,255,0.35);margin-top:2px;letter-spacing:0.5px;">{score_label.upper()}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


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
            render_cards(recs_cb, "similarity", "Similarity", "#a78bfa")

        elif strategy == "Collaborative Filtering":
            with st.spinner("Running SVD predictions…"):
                recs_cf = collab_model.recommend(selected_user, movies, top_n=top_n)
            render_cards(recs_cf, "est_rating", "SVD Score", "#60a5fa", max_score=5.0)

        elif strategy == "Hybrid":
            with st.spinner("Running hybrid model…"):
                recs_hy = hybrid_model.recommend(selected_user, selected_movie, top_n=top_n)
            render_cards(recs_hy, "hybrid_score", "Hybrid Score", "#f472b6")

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
        ("RMSE",         metrics["RMSE"],         "#ef4444", "Prediction error",  "Lower is better"),
        ("MAE",          metrics["MAE"],           "#f97316", "Absolute error",    "Lower is better"),
        ("Precision@10", metrics["Precision@K"],   "#22c55e", "Relevant in top 10","Higher is better"),
        ("Recall@10",    metrics["Recall@K"],      "#3b82f6", "Coverage of relevant","Higher is better"),
        ("F1-Score",     metrics["F1-Score"],      "#a78bfa", "Harmonic mean",     "P-R balance"),
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
