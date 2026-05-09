import streamlit as st
import os
import sys

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_and_preprocess_data
from src.content_engine import ContentBasedRecommender
from src.collaborative_engine import CollaborativeRecommender
from src.hybrid_engine import HybridRecommender
from app.ui import apply_custom_css

# Import pages
from app.pages import home, profile, content, hybrid, analytics

st.set_page_config(
    page_title="CineMatch Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global CSS
apply_custom_css()

# Initialize session state for router
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# ─── Data & Models ────────────────────────────────────────────────────────────
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

# ─── Sidebar Navigation ──────────────────────────────────────────────────────
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
    
    # Custom Navigation Buttons
    nav_items = [
        ("Home", "🏠 Home"),
        ("Profile", "👤 User Profile"),
        ("Content", "🎥 Content-Based"),
        ("Hybrid", "🔮 Hybrid Engine"),
        ("Analytics", "📊 Analytics")
    ]
    
    for page_id, label in nav_items:
        active_class = "nav-btn-active" if st.session_state.page == page_id else "nav-btn"
        st.markdown(f'<div class="{active_class}">', unsafe_allow_html=True)
        if st.button(label, key=f"nav_{page_id}", use_container_width=True):
            st.session_state.page = page_id
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Stats Footer
    st.markdown("<hr style='margin:2rem 0 1.5rem 0 !important;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;color:rgba(167,139,250,0.6);letter-spacing:2px;text-transform:uppercase;font-weight:600;margin-bottom:0.8rem;padding-left:1rem;">Database</div>', unsafe_allow_html=True)
    stats = [("Movies", f"{movies.shape[0]:,}"), ("Users", f"{ratings['userId'].nunique():,}"), ("Ratings", f"{len(ratings):,}")]
    for label, val in stats:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:0.35rem 1rem;border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="color:rgba(255,255,255,0.4);font-size:0.8rem;">{label}</span>
            <span style="color:#a78bfa;font-weight:700;font-size:0.82rem;">{val}</span>
        </div>""", unsafe_allow_html=True)

# ─── Router Logic ───────────────────────────────────────────────────────────
if st.session_state.page == "Home":
    home.render(movies, ratings)
elif st.session_state.page == "Profile":
    profile.render(movies, ratings)
elif st.session_state.page == "Content":
    content.render(movies, content_model)
elif st.session_state.page == "Hybrid":
    hybrid.render(movies, ratings, hybrid_model)
elif st.session_state.page == "Analytics":
    analytics.render(test, collab_model)
