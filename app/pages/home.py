import streamlit as st
import random
from app.ui import page_header, render_movie_grid

def render(movies, ratings):
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
        selected_genres = st.multiselect("What are you in the mood for?", all_genres, default=["Action", "Sci-Fi"] if "Action" in all_genres else [])
        st.session_state.current_mood = selected_genres

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.4rem;font-weight:700;color:#e2e8f0;margin-bottom:1.5rem;">🔥 Trending Now</div>', unsafe_allow_html=True)
    
    # Simple trending logic (most rated in dataset)
    if 'trending' not in st.session_state:
        trending = movies.sort_values(by='rating_count', ascending=False).head(10)
        st.session_state.trending = trending
        
    render_movie_grid(st.session_state.trending, score_col=None, score_label=None, accent="#f472b6", show_rank=False)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
    with col_btn2:
        if st.button("Continue to Profile ➔", key="home_continue"):
            st.session_state.page = "Profile"
            st.rerun()
