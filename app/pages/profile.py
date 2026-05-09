import streamlit as st
import pandas as pd
from app.ui import page_header, render_movie_grid, genre_badges

def render(movies, ratings):
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
    
    genre_counts = pd.Series(
        [g for glist in user_r["genres"].dropna().str.split("|") for g in glist if g != "(no genres listed)"]
    ).value_counts()
    top_genre = genre_counts.index[0] if len(genre_counts) > 0 else "N/A"
    stat_card(col3, "Top Genre", top_genre, "#f472b6")
    
    fav_decade = (user_r['year'] // 10 * 10).mode().iloc[0] if not user_r['year'].isna().all() else "N/A"
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
