import streamlit as st
import pandas as pd
from app.ui import page_header, render_movie_grid

def render(movies, ratings, hybrid_model):
    page_header("Hybrid Intelligence", "The ultimate personalized movie mix.", "#f472b6")
    
    user_id = st.session_state.get('active_user', sorted(ratings['userId'].unique())[0])
    movie_titles = sorted(movies['title'].dropna().unique())
    
    # Taste summary
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
            
            # Show the math visually for the top recommendation
            top_rec = recs_hy.iloc[0]
            sim_score = top_rec.get('similarity', 0)
            cf_score = top_rec.get('est_rating', 0)
            cf_norm = min(1.0, max(0, cf_score / 5.0)) # Normalize CF score for display
            hy_score = top_rec.get('hybrid_score', 0)
            
            import textwrap
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
