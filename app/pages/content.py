import streamlit as st
from app.ui import page_header, render_movie_grid

def render(movies, content_model):
    page_header("Content-Based Match", "Because you liked this movie...", "#22c55e")
    
    movie_titles = sorted(movies['title'].dropna().unique())
    
    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.markdown('<div style="font-size:0.9rem;font-weight:600;color:rgba(255,255,255,0.6);margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:1px;">Search for a movie</div>', unsafe_allow_html=True)
        selected_movie = st.selectbox("Select a movie", movie_titles, label_visibility="collapsed", key="cb_movie")
    with col2:
        st.markdown('<div style="font-size:0.9rem;font-weight:600;color:rgba(255,255,255,0.6);margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:1px;">Results</div>', unsafe_allow_html=True)
        top_n = st.slider("Top N", 5, 20, 10, label_visibility="collapsed", key="cb_topn")
        
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if selected_movie:
        with st.spinner("Finding similar movies..."):
            recs_cb = content_model.recommend(selected_movie, top_n=top_n)
            
        st.markdown(f'<div style="font-size:1.4rem;font-weight:700;color:#e2e8f0;margin-bottom:1.5rem;">Similar to <span style="color:#22c55e;">{selected_movie}</span></div>', unsafe_allow_html=True)
        render_movie_grid(recs_cb, "similarity", "Similarity", "#22c55e", show_rank=True)
