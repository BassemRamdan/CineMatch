import streamlit as st
from app.ui import page_header
from app.main import evaluate_models # I'll need to define this in main.py, or import the function directly
from src.evaluation import run_full_evaluation

def render(test_data, collab_model):
    page_header("System Analytics", "Model Performance & Evaluation Metrics", "#eab308")
    
    st.markdown("""
    <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1.2rem; display:flex;align-items:center;gap:0.5rem;">
        📊 Model Evaluation <span style="font-size:0.75rem;font-weight:500;color:rgba(255,255,255,0.35);margin-left:4px;">— Collaborative SVD on 20% test split</span>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Evaluating models... this takes a few seconds."):
        # Cache this so it doesn't run every time
        @st.cache_data
        def get_metrics(_model, _test):
            return run_full_evaluation(_model, _test)
            
        metrics = get_metrics(collab_model, test_data)

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
