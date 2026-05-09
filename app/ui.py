import streamlit as st
import requests
import base64
import re

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
    @keyframes fade-up {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
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
                img_html = f'<img src="{poster_url}" style="width:100%;height:240px;object-fit:cover;display:block;border-radius:0;">'
            else:
                color2 = accent + "55"
                first_letter = title[0].upper() if title else "?"
                img_html = f"""
                <div style="width:100%;height:240px;display:flex;align-items:center;justify-content:center;
                    background:linear-gradient(135deg,{accent}33,{color2},rgba(0,0,0,0.5));
                    font-size:3.5rem;font-weight:800;color:{accent};">
                    {first_letter}
                </div>"""

            with col:
                rank_badge = f'<div style="position:absolute;top:10px;left:10px;background:rgba(0,0,0,0.75);backdrop-filter:blur(8px);border:1px solid {accent}66;color:{accent};font-size:0.72rem;font-weight:800;padding:3px 9px;border-radius:50px;letter-spacing:0.5px;">#{rank:02d}</div>' if show_rank else ""
                score_badge = f'<div style="position:absolute;top:10px;right:10px;background:rgba(0,0,0,0.75);backdrop-filter:blur(8px);border:1px solid {accent}66;color:{accent};font-size:0.72rem;font-weight:800;padding:3px 9px;border-radius:50px;">{score:.2f}</div>' if score_col else ""
                score_bar = f'<div style="background:rgba(255,255,255,0.08);border-radius:50px;height:3px;width:100%;"><div style="background:{accent};height:3px;border-radius:50px;width:{pct:.1f}%;box-shadow:0 0 6px {accent};"></div></div>' if score_col else ""

                card = (
                    f'<div style="position:relative;border-radius:14px;overflow:hidden;background:#0d0d1a;'
                    f'border:1px solid rgba(255,255,255,0.07);box-shadow:0 4px 20px rgba(0,0,0,0.5);'
                    f'transition:all 0.3s ease;cursor:pointer;margin-bottom:1rem;"'
                    f' onmouseover="this.style.transform=\'translateY(-6px)\';this.style.boxShadow=\'0 16px 40px rgba(0,0,0,0.7),0 0 0 1px {accent}55\';"'
                    f' onmouseout="this.style.transform=\'\';this.style.boxShadow=\'0 4px 20px rgba(0,0,0,0.5)\';">'
                    f'{img_html}'
                    f'{rank_badge}'
                    f'{score_badge}'
                    f'<div style="padding:0.9rem 0.8rem 0.8rem;">'
                    f'<div style="font-size:0.88rem;font-weight:700;color:#f1f5f9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:4px;" title="{title}">{title}</div>'
                    f'<div style="font-size:0.7rem;color:rgba(255,255,255,0.4);margin-bottom:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{genres_clean}</div>'
                    f'{score_bar}'
                    f'</div></div>'
                )
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
