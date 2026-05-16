import streamlit as st

def apply_styles():
    st.set_page_config(page_title="DeepLens", page_icon="🔭", layout="centered")
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    html, body, .stApp { background-color: #080b12; color: #e2e8f0; font-family: 'Space Grotesk', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .main-title { font-size: 2.4rem; font-weight: 700; color: #fff; text-align: center; margin-top: 36px; margin-bottom: 4px; letter-spacing: -0.5px; }
    .sub-title { font-size: 0.95rem; color: #4b5563; text-align: center; margin-bottom: 28px; }
    .topic-badge { display: inline-block; background: #0f1623; border: 1px solid #1e3a5f; color: #60a5fa; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; margin: 3px; font-family: 'DM Mono', monospace; }
    .stChatMessage { background-color: #0f1623 !important; border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    .source-box { background: #0a0f1c; border-left: 3px solid #2563eb; padding: 10px 14px; border-radius: 8px; margin-top: 6px; font-size: 0.82rem; color: #94a3b8; }
    .doc-badge { background: #1d4ed8; color: #fff; padding: 2px 8px; border-radius: 10px; font-size: 0.72rem; margin-right: 5px; }
    .cache-badge { background: #065f46; color: #6ee7b7; padding: 2px 8px; border-radius: 10px; font-size: 0.72rem; margin-left: 6px; font-family: 'DM Mono', monospace; }
    .rewrite-badge { background: #1e1a4f; color: #a5b4fc; padding: 2px 8px; border-radius: 10px; font-size: 0.72rem; margin-left: 6px; font-family: 'DM Mono', monospace; }
    .halluc-warning { background: #1c0a0a; border-left: 3px solid #dc2626; padding: 8px 12px; border-radius: 6px; color: #fca5a5; font-size: 0.8rem; margin-top: 6px; }
    .confidence-bar-wrap { margin: 6px 0; }
    .confidence-label { font-size: 0.75rem; color: #64748b; margin-bottom: 2px; }
    .confidence-bar-bg { background: #1e293b; border-radius: 4px; height: 6px; }
    .confidence-bar-fill { height: 6px; border-radius: 4px; transition: width 0.4s; }
    .feedback-row { display: flex; gap: 8px; margin-top: 6px; }
    .summary-box { background: #0f1a2e; border-left: 3px solid #1d4ed8; padding: 6px 10px; border-radius: 6px; font-size: 0.75rem; color: #93c5fd; margin-top: 4px; font-style: italic; }
    .error-box { background: #1c0a0a; border-left: 3px solid #dc2626; padding: 10px 14px; border-radius: 8px; color: #fca5a5; font-size: 0.85rem; }
    section[data-testid="stSidebar"] { background: #090d16; }
    section[data-testid="stSidebar"] * { font-size: 0.78rem !important; }
    section[data-testid="stSidebar"] h2 { font-size: 0.92rem !important; color: #60a5fa !important; }
    section[data-testid="stSidebar"] h3 { font-size: 0.82rem !important; }
</style>""", unsafe_allow_html=True)