"""Shared layout for the local app."""

import streamlit as st


def apply_style():
    dark = st.context.theme.type == "dark"
    background, surface, ink, muted, line, accent = (
        ("#101923", "#182431", "#e9f0f6", "#9cabbc", "#304151", "#75d3ce")
        if dark else
        ("#f5f7fa", "#ffffff", "#203345", "#6e7d8c", "#dfe6ed", "#137d8d")
    )
    st.html(f"""<style>
    .stApp {{ --dfa-surface:{surface}; --dfa-ink:{ink}; --dfa-muted:{muted};
        --dfa-line:{line}; --dfa-accent:{accent}; background:{background}; }}
    [data-testid="stHeader"] {{ background:transparent; }}
    [data-testid="stAppDeployButton"] {{ display:none; }}
    [data-testid="stMainBlockContainer"] {{ max-width:1500px; padding:2rem 2.5rem 4rem; }}
    [data-testid="stSidebar"] {{ background:{surface}; border-right:1px solid {line}; }}
    [data-testid="stSidebarUserContent"] {{ padding-top:2.4rem; }}
    h1, h2, h3 {{ color:{ink}; letter-spacing:-.035em; }}
    h1 {{ font-size:2.1rem !important; font-weight:650 !important; padding-bottom:.3rem !important; }}
    h3 {{ font-size:1.15rem !important; letter-spacing:-.02em; }}
    .dfa-eyebrow {{ color:{accent}; font-size:.72rem; font-weight:700;
        letter-spacing:.15em; text-transform:uppercase; margin-bottom:.35rem; }}
    .dfa-version {{ display:inline-block; color:{muted}; border:1px solid {line};
        border-radius:20px; padding:.3rem .7rem; font-size:.75rem; margin-top:.7rem; }}
    .dfa-workspace {{ color:{ink}; font-size:1.2rem; font-weight:650; margin-bottom:1.4rem; }}
    .dfa-section {{ color:{muted}; font-size:.72rem; font-weight:650;
        letter-spacing:.12em; text-transform:uppercase; margin-top:.6rem; }}
    [data-testid="stTabs"] [role="tablist"] {{ gap:1.7rem; margin-bottom:.5rem; }}
    [data-testid="stTabs"] [role="tab"] {{ padding:.65rem 0; font-weight:550; }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ color:{accent}; }}
    .st-key-structure_panel, .st-key-task_panel {{ background:{surface}; border-radius:16px; }}
    .st-key-structure_panel > div, .st-key-task_panel > div {{ border-color:{line} !important; }}
    [data-testid="stForm"] {{ background:{surface}; border-color:{line}; border-radius:14px; }}
    [data-testid="stExpander"] details {{ border-color:{line}; border-radius:10px; }}
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {{ border-radius:9px; }}
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {{
        border:1px solid {line} !important; border-radius:8px; }}
    [data-testid="stTextArea"] textarea:focus, [data-testid="stTextInput"] input:focus {{
        border-color:{accent} !important; }}
    [data-testid="stBaseButton-primary"] {{ background:{accent}; border-color:{accent}; color:{surface}; }}
    [data-testid="stBaseButton-primary"]:disabled {{ opacity:.45; }}
    [data-testid="stMetricValue"] {{ font-size:1.4rem; }}
    .st-key-cluster_engine [role="radiogroup"] {{ display:inline-flex; gap:4px;
        padding:5px; background:{surface}; border:1px solid {line}; border-radius:12px; }}
    .st-key-cluster_engine [data-testid="stRadioOption"] {{ padding:.55rem 1.05rem;
        border-radius:8px; color:{muted}; transition:background .15s; }}
    .st-key-cluster_engine [data-testid="stRadioOption"][data-selected="true"] {{
        background:{ink}; color:{surface}; }}
    .st-key-cluster_engine [data-testid="stRadioOption"] > div > div:first-child {{ display:none; }}
    .st-key-cluster_engine [data-testid="stRadioOption"] p {{ font-weight:600; font-size:.9rem; }}
    .st-key-cluster_engine [data-testid="stRadioOption"]:focus-within {{ outline:2px solid {accent}; outline-offset:2px; }}
    .st-key-calculation_goal textarea, .st-key-qe_goal textarea {{ min-height:160px; }}
    @media (max-width:900px) {{
        [data-testid="stMainBlockContainer"] {{ padding:2rem 1.2rem; }}
        .st-key-calculation_workspace > [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap; }}
        .st-key-calculation_workspace > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{ width:100%; flex:1 1 100%; }}
    }}
    </style>""")
