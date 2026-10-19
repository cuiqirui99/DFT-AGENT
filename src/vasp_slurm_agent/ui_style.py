"""Shared look of the local app: one small set of design tokens, applied to Streamlit's widgets.

Light mode is warm paper with a deep teal accent; dark mode keeps the same
hierarchy. Native system fonts keep the app offline-friendly and familiar on
each platform. Classes prefixed `dfa-` are the app's own small components.
"""

import streamlit as st

LOGO = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
<circle cx="12" cy="12" r="2.3" fill="var(--dfa-accent)"/>
<ellipse cx="12" cy="12" rx="9.5" ry="4" stroke="var(--dfa-accent)" stroke-width="1.4"/>
<ellipse cx="12" cy="12" rx="9.5" ry="4" stroke="var(--dfa-accent)" stroke-width="1.4" transform="rotate(60 12 12)"/>
<ellipse cx="12" cy="12" rx="9.5" ry="4" stroke="var(--dfa-accent)" stroke-width="1.4" transform="rotate(120 12 12)"/>
</svg>"""

_LIGHT = dict(bg="#f6f5f1", surface="#ffffff", raised="#fbfaf7", ink="#1d2733", muted="#6b7684", faint="#9aa3ae",
              line="#e4e1da", line_strong="#cfcbc2", accent="#0f766e", accent_strong="#0b5d57", accent_soft="#e3f1ee",
              ok="#15803d", ok_soft="#e7f5ec", warn="#b45309", warn_soft="#fbf0df", bad="#b91c1c", bad_soft="#fbe9e9",
              code="#f1efe9", shadow="0 1px 2px rgba(29,39,51,.05), 0 8px 24px -16px rgba(29,39,51,.25)")
_DARK = dict(bg="#101317", surface="#171b21", raised="#1c2128", ink="#e7eaee", muted="#9aa4b1", faint="#6f7a88",
             line="#262d36", line_strong="#36404b", accent="#5fd3c4", accent_strong="#8ae3d8", accent_soft="#15302e",
             ok="#4ade80", ok_soft="#14281c", warn="#fbbf24", warn_soft="#2c2414", bad="#f87171", bad_soft="#2d1818",
             code="#1f252d", shadow="0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8)")

FONT = ('-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", "Helvetica Neue", "PingFang SC", '
        '"Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif')
MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace'


def tokens():
    return _DARK if st.context.theme.type == "dark" else _LIGHT


def apply_style():
    c = tokens()
    st.html(f"""<style>
    .stApp {{
        --dfa-bg:{c['bg']}; --dfa-surface:{c['surface']}; --dfa-raised:{c['raised']}; --dfa-ink:{c['ink']};
        --dfa-muted:{c['muted']}; --dfa-faint:{c['faint']}; --dfa-line:{c['line']}; --dfa-line-strong:{c['line_strong']};
        --dfa-accent:{c['accent']}; --dfa-accent-strong:{c['accent_strong']}; --dfa-accent-soft:{c['accent_soft']};
        --dfa-ok:{c['ok']}; --dfa-ok-soft:{c['ok_soft']}; --dfa-warn:{c['warn']}; --dfa-warn-soft:{c['warn_soft']};
        --dfa-bad:{c['bad']}; --dfa-bad-soft:{c['bad_soft']}; --dfa-code:{c['code']}; --dfa-shadow:{c['shadow']};
        background:var(--dfa-bg); color:var(--dfa-ink);
        font-family:{FONT}; font-size:15px; -webkit-font-smoothing:antialiased; }}
    .stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label, .stApp button, .stApp input, .stApp textarea,
    .stApp select, .stApp td, .stApp th, .stApp summary, .stApp [data-testid="stMarkdownContainer"] {{ font-family:{FONT}; }}
    /* Streamlit's icons are a font too; never override them. */
    .stApp [class*="material-symbols"], .stApp [data-testid="stIconMaterial"] {{ font-family:"Material Symbols Rounded" !important; }}
    code, pre, kbd, [data-testid="stCode"] *, .stApp [data-testid="stMarkdownContainer"] code {{ font-family:{MONO}; }}
    [data-testid="stMarkdownContainer"] p {{ font-size:15px; line-height:1.65; }}
    [data-testid="stMarkdownContainer"] code {{ background:var(--dfa-code); border-radius:5px; padding:.1em .35em; font-size:.88em; color:var(--dfa-ink); }}
    [data-testid="stHeader"] {{ background:transparent; }}
    [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"] {{ display:none; }}
    [data-testid="stMainBlockContainer"] {{ max-width:1500px; padding:1.6rem 2.5rem 4rem; }}
    h1, h2, h3, h4 {{ color:var(--dfa-ink); letter-spacing:-.02em; font-weight:600; }}
    h1 {{ font-size:1.9rem !important; padding-bottom:.2rem !important; }}
    h2 {{ font-size:1.45rem !important; padding:.2rem 0 .6rem !important; }}
    h3 {{ font-size:1.12rem !important; padding:.3rem 0 .4rem !important; }}
    [data-testid="stCaptionContainer"] p {{ color:var(--dfa-muted); font-size:.86rem; line-height:1.5; }}

    /* sidebar */
    [data-testid="stSidebar"] {{ background:var(--dfa-surface); border-right:1px solid var(--dfa-line); }}
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding:1.2rem 1rem 1rem; }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{ font-size:1rem !important; margin:0; padding:0 .4rem .4rem !important; }}
    .dfa-brand {{ display:flex; align-items:center; gap:.55rem; padding:.2rem .45rem .9rem; color:var(--dfa-ink); font-weight:650; font-size:1rem; letter-spacing:-.01em; }}
    .dfa-brand small {{ color:var(--dfa-faint); font-weight:500; font-size:.72rem; margin-left:auto; }}
    .dfa-eyebrow {{ color:var(--dfa-accent); font-size:.72rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; margin-bottom:.35rem; }}
    .dfa-workspace {{ color:var(--dfa-ink); font-size:1.2rem; font-weight:650; margin-bottom:1.4rem; }}
    .dfa-section, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ color:var(--dfa-faint); font-size:.68rem; font-weight:650; letter-spacing:.12em; text-transform:uppercase; margin-top:.4rem; }}
    .dfa-version {{ display:inline-block; color:var(--dfa-muted); border:1px solid var(--dfa-line); border-radius:20px; padding:.3rem .7rem; font-size:.75rem; margin-top:.7rem; }}

    /* buttons and inputs */
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {{
        border-radius:9px; border:1px solid var(--dfa-line-strong); background:var(--dfa-surface); color:var(--dfa-ink);
        font-weight:500; min-height:2.3rem; padding:.3rem .9rem; transition:background .12s, border-color .12s, box-shadow .12s; }}
    [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {{ border-color:var(--dfa-accent); color:var(--dfa-accent-strong); background:var(--dfa-raised); }}
    [data-testid="stButton"] button:focus-visible {{ outline:none; box-shadow:0 0 0 3px var(--dfa-accent-soft); }}
    [data-testid="stBaseButton-primary"], [data-testid="stButton"] button[kind="primary"] {{ background:var(--dfa-accent) !important; border-color:var(--dfa-accent) !important; color:#fff !important; font-weight:600; }}
    [data-testid="stBaseButton-primary"]:hover, [data-testid="stButton"] button[kind="primary"]:hover {{ background:var(--dfa-accent-strong) !important; border-color:var(--dfa-accent-strong) !important; }}
    [data-testid="stBaseButton-primary"]:disabled {{ opacity:.45; }}
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] > div > div, [data-testid="stMultiSelect"] > div > div {{
        border:1px solid var(--dfa-line-strong) !important; border-radius:9px; background:var(--dfa-surface); color:var(--dfa-ink); }}
    [data-testid="stTextArea"] textarea:focus, [data-testid="stTextInput"] input:focus {{ border-color:var(--dfa-accent) !important; box-shadow:0 0 0 3px var(--dfa-accent-soft); }}
    [data-testid="stTextInput"] > div > div, [data-testid="stTextArea"] > div > div {{ border:none; background:transparent; }}
    [data-testid="stPills"] button {{ border-radius:999px; font-size:.86rem; padding:.25rem .8rem; border:1px solid var(--dfa-line-strong); background:var(--dfa-surface); color:var(--dfa-ink); }}
    [data-testid="stPills"] button[kind="pillsActive"], [data-testid="stPills"] button[aria-checked="true"] {{ background:var(--dfa-accent-soft); border-color:var(--dfa-accent); color:var(--dfa-accent-strong); }}
    [data-testid="stCheckbox"] p, [data-testid="stRadio"] p {{ font-size:.92rem; }}

    /* surfaces */
    [data-testid="stForm"] {{ background:var(--dfa-surface); border-color:var(--dfa-line); border-radius:14px; }}
    [data-testid="stExpander"] details {{ border:1px solid var(--dfa-line); border-radius:10px; background:var(--dfa-surface); }}
    [data-testid="stExpander"] summary {{ font-size:.92rem; color:var(--dfa-ink); }}
    [data-testid="stExpander"] summary:hover {{ color:var(--dfa-accent-strong); }}
    [data-testid="stVerticalBlockBorderWrapper"] > div {{ border-color:var(--dfa-line) !important; border-radius:14px !important; background:var(--dfa-surface); }}
    [data-testid="stPopover"] button, [data-testid="stPopoverButton"] {{ border-radius:9px; }}
    [data-testid="stAlert"] {{ border-radius:10px; }}
    [data-testid="stCode"] pre, [data-testid="stCodeBlock"] pre {{ background:var(--dfa-code) !important; border:1px solid var(--dfa-line); border-radius:9px; font-size:.84rem; }}
    [data-testid="stDataFrame"] {{ border:1px solid var(--dfa-line); border-radius:10px; overflow:hidden; }}
    [data-testid="stMetricValue"] {{ font-size:1.4rem; }}
    [data-testid="stTabs"] [role="tablist"] {{ gap:1.7rem; margin-bottom:.5rem; border-bottom:1px solid var(--dfa-line); }}
    [data-testid="stTabs"] [role="tab"] {{ padding:.65rem 0; font-weight:550; color:var(--dfa-muted); }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ color:var(--dfa-accent-strong); }}
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background:var(--dfa-accent); }}

    /* chat */
    [data-testid="stChatMessage"] {{ background:transparent; padding:.4rem 0; gap:.6rem; }}
    [data-testid^="stChatMessageAvatar"] {{ display:none; }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ justify-content:flex-end; }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > [data-testid="stChatMessageContent"] {{
        background:var(--dfa-accent-soft); color:var(--dfa-ink); border-radius:16px 16px 4px 16px; padding:.7rem 1rem; max-width:78%; flex:0 1 auto; }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > [data-testid="stChatMessageContent"] {{ padding:.3rem 0; }}
    [data-testid="stChatMessage"] p {{ line-height:1.7; }}
    [data-testid="stChatMessage"] table {{ font-size:.9rem; }}
    [data-testid="stBottom"] > div {{ background:transparent; }}
    [data-testid="stBottomBlockContainer"] {{ background:linear-gradient(to bottom, transparent, var(--dfa-bg) 30%); padding-top:1rem; }}
    [data-testid="stChatInput"] {{ border:1px solid var(--dfa-line-strong); border-radius:16px; background:var(--dfa-surface); box-shadow:var(--dfa-shadow); }}
    [data-testid="stChatInput"]:focus-within {{ border-color:var(--dfa-accent); box-shadow:0 0 0 3px var(--dfa-accent-soft); }}
    [data-testid="stChatInput"] textarea {{ font-size:.96rem; }}

    /* app components */
    .dfa-chip {{ display:inline-block; font-size:.74rem; font-weight:600; letter-spacing:.02em; padding:.14rem .55rem; border-radius:999px;
        background:var(--dfa-code); color:var(--dfa-muted); vertical-align:middle; }}
    .dfa-chip.ok {{ background:var(--dfa-ok-soft); color:var(--dfa-ok); }}
    .dfa-chip.warn {{ background:var(--dfa-warn-soft); color:var(--dfa-warn); }}
    .dfa-chip.bad {{ background:var(--dfa-bad-soft); color:var(--dfa-bad); }}
    .dfa-chip.accent {{ background:var(--dfa-accent-soft); color:var(--dfa-accent-strong); }}
    .dfa-tag {{ display:inline-block; font-family:{MONO}; font-size:.72rem; padding:.1rem .45rem; border:1px solid var(--dfa-line); border-radius:6px; color:var(--dfa-muted); vertical-align:middle; }}
    .dfa-step {{ display:flex; gap:.8rem; align-items:flex-start; padding:.9rem 1rem; border:1px solid var(--dfa-line); border-radius:14px; background:var(--dfa-surface); min-height:7.4rem; }}
    .dfa-step .n {{ flex:0 0 1.7rem; height:1.7rem; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:.8rem; font-weight:700;
        border:1.5px solid var(--dfa-line-strong); color:var(--dfa-muted); background:var(--dfa-raised); }}
    .dfa-step.done .n {{ background:var(--dfa-accent); border-color:var(--dfa-accent); color:#fff; }}
    .dfa-step.now .n {{ border-color:var(--dfa-accent); color:var(--dfa-accent-strong); }}
    .dfa-step .t {{ font-weight:600; font-size:.95rem; color:var(--dfa-ink); margin-bottom:.15rem; }}
    .dfa-step .s {{ font-size:.86rem; color:var(--dfa-muted); line-height:1.45; }}
    .dfa-step.done .s {{ color:var(--dfa-ok); }}
    .dfa-hello {{ font-size:1.75rem; font-weight:600; letter-spacing:-.03em; color:var(--dfa-ink); margin:2.5rem 0 .3rem; }}
    .dfa-sub {{ color:var(--dfa-muted); font-size:.98rem; margin-bottom:1.4rem; }}
    .dfa-muted {{ color:var(--dfa-muted); }}

    /* legacy workbench */
    .st-key-structure_panel, .st-key-task_panel {{ background:var(--dfa-surface); border-radius:16px; }}
    .st-key-structure_panel > div, .st-key-task_panel > div {{ border-color:var(--dfa-line) !important; }}
    .st-key-cluster_engine [role="radiogroup"] {{ display:inline-flex; gap:4px; padding:5px; background:var(--dfa-surface); border:1px solid var(--dfa-line); border-radius:12px; }}
    .st-key-cluster_engine [data-testid="stRadioOption"] {{ padding:.55rem 1.05rem; border-radius:8px; color:var(--dfa-muted); transition:background .15s; }}
    .st-key-cluster_engine [data-testid="stRadioOption"][data-selected="true"] {{ background:var(--dfa-ink); color:var(--dfa-surface); }}
    .st-key-cluster_engine [data-testid="stRadioOption"] > div > div:first-child {{ display:none; }}
    .st-key-cluster_engine [data-testid="stRadioOption"] p {{ font-weight:600; font-size:.9rem; }}
    .st-key-cluster_engine [data-testid="stRadioOption"]:focus-within {{ outline:2px solid var(--dfa-accent); outline-offset:2px; }}
    .st-key-calculation_goal textarea, .st-key-qe_goal textarea {{ min-height:160px; }}
    @media (max-width:900px) {{
        [data-testid="stMainBlockContainer"] {{ padding:1.5rem 1.1rem 4rem; }}
        .st-key-calculation_workspace > [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap; }}
        .st-key-calculation_workspace > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{ width:100%; flex:1 1 100%; }}
    }}
    </style>""")


def chip(text, kind=""):
    """A small status label; kind is ok, warn, bad, accent or empty."""
    import html

    return f'<span class="dfa-chip {kind}">{html.escape(str(text))}</span>'


def tag(text):
    import html

    return f'<span class="dfa-tag">{html.escape(str(text))}</span>'
