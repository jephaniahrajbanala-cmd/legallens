"""Streamlit interface for LegalLens — Indian Legal Document Intelligence."""

from __future__ import annotations

import html
import json

import streamlit as st

from legal_lens_runtime import (
    DEMO_DOCUMENTS,
    analyze_document,
    answer_question,
    clean_text,
    extract_claims,
    extract_outcome_rules,
    extract_timeline,
    legal_knowledge_status,
    read_pdf_bytes,
    runtime_capabilities,
    search_judgment,
    word_tokens,
)


st.set_page_config(
    page_title="LegalLens",
    page_icon="⚖️",
    layout="wide",
)

THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@400;700&family=JetBrains+Mono:wght@300;400;500;700&display=swap');

:root {
  --ink: #0a1119;
  --panel: #0e1721;
  --panel-2: #121d29;
  --line: #22303e;
  --line-soft: #18232f;
  --text: #dde6ef;
  --muted: #9aabbd;
  --dim: #74869a;
  --white: #f3f7fb;
  --pixel: 'Silkscreen', 'Courier New', monospace;
  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}

.stApp {
  background-color: var(--ink);
  background-image:
    radial-gradient(900px 500px at 50% -8%, rgba(96,124,153,.12), transparent 65%),
    radial-gradient(circle at 1px 1px, rgba(154,171,189,.07) 1px, transparent 0);
  background-size: auto, 26px 26px;
  color: var(--text);
  font-family: var(--mono);
}

[data-testid="stAppViewContainer"] { background: transparent; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stDecoration"],
#MainMenu,
footer,
[data-testid="stToolbar"] { display: none !important; }

.block-container { padding: 2.4rem 1.6rem 4rem; max-width: 1160px; }

::selection { background: var(--white); color: var(--ink); }

.stApp p, .stApp li, .stApp label, .stApp td, .stApp th { font-family: var(--mono); }
.stApp p { color: var(--text); line-height: 1.7; }

h1, h2, h3, h4, h5, h6 {
  font-family: var(--pixel) !important;
  font-weight: 400 !important;
  letter-spacing: .04em;
  color: var(--white) !important;
}
h1 { font-size: 1.55rem; }
h2 { font-size: 1.15rem; }
h3 { font-size: .95rem; }
h4 { font-size: .8rem; }
h5 { font-size: .7rem; }

a { color: var(--white); text-decoration: underline; text-underline-offset: 3px; }

[data-testid="stCaptionContainer"] p, .stCaption p {
  font-family: var(--mono) !important;
  font-size: .7rem !important;
  color: var(--dim) !important;
  letter-spacing: .04em;
}

code {
  font-family: var(--mono) !important;
  background: var(--panel-2) !important;
  color: var(--text) !important;
  border: 1px solid var(--line);
  border-radius: 4px;
}

.ll-appbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 2rem;
  border-bottom: 1px solid var(--line);
  padding-bottom: 1rem;
  margin-bottom: 1.5rem;
}
.ll-brand {
  font-family: var(--pixel);
  font-size: 1.6rem;
  line-height: 1;
  letter-spacing: .04em;
  color: var(--white);
}
.ll-brand span { color: var(--dim); margin-left: .15rem; }
.ll-brand-sub {
  font-family: var(--pixel);
  font-size: .48rem;
  line-height: 1.6;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--dim);
  margin-top: .5rem;
}
.ll-appbar-right {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: .4rem;
  flex-wrap: wrap;
}
.ll-chip {
  font-family: var(--pixel);
  font-size: .5rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: .34rem .65rem;
  background: rgba(255,255,255,.02);
  white-space: nowrap;
}
.ll-mini-dot {
  display: inline-block;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--white);
  margin-right: .4rem;
  vertical-align: middle;
}

[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .ll-card-mark) {
  background:
    linear-gradient(180deg, rgba(255,255,255,.025), transparent 140px),
    var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  padding: 1.15rem 1.25rem !important;
}
.ll-card-mark { display: none; }

.ll-card-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  border-bottom: 1px solid var(--line);
  padding-bottom: .7rem;
  margin-bottom: .9rem;
}
.ll-card-head .ll-label { margin: 0; color: var(--muted); }
.ll-card-head .ll-meta { margin: 0; }

.ll-nav-sep {
  border-top: 1px solid var(--line);
  margin: 1.1rem 0 .85rem;
}

.ll-view-head {
  border-bottom: 1px solid var(--line);
  padding-bottom: .85rem;
  margin-bottom: 1.15rem;
}
.ll-view-title {
  font-family: var(--pixel);
  font-size: .92rem;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--white);
}
.ll-view-sub {
  font-family: var(--mono);
  font-size: .72rem;
  line-height: 1.6;
  color: var(--dim);
  margin-top: .35rem;
}

.ll-stat {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: .5rem;
  padding: .4rem 0;
  border-bottom: 1px dashed var(--line-soft);
}
.ll-stat span {
  font-family: var(--pixel);
  font-size: .48rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--dim);
}
.ll-stat b {
  font-family: var(--mono);
  font-size: .76rem;
  font-weight: 400;
  color: var(--text);
}
.st-key-export_json button {
  width: 100%;
  white-space: nowrap;
}
.st-key-export_json button p {
  white-space: nowrap;
  font-size: .5rem !important;
  letter-spacing: .08em !important;
}

/* Stack the rail above the workspace on small screens. */
@media (max-width: 900px) {
  [data-testid="stHorizontalBlock"]:has(.st-key-workspace_nav) {
    flex-wrap: wrap;
  }
  [data-testid="stHorizontalBlock"]:has(.st-key-workspace_nav) > [data-testid="stColumn"] {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }
  .st-key-workspace_nav [role="radiogroup"] {
    flex-direction: row;
    flex-wrap: wrap;
  }
  .st-key-workspace_nav [data-testid="stRadioOption"] {
    width: auto !important;
  }
  [data-testid="stColumn"]:has(.st-key-workspace_nav) .ll-nav-sep {
    display: none;
  }
}

.ll-label {
  font-family: var(--pixel);
  font-size: .55rem;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--dim);
  margin: .1rem 0 .55rem;
}

.ll-answer {
  font-family: var(--mono);
  font-size: 1.1rem;
  line-height: 1.6;
  color: var(--white);
  border-left: 2px solid var(--white);
  padding: .15rem 0 .15rem 1rem;
  margin: .4rem 0 1.1rem;
}
.ll-evidence {
  font-family: var(--mono);
  font-size: .86rem;
  line-height: 1.75;
  color: var(--muted);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: .9rem 1.1rem;
  margin-top: .4rem;
}
.ll-outcome {
  font-family: var(--pixel);
  font-size: 1.1rem;
  line-height: 1.5;
  color: var(--white);
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--panel);
  padding: 1rem 1.2rem;
  margin-bottom: .6rem;
}
.ll-meta {
  font-family: var(--mono);
  font-size: .7rem;
  letter-spacing: .06em;
  color: var(--dim);
  text-transform: uppercase;
  margin: .2rem 0 1rem;
}
.ll-empty {
  font-family: var(--mono);
  font-size: .82rem;
  color: var(--dim);
  border: 1px dashed var(--line);
  border-radius: 12px;
  padding: 1rem 1.1rem;
}

.ll-result {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--panel);
  padding: .85rem 1.05rem;
  margin-bottom: .6rem;
}
.ll-result-head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  font-family: var(--pixel);
  font-size: .52rem;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--dim);
  margin-bottom: .45rem;
}
.ll-result-body {
  font-family: var(--mono);
  font-size: .84rem;
  line-height: 1.7;
  color: var(--text);
}

[data-testid="stSidebar"] {
  background: var(--panel);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebarContent"] { padding: 1.6rem 1.15rem 2rem; }
[data-testid="stSidebar"] .block-container { padding: 1.6rem 1.15rem 2rem; }
.ll-side-mark {
  font-family: var(--pixel);
  font-size: 1rem;
  letter-spacing: .05em;
  color: var(--white);
  margin-bottom: 1.4rem;
}
.ll-side-mark span { color: var(--dim); }

.ll-status {
  display: flex;
  align-items: baseline;
  gap: .5rem;
  padding: .45rem 0;
  border-bottom: 1px dashed var(--line-soft);
}
.ll-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  border: 1px solid var(--white);
  background: var(--white);
  flex: none;
  align-self: center;
}
.ll-dot.off { background: transparent; border-color: var(--dim); }
.ll-status-name {
  font-family: var(--pixel);
  font-size: .55rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--text);
}
.ll-status-detail {
  color: var(--dim);
  font-size: .66rem;
  margin-left: auto;
  text-align: right;
}

[data-testid="stWidgetLabel"] p {
  font-family: var(--pixel) !important;
  font-size: .52rem !important;
  letter-spacing: .14em !important;
  text-transform: uppercase;
  color: var(--muted) !important;
}

[data-testid="stTextInputRootElement"],
[data-testid="stTextAreaRootElement"],
[data-baseweb="input"],
[data-baseweb="textarea"] {
  background: var(--panel-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
}
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stTextAreaRootElement"]:focus-within,
[data-baseweb="input"]:focus-within,
[data-baseweb="textarea"]:focus-within {
  border-color: var(--muted) !important;
  box-shadow: 0 0 0 1px var(--muted) !important;
}
.stTextInput input, .stTextArea textarea, .stNumberInput input {
  background: transparent !important;
  color: var(--text) !important;
  border: none !important;
  box-shadow: none !important;
  font-family: var(--mono) !important;
  font-size: .84rem !important;
  line-height: 1.7 !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder { color: #526275 !important; }

[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: var(--panel-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: .84rem !important;
}
[data-baseweb="popover"] [role="listbox"] {
  background: var(--panel-2) !important;
  border: 1px solid var(--line) !important;
}
[data-baseweb="popover"] li {
  background: var(--panel-2) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
}
[data-baseweb="popover"] li:hover { background: var(--line-soft) !important; }

.stButton > button, .stDownloadButton > button {
  font-family: var(--pixel) !important;
  font-size: .58rem !important;
  letter-spacing: .14em !important;
  text-transform: uppercase;
  color: var(--text) !important;
  background: transparent !important;
  border: 1px solid var(--line) !important;
  border-radius: 999px !important;
  padding: .62rem 1.15rem !important;
  min-height: 0 !important;
  transition: all .15s ease;
}
.stButton > button p, .stDownloadButton > button p {
  font-family: var(--pixel) !important;
  font-size: .58rem !important;
  letter-spacing: .14em !important;
  color: inherit !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  color: var(--ink) !important;
  background: var(--white) !important;
  border-color: var(--white) !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
  color: var(--ink) !important;
  background: var(--white) !important;
  border-color: var(--white) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
  color: var(--text) !important;
  background: transparent !important;
  border-color: var(--white) !important;
}

.stTabs [role="tablist"] {
  gap: .45rem;
  background: transparent;
  border-bottom: none;
  flex-wrap: wrap;
  padding: 0;
}
.stTabs [role="tablist"]::after { display: none !important; }
.stTabs [data-testid="stTab"] {
  font-family: var(--pixel) !important;
  font-size: .56rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--muted);
  background: transparent;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: .45rem .9rem;
  height: auto;
  min-height: 0;
  cursor: pointer;
  transition: all .15s ease;
}
.stTabs [data-testid="stTab"] p {
  font-family: var(--pixel) !important;
  font-size: .56rem !important;
  letter-spacing: .1em !important;
  color: inherit !important;
}
.stTabs [data-testid="stTab"]:hover { color: var(--text); border-color: var(--muted); }
.stTabs [data-testid="stTab"][data-selected="true"] {
  background: var(--white);
  border-color: var(--white);
  color: var(--ink);
}
.stTabs .react-aria-SelectionIndicator { display: none !important; }
.stTabs [data-testid="stTabPanel"] { padding-top: 1.5rem; }

.stRadio [role="radiogroup"] { gap: .45rem; flex-wrap: wrap; }
.stRadio [role="radiogroup"] label {
  background: transparent;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: .34rem .8rem;
  margin: 0;
  transition: all .15s ease;
}
.stRadio [role="radiogroup"] label:hover { border-color: var(--muted); }
.stRadio [role="radiogroup"] label:has(input:checked) {
  background: var(--white);
  border-color: var(--white);
}
.stRadio [role="radiogroup"] label:has(input:checked) p { color: var(--ink) !important; }
.stRadio [role="radiogroup"] label > div:first-child { display: none; }
.stRadio [data-testid="stRadioOption"] > div > div > div:first-child { display: none !important; }
.stRadio [role="radiogroup"] label p {
  font-family: var(--pixel) !important;
  font-size: .52rem !important;
  letter-spacing: .1em !important;
  text-transform: uppercase;
  color: var(--muted) !important;
}

.st-key-workspace_nav [role="radiogroup"] {
  flex-direction: column;
  align-items: stretch;
  gap: .18rem;
  flex-wrap: nowrap;
  counter-reset: nav;
}
.st-key-workspace_nav [data-testid="stRadioOption"] {
  width: 100%;
  display: flex;
  align-items: center;
  border: 1px solid transparent !important;
  border-radius: 8px !important;
  padding: .5rem .6rem !important;
  margin: 0 !important;
  background: transparent !important;
}
.st-key-workspace_nav [data-testid="stRadioOption"]::before {
  counter-increment: nav;
  content: "0" counter(nav);
  font-family: var(--pixel);
  font-size: .48rem;
  letter-spacing: .06em;
  color: var(--dim);
  margin-right: .55rem;
}
.st-key-workspace_nav [data-testid="stRadioOption"]:hover {
  background: rgba(255,255,255,.03) !important;
  border-color: transparent !important;
}
.st-key-workspace_nav [data-testid="stRadioOption"][data-selected="true"] {
  background: var(--white) !important;
  border-color: var(--white) !important;
}
.st-key-workspace_nav [data-testid="stRadioOption"][data-selected="true"]::before { color: var(--ink); }
.st-key-workspace_nav [role="radiogroup"] label p {
  font-family: var(--pixel) !important;
  font-size: .54rem !important;
  letter-spacing: .08em !important;
  text-transform: uppercase;
  color: var(--muted) !important;
  white-space: nowrap;
}
.st-key-workspace_nav [data-testid="stRadioOption"][data-selected="true"] p {
  color: var(--ink) !important;
}

[data-testid="stAlert"] {
  background: transparent !important;
  border: none !important;
}
[data-testid="stAlertContainer"] {
  background: var(--panel-2) !important;
  border: 1px solid var(--line) !important;
  border-left: 2px solid var(--white) !important;
  border-radius: 10px !important;
}
[data-testid="stAlertContainer"] p,
[data-testid="stAlertContainer"] div,
[data-testid="stAlertContainer"] span { color: var(--text) !important; }
[data-testid="stAlertContainer"] svg { color: var(--muted) !important; }

[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: .9rem 1rem;
}
[data-testid="stMetricLabel"] p {
  font-family: var(--pixel) !important;
  font-size: .5rem !important;
  letter-spacing: .12em !important;
  text-transform: uppercase;
  color: var(--dim) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--pixel);
  color: var(--white);
  font-size: 1.2rem;
}
[data-testid="stMetricDelta"] { font-family: var(--mono); font-size: .66rem; }

[data-testid="stExpander"] {
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  background: var(--panel);
}
[data-testid="stExpander"] summary {
  font-family: var(--pixel) !important;
  font-size: .55rem !important;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted) !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text) !important; }

[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
}

[data-testid="stFileUploaderDropzone"] {
  background: var(--panel-2) !important;
  border: 1px dashed var(--line) !important;
  border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
  color: var(--muted) !important;
  font-family: var(--mono) !important;
}
[data-testid="stFileUploaderDropzone"] button {
  font-family: var(--pixel) !important;
  font-size: .55rem !important;
  letter-spacing: .1em;
  border: 1px solid var(--line) !important;
  border-radius: 999px !important;
  background: transparent !important;
  color: var(--text) !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
  background: var(--white) !important;
  color: var(--ink) !important;
}
[data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"] {
  display: none !important;
}

[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons {
  font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
  letter-spacing: normal !important;
  text-transform: none !important;
}

hr { border: none; border-top: 1px solid var(--line); }
[data-testid="stDivider"] { background: var(--line); }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--ink); }
::-webkit-scrollbar-thumb {
  background: #22303e;
  border-radius: 999px;
  border: 2px solid var(--ink);
}
::-webkit-scrollbar-thumb:hover { background: #314355; }

@media (max-width: 640px) {
  .ll-appbar { flex-direction: column; align-items: flex-start; gap: .7rem; }
  .ll-appbar-right { justify-content: flex-start; }
  .ll-brand { font-size: 1.4rem; }
  .block-container { padding: 1.6rem 1rem 3rem; }
}
</style>
"""

st.markdown(THEME, unsafe_allow_html=True)


def _label(text: str) -> None:
    st.markdown(f'<div class="ll-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def _empty(text: str) -> None:
    st.markdown(f'<div class="ll-empty">{html.escape(text)}</div>', unsafe_allow_html=True)


def _status(name: str, ready: bool, detail: str = "") -> None:
    dot = "ll-dot" if ready else "ll-dot off"
    st.markdown(
        f'<div class="ll-status"><span class="{dot}"></span>'
        f'<span class="ll-status-name">{html.escape(name)}</span>'
        f'<span class="ll-status-detail">{html.escape(detail)}</span></div>',
        unsafe_allow_html=True,
    )


def _render_timeline(events: list[dict]) -> None:
    if not events:
        _empty("No recognisable dated events found.")
        return
    st.dataframe(
        [{"Date": e["date"], "Type": e["event_type"], "Event": e["event"]} for e in events],
        use_container_width=True,
        hide_index=True,
    )


def _render_claims(claims: dict[str, list[dict]]) -> None:
    if not claims:
        _empty("No petitioner, respondent, or court positions were detected.")
        return
    for role, rows in claims.items():
        _label(f"{role} // {len(rows)}")
        for row in rows[:20]:
            st.markdown(f"- {row['claim']}")


WORKSPACE_VIEWS = {
    "Summary": "Condense the judgment into the passages that matter.",
    "QA + Evidence": "Ask about the case and see the sentence each answer came from.",
    "Decision": "Read the court's order and the line that supports it.",
    "Timeline": "List the dated events in the order they happened.",
    "Claims": "Group the positions taken by each side and by the court.",
    "Search": "Find the sentences closest to a phrase.",
}


def main() -> None:
    st.markdown(
        """
<div class="ll-appbar">
  <div>
    <div class="ll-brand">LEGALLENS<span>//</span></div>
    <div class="ll-brand-sub">Indian legal document intelligence</div>
  </div>
  <div class="ll-appbar-right">
    <span class="ll-chip"><span class="ll-mini-dot"></span>local engine</span>
    <span class="ll-chip">workspace</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            '<div class="ll-side-mark">LEGALLENS<span>//</span></div>',
            unsafe_allow_html=True,
        )
        _label("document")
        demo_name = st.selectbox(
            "Load a sample document",
            ["None"] + list(DEMO_DOCUMENTS),
            help="Samples make it possible to try the app immediately.",
        )
        uploaded = st.file_uploader("Upload a judgment PDF", type=["pdf"])

        _label("settings")
        artifact_dir = st.text_input(
            "Notebook artifact folder",
            value="legallens_artifacts",
            help="Point this at the folder containing the notebook's saved abstractive/ checkpoint.",
        )

        _label("runtime")
        try:
            capabilities = runtime_capabilities(artifact_dir)
        except TypeError:
            # Compatibility with a runtime module loaded before artifact_dir
            # support was added.
            capabilities = runtime_capabilities()
        # Keep the UI compatible with a stale/reloaded runtime module from
        # before abstractive checkpoint support was added.
        abstractive_capability = capabilities.get(
            "abstractive",
            {
                "available": False,
                "message": "Abstractive checkpoint status is unavailable; extractive fallback is active.",
            },
        )
        _status(
            "model artifacts",
            capabilities["available"],
            "detected" if capabilities["available"] else "lightweight mode",
        )
        _status(
            "abstractive",
            abstractive_capability.get("available", False),
            "checkpoint ready" if abstractive_capability.get("available", False) else "extractive fallback",
        )
        knowledge_capability = legal_knowledge_status(artifact_dir)
        _status(
            "statute index",
            knowledge_capability["available"],
            "indexed" if knowledge_capability["available"] else "built-in refs",
        )
        if not abstractive_capability.get("available", False):
            st.caption(
                abstractive_capability.get(
                    "message",
                    "Abstractive mode will use an extractive fallback until the notebook checkpoint exists.",
                )
            )

    sample_text = DEMO_DOCUMENTS.get(demo_name, "")
    pdf_text = ""
    pdf_error = ""
    if uploaded is not None:
        try:
            pdf_text = read_pdf_bytes(uploaded.getvalue())
        except Exception as exc:
            pdf_error = f"Could not read the PDF: {exc}"

    with st.container(border=True):
        st.markdown('<div class="ll-card-mark"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ll-card-head"><div class="ll-label">source document</div>'
            '<div class="ll-meta">paste, upload, or pick a sample</div></div>',
            unsafe_allow_html=True,
        )
        pasted = st.text_area(
            "Judgment text",
            value=sample_text,
            height=190,
            placeholder="Paste the judgment text here, or select a sample / upload a PDF.",
            label_visibility="collapsed",
        )
        if pdf_error:
            st.error(pdf_error)
        elif uploaded is not None and pdf_text:
            st.caption(f"Extracted {len(word_tokens(pdf_text)):,} words from {uploaded.name}.")

        text = clean_text(pasted or pdf_text or sample_text)
        if text:
            st.markdown(
                f'<div class="ll-meta">{len(word_tokens(text)):,} words &nbsp;&middot;&nbsp; {len(text):,} characters</div>',
                unsafe_allow_html=True,
            )
        else:
            _empty("Provide a PDF, paste text, or select a sample document to begin.")

    if not text:
        return

    analysis = analyze_document(text)

    with st.container(border=True):
        st.markdown('<div class="ll-card-mark"></div>', unsafe_allow_html=True)
        nav_col, body_col = st.columns([1, 3.5], gap="large")

        with nav_col:
            _label("workspace")
            view = st.radio(
                "Workspace view",
                list(WORKSPACE_VIEWS),
                key="workspace_nav",
                label_visibility="collapsed",
            )
            st.markdown('<div class="ll-nav-sep"></div>', unsafe_allow_html=True)
            _label("document")
            st.markdown(
                f'<div class="ll-stat"><span>words</span><b>{analysis["word_count"]:,}</b></div>'
                f'<div class="ll-stat"><span>sentences</span><b>{analysis["sentence_count"]:,}</b></div>'
                f'<div class="ll-stat"><span>events</span><b>{len(analysis["timeline"]):,}</b></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="ll-nav-sep"></div>', unsafe_allow_html=True)
            st.download_button(
                "Export JSON",
                data=json.dumps(analysis, indent=2, default=str),
                file_name="legallens_analysis.json",
                mime="application/json",
                key="export_json",
            )

        with body_col:
            st.markdown(
                f'<div class="ll-view-head"><div class="ll-view-title">{html.escape(view)}</div>'
                f'<div class="ll-view-sub">{html.escape(WORKSPACE_VIEWS[view])}</div></div>',
                unsafe_allow_html=True,
            )

            if view == "Summary":
                method = st.radio(
                    "Summary method",
                    ["Extractive (local)", "Abstractive (fine-tuned DistilBART)"],
                    horizontal=True,
                    key="summary_method",
                )
                if method.startswith("Abstractive"):
                    st.caption("The first run loads a large checkpoint, so it can take a minute.")

                if st.button("Generate summary", type="primary", key="summary_button"):
                    wants_abstractive = method.startswith("Abstractive")
                    abstractive_ready = abstractive_capability.get("available", False)
                    if wants_abstractive and abstractive_ready:
                        with st.spinner("Loading the checkpoint and writing the summary..."):
                            try:
                                summary_analysis = analyze_document(
                                    text, "abstractive", artifact_dir=artifact_dir
                                )
                            except TypeError:
                                # A stale imported runtime may not accept artifact_dir.
                                summary_analysis = analyze_document(text, "abstractive")
                    else:
                        summary_analysis = analyze_document(text, "extractive")
                        if wants_abstractive:
                            summary_analysis["summary_note"] = abstractive_capability.get(
                                "message",
                                "Abstractive checkpoint unavailable; extractive summary selected.",
                            )
                            summary_analysis["summary_source"] = "extractive fallback"
                    if summary_analysis["summary_note"]:
                        if summary_analysis["summary_source"] == "notebook abstractive checkpoint":
                            st.success(summary_analysis["summary_note"])
                        else:
                            st.warning(summary_analysis["summary_note"])
                    st.caption(f"Summary source: {summary_analysis['summary_source']}")
                    _label("summary")
                    st.write(summary_analysis["summary"] or "No summary could be generated.")

            elif view == "QA + Evidence":
                question = st.text_input(
                    "Question",
                    value="What was the outcome of the case?",
                    placeholder="e.g. Who is the petitioner? Which section is referred to? What is Section 420?",
                    key="qa_question",
                )
                if st.button("Answer", type="primary", key="qa_button"):
                    result = answer_question(question, text, artifact_dir=artifact_dir)
                    _label("answer")
                    st.markdown(
                        f'<div class="ll-answer">{html.escape(result["answer"] or "No confident answer found")}</div>',
                        unsafe_allow_html=True,
                    )
                    if result["evidence"]:
                        _label("evidence")
                        st.markdown(
                            f'<div class="ll-evidence">{html.escape(result["evidence"])}</div>',
                            unsafe_allow_html=True,
                        )

            elif view == "Decision":
                if st.button("Extract outcome", type="primary", key="decision_button"):
                    outcome = extract_outcome_rules(text)
                    _label("outcome")
                    st.markdown(
                        f'<div class="ll-outcome">{html.escape(outcome["outcome"])}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"confidence {outcome['confidence']:.2f}")
                    if outcome["evidence"]:
                        _label("evidence")
                        st.markdown(
                            f'<div class="ll-evidence">{html.escape(outcome["evidence"])}</div>',
                            unsafe_allow_html=True,
                        )

            elif view == "Timeline":
                if st.button("Extract timeline", type="primary", key="timeline_button"):
                    _render_timeline(extract_timeline(text))

            elif view == "Claims":
                if st.button("Extract positions", type="primary", key="claims_button"):
                    _render_claims(extract_claims(text))

            else:
                query = st.text_input(
                    "Search query",
                    placeholder="e.g. alternative remedy",
                    key="search_query",
                )
                if st.button("Search", type="primary", key="search_button"):
                    results = search_judgment(query, text, limit=3)
                    if not results:
                        _empty("No matching sentences.")
                    for index, result in enumerate(results, 1):
                        st.markdown(
                            f'<div class="ll-result"><div class="ll-result-head">'
                            f'<span>result {index:02d}</span><span>score {result["score"]:.3f}</span></div>'
                            f'<div class="ll-result-body">{html.escape(result["sentence"])}</div></div>',
                            unsafe_allow_html=True,
                        )


if __name__ == "__main__":
    main()
