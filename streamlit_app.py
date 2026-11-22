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
  --dim: #7d8fa3;
  --white: #f3f7fb;
  --pixel: 'Silkscreen', 'Courier New', monospace;
  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}

.stApp {
  background-color: var(--ink);
  background-image:
    radial-gradient(1000px 560px at 50% -10%, rgba(96,124,153,.14), transparent 65%),
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
[data-testid="stToolbar"],
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

.block-container { padding: 1.6rem 2.5rem 5rem; max-width: 1300px; }

::selection { background: var(--white); color: var(--ink); }

.stApp p, .stApp li, .stApp label, .stApp td, .stApp th { font-family: var(--mono); }
.stApp p { color: var(--text); line-height: 1.8; font-size: 1rem; }

h1, h2, h3, h4, h5, h6 {
  font-family: var(--pixel) !important;
  font-weight: 400 !important;
  letter-spacing: .02em;
  color: var(--white) !important;
}
h1 { font-size: 2.2rem; }
h2 { font-size: 1.6rem; }
h3 { font-size: 1.25rem; }
h4 { font-size: 1rem; }
h5 { font-size: .85rem; }

a { color: var(--white); text-decoration: underline; text-underline-offset: 3px; }

[data-testid="stCaptionContainer"] p, .stCaption p {
  font-family: var(--mono) !important;
  font-size: .82rem !important;
  color: var(--dim) !important;
  line-height: 1.7 !important;
  letter-spacing: .01em;
}

code {
  font-family: var(--mono) !important;
  background: var(--panel-2) !important;
  color: var(--text) !important;
  border: 1px solid var(--line);
  border-radius: 4px;
}

/* ── top bar ───────────────────────────────────────────── */

.ll-appbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 2rem;
  border-bottom: 1px solid var(--line);
  padding: .4rem 0 1.15rem;
}
.ll-brand {
  font-family: var(--pixel);
  font-size: 1.55rem;
  line-height: 1;
  letter-spacing: .04em;
  color: var(--white);
}
.ll-brand span { color: var(--dim); margin-left: .12rem; }
.ll-brand-sub {
  font-family: var(--pixel);
  font-size: .5rem;
  line-height: 1.6;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--dim);
  margin-top: .55rem;
}
.ll-appbar-right {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: .45rem;
  flex-wrap: wrap;
}
.ll-chip {
  font-family: var(--pixel);
  font-size: .6rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: .42rem .8rem;
  background: rgba(255,255,255,.02);
  white-space: nowrap;
}
.ll-mini-dot {
  display: inline-block;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--white);
  margin-right: .42rem;
  vertical-align: middle;
}
.ll-mini-dot.off {
  background: transparent;
  border: 1px solid var(--dim);
}

/* ── hero ──────────────────────────────────────────────── */

.ll-hero { padding: 4rem 0 3.2rem; }
.ll-hero-title {
  font-family: var(--pixel);
  font-size: clamp(2.6rem, 5.4vw, 4.4rem);
  line-height: 1.06;
  letter-spacing: .01em;
  color: var(--white);
  text-wrap: balance;
  max-width: 15ch;
}
.ll-hero-title span { color: var(--dim); }
.ll-hero-sub {
  font-family: var(--mono);
  font-size: 1.05rem;
  line-height: 1.85;
  color: var(--muted);
  max-width: 60ch;
  margin: 1.5rem 0 0;
}
.ll-hero-note {
  display: flex;
  gap: .5rem;
  flex-wrap: wrap;
  margin-top: 1.8rem;
}

/* ── source panel ──────────────────────────────────────── */

[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .ll-card-mark) {
  background:
    linear-gradient(180deg, rgba(255,255,255,.03), transparent 180px),
    var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  padding: 1.5rem 1.6rem !important;
}
.ll-card-mark { display: none; }

.ll-card-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  border-bottom: 1px solid var(--line);
  padding-bottom: .85rem;
  margin-bottom: 1.15rem;
}
.ll-card-head .ll-label { margin: 0; color: var(--muted); }
.ll-meta {
  font-family: var(--mono);
  font-size: .82rem;
  letter-spacing: .02em;
  color: var(--dim);
  margin: 0;
}

/* document counters under the textarea */
.ll-docmeta {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: .4rem 1.8rem;
  margin: .9rem 0 .1rem;
}
.ll-docmeta span {
  font-family: var(--pixel);
  font-size: .6rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--dim);
  white-space: nowrap;
}
.ll-docmeta b {
  font-family: var(--mono);
  font-size: 1.15rem;
  font-weight: 400;
  letter-spacing: 0;
  color: var(--text);
  margin-right: .4rem;
}

/* ── workspace ─────────────────────────────────────────── */

.ll-workspace-rule {
  border-top: 1px solid var(--line);
  margin: 3.2rem 0 2.1rem;
}
.ll-view-head {
  border-bottom: 1px solid var(--line);
  padding-bottom: 1rem;
  margin-bottom: 1.4rem;
}
.ll-view-title {
  font-family: var(--pixel);
  font-size: 1.8rem;
  line-height: 1.25;
  letter-spacing: .03em;
  text-transform: uppercase;
  color: var(--white);
}
.ll-view-sub {
  font-family: var(--mono);
  font-size: 1.02rem;
  line-height: 1.7;
  color: var(--muted);
  margin-top: .55rem;
  max-width: 64ch;
}

.ll-label {
  font-family: var(--pixel);
  font-size: .62rem;
  letter-spacing: .15em;
  text-transform: uppercase;
  color: var(--dim);
  margin: .1rem 0 .6rem;
}
.ll-nav-sep {
  border-top: 1px solid var(--line);
  margin: 1.2rem 0 1rem;
}

/* ── outputs ───────────────────────────────────────────── */

.ll-summary {
  font-family: var(--mono);
  font-size: 1.12rem;
  line-height: 1.95;
  color: var(--text);
  max-width: 74ch;
  white-space: pre-wrap;
  margin-top: .3rem;
}
.ll-answer {
  font-family: var(--mono);
  font-size: 1.5rem;
  line-height: 1.55;
  color: var(--white);
  border-left: 1px solid var(--white);
  padding: .2rem 0 .2rem 1.15rem;
  margin: .5rem 0 1.3rem;
  max-width: 60ch;
}
.ll-evidence {
  font-family: var(--mono);
  font-size: 1rem;
  line-height: 1.8;
  color: var(--muted);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 1.05rem 1.25rem;
  margin-top: .45rem;
}
.ll-outcome {
  font-family: var(--mono);
  font-size: 1.3rem;
  line-height: 1.6;
  color: var(--white);
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--panel);
  padding: 1.1rem 1.3rem;
  margin-bottom: .7rem;
}
.ll-empty {
  font-family: var(--mono);
  font-size: .95rem;
  line-height: 1.7;
  color: var(--dim);
  border: 1px dashed var(--line);
  border-radius: 12px;
  padding: 1.05rem 1.2rem;
}

.ll-result {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--panel);
  padding: 1.05rem 1.25rem;
  margin-bottom: .75rem;
}
.ll-result-head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  font-family: var(--pixel);
  font-size: .56rem;
  letter-spacing: .13em;
  text-transform: uppercase;
  color: var(--dim);
  margin-bottom: .55rem;
}
.ll-result-body {
  font-family: var(--mono);
  font-size: 1rem;
  line-height: 1.8;
  color: var(--text);
}

/* ── widgets ───────────────────────────────────────────── */

[data-testid="stWidgetLabel"] p {
  font-family: var(--pixel) !important;
  font-size: .62rem !important;
  letter-spacing: .12em !important;
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
  font-size: 1rem !important;
  line-height: 1.8 !important;
  caret-color: var(--white);
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder { color: var(--dim) !important; }

[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: var(--panel-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: .95rem !important;
  min-height: 44px;
}
[data-baseweb="popover"] [role="listbox"] {
  background: var(--panel-2) !important;
  border: 1px solid var(--line) !important;
}
[data-baseweb="popover"] li {
  background: var(--panel-2) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: .95rem !important;
}
[data-baseweb="popover"] li:hover { background: var(--line-soft) !important; }

.stButton > button, .stDownloadButton > button {
  font-family: var(--pixel) !important;
  font-size: .66rem !important;
  letter-spacing: .11em !important;
  text-transform: uppercase;
  color: var(--text) !important;
  background: transparent !important;
  border: 1px solid var(--line) !important;
  border-radius: 999px !important;
  padding: .76rem 1.4rem !important;
  min-height: 0 !important;
  transition: background .16s ease, color .16s ease, border-color .16s ease;
}
.stButton > button p, .stDownloadButton > button p {
  font-family: var(--pixel) !important;
  font-size: .66rem !important;
  letter-spacing: .11em !important;
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
.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible,
[data-testid="stPopover"] button:focus-visible {
  outline: 2px solid var(--white);
  outline-offset: 2px;
}

[data-testid="stPopover"] button {
  font-family: var(--pixel) !important;
  font-size: .58rem !important;
  letter-spacing: .1em !important;
  text-transform: uppercase;
  color: var(--text) !important;
  background: transparent !important;
  border: 1px solid var(--line) !important;
  border-radius: 999px !important;
  min-height: 44px !important;
  padding: .6rem .9rem !important;
  transition: background .16s ease, color .16s ease, border-color .16s ease;
}
[data-testid="stPopover"] button p {
  font-family: var(--pixel) !important;
  font-size: .6rem !important;
  letter-spacing: .1em !important;
  text-transform: uppercase;
  color: inherit !important;
  white-space: nowrap;
}
[data-testid="stPopover"] button:hover {
  background: var(--white) !important;
  color: var(--ink) !important;
  border-color: var(--white) !important;
}

.stRadio [role="radiogroup"] { gap: .45rem; flex-wrap: wrap; }
.stRadio [role="radiogroup"] label {
  background: transparent;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: .42rem .9rem;
  margin: 0;
  transition: background .16s ease, color .16s ease, border-color .16s ease;
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
  font-size: .62rem !important;
  letter-spacing: .07em !important;
  text-transform: uppercase;
  color: var(--muted) !important;
}
.stRadio [role="radiogroup"] label:has(input:focus-visible) {
  outline: 2px solid var(--white);
  outline-offset: 2px;
}

/* vertical tool rail */
.st-key-workspace_nav [role="radiogroup"] {
  flex-direction: column;
  align-items: stretch;
  gap: .22rem;
  flex-wrap: nowrap;
}
.st-key-workspace_nav [data-testid="stRadioOption"] {
  width: 100%;
  display: flex;
  align-items: center;
  border: 1px solid transparent !important;
  border-radius: 10px !important;
  padding: .72rem .85rem !important;
  margin: 0 !important;
  background: transparent !important;
}
.st-key-workspace_nav [data-testid="stRadioOption"]:hover {
  background: rgba(255,255,255,.035) !important;
  border-color: transparent !important;
}
.st-key-workspace_nav [data-testid="stRadioOption"][data-selected="true"] {
  background: var(--white) !important;
  border-color: var(--white) !important;
}
.st-key-workspace_nav [role="radiogroup"] label p {
  font-family: var(--pixel) !important;
  font-size: .72rem !important;
  letter-spacing: .05em !important;
  text-transform: uppercase;
  color: var(--muted) !important;
  white-space: nowrap;
}
.st-key-workspace_nav [data-testid="stRadioOption"][data-selected="true"] p {
  color: var(--ink) !important;
}
[data-testid="stColumn"]:has(.st-key-workspace_nav) {
  border-right: 1px solid var(--line);
  padding-right: 1.6rem;
}
.st-key-export_json button { width: 100%; }
.st-key-export_json button p {
  white-space: nowrap;
  font-size: .62rem !important;
}

/* ── status, alerts, data ──────────────────────────────── */

.ll-status {
  display: flex;
  align-items: baseline;
  gap: .5rem;
  padding: .5rem 0;
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
  font-size: .58rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--text);
}
.ll-status-detail {
  color: var(--dim);
  font-size: .78rem;
  margin-left: auto;
  text-align: right;
}

[data-testid="stAlert"] {
  background: transparent !important;
  border: none !important;
}
[data-testid="stAlertContainer"] {
  background: var(--panel-2) !important;
  border: 1px solid var(--line) !important;
  border-left: 1px solid var(--white) !important;
  border-radius: 10px !important;
  padding: .9rem 1.1rem !important;
}
[data-testid="stAlertContainer"] p,
[data-testid="stAlertContainer"] div,
[data-testid="stAlertContainer"] span { color: var(--text) !important; font-size: .95rem !important; }
[data-testid="stAlertContainer"] svg { color: var(--muted) !important; }

[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: .9rem 1rem;
}
[data-testid="stMetricLabel"] p {
  font-family: var(--pixel) !important;
  font-size: .55rem !important;
  letter-spacing: .12em !important;
  text-transform: uppercase;
  color: var(--dim) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--pixel);
  color: var(--white);
  font-size: 1.3rem;
}
[data-testid="stMetricDelta"] { font-family: var(--mono); font-size: .78rem; }

[data-testid="stExpander"] {
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  background: var(--panel);
}
[data-testid="stExpander"] summary {
  font-family: var(--pixel) !important;
  font-size: .6rem !important;
  letter-spacing: .1em;
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
  padding: .85rem 1rem !important;
  min-height: 0 !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
  color: var(--muted) !important;
  font-family: var(--mono) !important;
  font-size: .8rem !important;
}
[data-testid="stFileUploaderDropzone"] button {
  font-family: var(--pixel) !important;
  font-size: .56rem !important;
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

/* ── responsive ────────────────────────────────────────── */

@media (max-width: 900px) {
  .block-container { padding: 1.2rem 1.25rem 3.5rem; }
  .ll-appbar { flex-direction: column; align-items: flex-start; gap: .9rem; }
  .ll-appbar-right { justify-content: flex-start; }
  .ll-hero { padding: 2.6rem 0 2.2rem; }
  .ll-hero-title { font-size: 2.6rem; max-width: 100%; }
  .ll-hero-sub { font-size: .98rem; }
  [data-testid="stHorizontalBlock"]:has(.st-key-workspace_nav) { flex-wrap: wrap; }
  [data-testid="stHorizontalBlock"]:has(.st-key-workspace_nav) > [data-testid="stColumn"] {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }
  [data-testid="stColumn"]:has(.st-key-workspace_nav) {
    border-right: none;
    padding-right: 0;
    border-bottom: 1px solid var(--line);
    padding-bottom: 1.3rem;
  }
  .st-key-workspace_nav [role="radiogroup"] {
    flex-direction: row;
    flex-wrap: wrap;
  }
  .st-key-workspace_nav [data-testid="stRadioOption"] {
    width: auto !important;
  }
  .ll-workspace-rule { margin: 2.2rem 0 1.6rem; }
  .ll-view-title { font-size: 1.4rem; }
}

@media (max-width: 900px) {
  [data-testid="stHorizontalBlock"]:has(.st-key-sample_pick) {
    flex-wrap: wrap;
    gap: 1rem !important;
  }
  [data-testid="stHorizontalBlock"]:has(.st-key-sample_pick) > [data-testid="stColumn"] {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }
  .st-key-engine_popover button {
    width: auto !important;
    min-width: 150px;
  }
  [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .ll-card-mark) {
    padding: 1.15rem 1.1rem !important;
  }
  .ll-card-head { flex-direction: column; gap: .25rem; }
  .ll-hero-title { font-size: 2.25rem; }
}

@media (max-width: 480px) {
  .ll-hero-title { font-size: 1.9rem; }
  .ll-hero-sub { font-size: .95rem; }
  .ll-brand { font-size: 1.35rem; }
  .ll-docmeta { gap: .35rem 1.2rem; }
}
</style>
"""

st.markdown(THEME, unsafe_allow_html=True)


def _label(text: str) -> None:
    st.markdown(f'<div class="ll-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def _empty(text: str) -> None:
    st.markdown(f'<div class="ll-empty">{html.escape(text)}</div>', unsafe_allow_html=True)


def _chip(label: str, ready: bool) -> str:
    dot = "ll-mini-dot" if ready else "ll-mini-dot off"
    return f'<span class="ll-chip"><span class="{dot}"></span>{html.escape(label)}</span>'


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

HERO = """
<div class="ll-hero">
  <div class="ll-hero-title">READ THE ORDER.<br><span>FIND THE LINE.</span></div>
  <p class="ll-hero-sub">Turn an Indian judgment into a summary, a dated timeline, and the
  sentence behind every answer. Pick a sample or paste your own text below.</p>
  <div class="ll-hero-note">
    <span class="ll-chip">summary</span>
    <span class="ll-chip">timeline</span>
    <span class="ll-chip">claims</span>
    <span class="ll-chip">sentence evidence</span>
  </div>
</div>
"""


def main() -> None:
    artifact_dir = st.session_state.get("artifact_dir", "legallens_artifacts")
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
    knowledge_capability = legal_knowledge_status(artifact_dir)

    abstractive_ready = abstractive_capability.get("available", False)
    st.markdown(
        f"""
<div class="ll-appbar">
  <div>
    <div class="ll-brand">LEGALLENS<span>//</span></div>
    <div class="ll-brand-sub">Indian legal document intelligence</div>
  </div>
  <div class="ll-appbar-right">
    {_chip("local engine", True)}
    {_chip("abstractive ready" if abstractive_ready else "extractive only", abstractive_ready)}
    {_chip("statute index" if knowledge_capability["available"] else "built-in refs", knowledge_capability["available"])}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    hero_slot = st.empty()

    with st.container(border=True):
        st.markdown('<div class="ll-card-mark"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ll-card-head"><div class="ll-label">source document</div>'
            '<div class="ll-meta">paste &middot; upload &middot; sample</div></div>',
            unsafe_allow_html=True,
        )

        pick_col, upload_col, engine_col = st.columns(
            [1.35, 1.35, 0.65], gap="large", vertical_alignment="top"
        )
        with pick_col:
            demo_name = st.selectbox(
                "Start from a sample",
                ["None"] + list(DEMO_DOCUMENTS),
                key="sample_pick",
                help="Samples make it possible to try the app immediately.",
            )
        with upload_col:
            uploaded = st.file_uploader("Or upload a judgment PDF", type=["pdf"])
        with engine_col:
            _label("engine")
            with st.popover("Settings", use_container_width=True, key="engine_popover"):
                st.text_input(
                    "Checkpoint folder",
                    value="legallens_artifacts",
                    key="artifact_dir",
                    help="Folder containing the notebook's saved abstractive/ checkpoint.",
                )
                _status(
                    "model artifacts",
                    capabilities["available"],
                    "detected" if capabilities["available"] else "lightweight mode",
                )
                _status(
                    "abstractive",
                    abstractive_ready,
                    "checkpoint ready" if abstractive_ready else "extractive fallback",
                )
                _status(
                    "statute index",
                    knowledge_capability["available"],
                    "indexed" if knowledge_capability["available"] else "built-in refs",
                )
                if not abstractive_ready:
                    st.caption(
                        abstractive_capability.get(
                            "message",
                            "Abstractive mode will use an extractive fallback until the notebook checkpoint exists.",
                        )
                    )

        sample_text = DEMO_DOCUMENTS.get(demo_name, "")
        pasted = st.text_area(
            "Judgment text",
            value=sample_text,
            height=170,
            placeholder="Paste the judgment text here, or select a sample / upload a PDF.",
            label_visibility="collapsed",
        )

        pdf_text = ""
        pdf_error = ""
        if uploaded is not None:
            try:
                pdf_text = read_pdf_bytes(uploaded.getvalue())
            except Exception as exc:
                pdf_error = f"Could not read the PDF: {exc}"

        if pdf_error:
            st.error(pdf_error)
        elif uploaded is not None and pdf_text:
            st.caption(f"Extracted {len(word_tokens(pdf_text)):,} words from {uploaded.name}.")

        text = clean_text(pasted or pdf_text or sample_text)

        if text:
            analysis = analyze_document(text)
            st.markdown(
                f'<div class="ll-docmeta">'
                f'<span><b>{analysis["word_count"]:,}</b>words</span>'
                f'<span><b>{analysis["sentence_count"]:,}</b>sentences</span>'
                f'<span><b>{len(analysis["timeline"]):,}</b>events</span>'
                f'<span><b>{len(text):,}</b>characters</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if not text:
        with hero_slot.container():
            st.markdown(HERO, unsafe_allow_html=True)
        return

    st.markdown('<div class="ll-workspace-rule"></div>', unsafe_allow_html=True)

    nav_col, body_col = st.columns([0.85, 3.15], gap="large")

    with nav_col:
        _label("tools")
        view = st.radio(
            "Workspace view",
            list(WORKSPACE_VIEWS),
            key="workspace_nav",
            label_visibility="collapsed",
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
                st.markdown(
                    f'<div class="ll-summary">{html.escape(summary_analysis["summary"] or "No summary could be generated.")}</div>',
                    unsafe_allow_html=True,
                )

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
