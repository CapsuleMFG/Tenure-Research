"""Visual identity: global CSS for LivestockBrief.

Streamlit's default theme picks up the colors in ``.streamlit/config.toml``;
this module layers on typography, card chrome, and headline styling via
:func:`inject_global_css`. Call that helper once per page module.

Page config (title, favicon, layout) is set centrally in
``streamlit_app.py`` because ``st.navigation`` requires
``st.set_page_config`` to be called once before navigation runs.
"""

from __future__ import annotations

import streamlit as st

BRAND_NAME = "LivestockBrief"
BRAND_TAGLINE = "USDA-grounded livestock price forecasts in plain English."

# Clay/rust accent for primary, parchment for background.
ACCENT = "#B4521E"
ACCENT_SOFT = "#E8C7B0"
PARCHMENT = "#FAF7F2"
PARCHMENT_DEEP = "#F0EBE1"
INK = "#1F1F1F"
INK_SOFT = "#5A5550"
UP = "#3E7D5A"
DOWN = "#B4521E"

_GLOBAL_CSS = f"""
<style>
/* Typography ---------------------------------------------------------- */
html, body, [class*="css"], .stMarkdown, .stApp {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto,
               "Helvetica Neue", Arial, sans-serif;
  color: {INK};
}}

h1, h2, h3, h4 {{
  font-family: "Iowan Old Style", "Source Serif Pro", Georgia, "Times New Roman",
               serif;
  color: {INK};
  letter-spacing: -0.01em;
}}

h1 {{ font-weight: 600; }}
h2 {{ font-weight: 600; margin-top: 1.5rem; }}
h3 {{ font-weight: 600; }}

/* Brand wordmark (used in sidebar) ------------------------------------ */
.lb-wordmark {{
  font-family: "Iowan Old Style", "Source Serif Pro", Georgia, serif;
  font-size: 1.45rem;
  font-weight: 600;
  color: {ACCENT};
  letter-spacing: -0.01em;
  line-height: 1.1;
}}
.lb-wordmark-sub {{
  font-size: 0.78rem;
  color: {INK_SOFT};
  margin-top: 0.15rem;
}}

/* Sidebar styling ----------------------------------------------------- */
[data-testid="stSidebar"] {{
  background-color: {PARCHMENT_DEEP};
  border-right: 1px solid rgba(180, 82, 30, 0.12);
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {{
  font-size: 0.95rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: {INK_SOFT};
  font-family: -apple-system, "Segoe UI", Inter, sans-serif;
  font-weight: 600;
  margin-top: 1.2rem;
}}

/* Headline brief block ----------------------------------------------- */
.lb-brief-headline {{
  font-family: "Iowan Old Style", "Source Serif Pro", Georgia, serif;
  font-size: 1.18rem;
  line-height: 1.55;
  color: {INK};
  padding: 1.1rem 1.25rem;
  background: linear-gradient(180deg, #FFFBF5 0%, {PARCHMENT} 100%);
  border-left: 4px solid {ACCENT};
  border-radius: 4px;
  margin: 0.5rem 0 1.5rem 0;
}}
.lb-brief-headline em {{ color: {ACCENT}; font-style: normal; font-weight: 600; }}

/* Commodity cards ---------------------------------------------------- */
.lb-card {{
  background: #FFFFFF;
  border: 1px solid rgba(180, 82, 30, 0.18);
  border-radius: 10px;
  padding: 1.0rem 1.1rem 0.85rem 1.1rem;
  height: 100%;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
  transition: box-shadow 0.15s ease, transform 0.15s ease;
}}
.lb-card:hover {{
  box-shadow: 0 3px 10px rgba(180, 82, 30, 0.10);
  transform: translateY(-1px);
}}
.lb-card-eyebrow {{
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: {INK_SOFT};
  margin-bottom: 0.1rem;
}}
.lb-card-title {{
  font-family: "Iowan Old Style", "Source Serif Pro", Georgia, serif;
  font-size: 1.02rem;
  font-weight: 600;
  color: {INK};
  margin-bottom: 0.55rem;
  line-height: 1.25;
}}
.lb-card-price {{
  font-size: 1.85rem;
  font-weight: 600;
  color: {INK};
  letter-spacing: -0.02em;
  line-height: 1.0;
}}
.lb-card-unit {{
  font-size: 0.78rem;
  color: {INK_SOFT};
  margin-left: 0.25rem;
}}
.lb-card-deltas {{
  margin-top: 0.4rem;
  font-size: 0.82rem;
  color: {INK_SOFT};
}}
.lb-up    {{ color: {UP}; font-weight: 600; }}
.lb-down  {{ color: {DOWN}; font-weight: 600; }}
.lb-card-fcst {{
  font-size: 0.85rem;
  color: {INK};
  margin-top: 0.65rem;
  padding-top: 0.5rem;
  border-top: 1px dashed rgba(180, 82, 30, 0.22);
}}
.lb-card-fcst-label {{
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: {INK_SOFT};
  margin-bottom: 0.1rem;
}}

/* Buttons --------------------------------------------------------- */
.stButton > button {{
  border-radius: 6px;
  border: 1px solid rgba(180, 82, 30, 0.25);
  font-weight: 500;
}}
.stButton > button[kind="primary"] {{
  background: {ACCENT};
  border-color: {ACCENT};
  color: white;
}}
.stButton > button[kind="primary"]:hover {{
  background: #9B461A;
  border-color: #9B461A;
}}

/* Misc -------------------------------------------------------------- */
hr {{ border-color: rgba(180, 82, 30, 0.18); }}
a {{ color: {ACCENT}; }}
a:hover {{ color: #9B461A; }}

/* Hide Streamlit's default footer "Made with Streamlit" branding ---- */
footer {{ visibility: hidden; }}

/* Trim top padding so headlines sit closer to the top ---------------- */
.block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
</style>
"""


def inject_global_css() -> None:
    """Inject the brand stylesheet. Safe to call repeatedly per session."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


__all__ = [
    "ACCENT",
    "ACCENT_SOFT",
    "BRAND_NAME",
    "BRAND_TAGLINE",
    "DOWN",
    "INK",
    "INK_SOFT",
    "PARCHMENT",
    "PARCHMENT_DEEP",
    "UP",
    "inject_global_css",
]
