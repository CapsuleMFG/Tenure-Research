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

/* Mobile + narrow tablet: dial back type and pack the cards tighter. */
@media (max-width: 768px) {{
  .block-container {{ padding-top: 1rem; padding-left: 0.5rem; padding-right: 0.5rem; }}
  h1 {{ font-size: 1.55rem; line-height: 1.2; }}
  h2 {{ font-size: 1.25rem; }}
  h3 {{ font-size: 1.1rem; }}
  .lb-brief-headline {{ font-size: 1.0rem; padding: 0.85rem 1.0rem; line-height: 1.45; }}
  .lb-card {{ padding: 0.85rem 0.95rem 0.75rem 0.95rem; border-radius: 8px; }}
  .lb-card-title {{ font-size: 0.95rem; }}
  .lb-card-price {{ font-size: 1.5rem; }}
  .lb-card-eyebrow {{ font-size: 0.68rem; }}
  .lb-card-deltas, .lb-card-fcst {{ font-size: 0.78rem; }}
  .lb-wordmark {{ font-size: 1.2rem; }}
  .lb-wordmark-sub {{ font-size: 0.72rem; }}
  /* Streamlit's horizontal columns squeeze badly on phones; force them
     to stack and use full width when the screen is < 768px. */
  [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; }}
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    flex: 1 1 100% !important;
    width: 100% !important;
    min-width: 100% !important;
    margin-bottom: 0.5rem;
  }}
  /* Wide dataframes get a horizontal scroll instead of overflowing. */
  [data-testid="stDataFrame"] {{ overflow-x: auto; }}
  /* Tighter metric tiles on phones. */
  [data-testid="stMetricValue"] {{ font-size: 1.15rem !important; }}
  [data-testid="stMetricLabel"] {{ font-size: 0.78rem !important; }}
  [data-testid="stMetricDelta"] {{ font-size: 0.72rem !important; }}
  /* Sidebar is huge by default — let users collapse it on phones. */
  [data-testid="stSidebar"] {{ max-width: 85vw; }}
  /* Trim hero paragraph padding so the home page doesn't feel airy. */
  p {{ margin-bottom: 0.6rem; }}
}}

/* Very narrow (phone) viewport: drop type one more notch. */
@media (max-width: 480px) {{
  h1 {{ font-size: 1.35rem; }}
  .lb-card-price {{ font-size: 1.3rem; }}
  .lb-brief-headline {{ font-size: 0.95rem; padding: 0.7rem 0.85rem; }}
}}
</style>
"""


def inject_global_css() -> None:
    """Inject the brand stylesheet. Safe to call repeatedly per session."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def inject_og_tags(
    *,
    title: str,
    description: str,
    image_url: str | None = None,
) -> None:
    """Inject Open Graph + Twitter meta tags into the parent ``<head>``.

    Streamlit doesn't expose head injection directly; we use a 0-height
    component-html script that, from inside the iframe, mutates
    ``window.parent.document.head`` to add the meta tags. Most social-card
    crawlers (Twitter / X, Slack, Facebook, LinkedIn, iMessage) read OG
    tags from anywhere in the rendered HTML, so this works for them; the
    parent-head mutation makes it work for stricter crawlers too.
    """
    import json as _json

    import streamlit.components.v1 as _components

    tags: list[tuple[str, str, str]] = [
        ("property", "og:title", title),
        ("property", "og:description", description),
        ("property", "og:type", "website"),
        ("property", "og:site_name", BRAND_NAME),
        ("name", "twitter:card", "summary_large_image" if image_url else "summary"),
        ("name", "twitter:title", title),
        ("name", "twitter:description", description),
        ("name", "description", description),
    ]
    if image_url:
        tags.append(("property", "og:image", image_url))
        tags.append(("name", "twitter:image", image_url))

    payload = _json.dumps([list(t) for t in tags])
    _components.html(
        f"""
        <script>
        (function() {{
          try {{
            const head = window.parent.document.head;
            const seen = new Set(
              Array.from(head.querySelectorAll('meta'))
                .map(m => (m.getAttribute('property') || m.getAttribute('name') || ''))
            );
            const tags = {payload};
            tags.forEach(function(t) {{
              const [attr, key, content] = t;
              if (seen.has(key)) return;
              const m = window.parent.document.createElement('meta');
              m.setAttribute(attr, key);
              m.setAttribute('content', content);
              head.appendChild(m);
            }});
          }} catch (e) {{ /* iframe sandbox may block; tags still render in body */ }}
        }})();
        </script>
        """,
        height=0,
    )


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
    "inject_og_tags",
]
