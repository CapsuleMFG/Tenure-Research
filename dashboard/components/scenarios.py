"""Named-scenario save/load using browser localStorage.

Plan inputs already round-trip through ``st.query_params`` — so the
URL itself IS the full state. A "scenario" is just a (name, querystring)
pair. We store the dict of those pairs in ``localStorage`` on the
browser side, and render the save/load UI as a single
``components.html`` block of vanilla JS that mutates the parent frame's
URL on click.

Why pure JS instead of a bidirectional Streamlit component? localStorage
is a browser-only API and we don't need anything from Python side
beyond writing the panel. The component renders, the user clicks, the
parent URL changes, Streamlit re-runs the page with the new query
params, and the regional-preset + ``_qp_get_*`` machinery in the Plan
page does the rest. No bridge required.

Limitations of this approach:
* Scenarios live on the device they were saved from. A producer with
  a desktop and a phone has two scenario lists. Acceptable for a free
  public dashboard; would need a real backend to fix.
* localStorage is sandboxed per host; a deploy URL change wipes saved
  scenarios. Document this so users aren't surprised.
"""

from __future__ import annotations

import streamlit.components.v1 as components

# Approx. minimum height needed so the iframe doesn't clip the JS-rendered
# save form + scenario list. The component grows internally; we just need
# enough room for the controls.
_DEFAULT_HEIGHT = 200


_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  :root {{
    --accent: {accent};
    --ink: #1F1F1F;
    --ink-soft: #5A5550;
    --parchment: {parchment_deep};
    --border: rgba(180, 82, 30, 0.25);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
    color: var(--ink);
    background: transparent;
  }}
  .lb-scn-row {{
    display: flex; flex-wrap: wrap; gap: 0.5rem;
    align-items: center;
    margin-bottom: 0.7rem;
  }}
  .lb-scn-input {{
    flex: 1 1 220px;
    padding: 0.45rem 0.7rem;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: #FFFFFF;
    font-size: 0.95rem;
  }}
  .lb-scn-btn {{
    padding: 0.45rem 0.85rem;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: #FFFFFF;
    color: var(--ink);
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 500;
  }}
  .lb-scn-btn:hover {{ background: var(--parchment); }}
  .lb-scn-btn-primary {{
    background: var(--accent); color: white; border-color: var(--accent);
  }}
  .lb-scn-btn-primary:hover {{ background: #9B461A; border-color: #9B461A; }}
  .lb-scn-btn-danger {{
    background: transparent;
    color: var(--ink-soft);
    border: none;
    padding: 0.1rem 0.4rem;
    font-size: 1.05rem;
  }}
  .lb-scn-btn-danger:hover {{ color: var(--accent); }}
  .lb-scn-saved {{
    display: flex; flex-direction: column; gap: 0.35rem;
    margin-top: 0.4rem;
  }}
  .lb-scn-saved-item {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    background: #FFFFFF;
    border: 1px solid var(--border);
  }}
  .lb-scn-saved-link {{
    color: var(--accent); text-decoration: none;
    font-weight: 500; cursor: pointer;
    flex: 1;
  }}
  .lb-scn-saved-link:hover {{ text-decoration: underline; }}
  .lb-scn-saved-meta {{
    font-size: 0.78rem; color: var(--ink-soft); margin-right: 0.6rem;
  }}
  .lb-scn-empty {{
    color: var(--ink-soft);
    font-size: 0.9rem;
    font-style: italic;
    padding: 0.4rem 0;
  }}
  .lb-scn-status {{
    font-size: 0.82rem; color: var(--ink-soft);
    margin-top: 0.3rem; min-height: 1.1em;
  }}
</style>
</head>
<body>
<div class="lb-scn-row">
  <input id="lb_name" class="lb-scn-input" type="text"
         placeholder="Name this scenario (e.g., 'Current — 60 cows')"
         maxlength="60">
  <button id="lb_save" class="lb-scn-btn lb-scn-btn-primary">Save current</button>
</div>
<div id="lb_status" class="lb-scn-status"></div>
<div id="lb_saved" class="lb-scn-saved"></div>

<script>
const STORAGE_KEY = "{storage_key}";
const $name = document.getElementById("lb_name");
const $save = document.getElementById("lb_save");
const $list = document.getElementById("lb_saved");
const $status = document.getElementById("lb_status");

function getParentSearch() {{
  // Streamlit renders this iframe inside the parent document.
  // ``window.parent.location.search`` is what's in the address bar.
  try {{
    return window.parent.location.search || "";
  }} catch (e) {{
    return "";
  }}
}}

function setParentSearch(qs) {{
  try {{
    // Navigate the parent frame to the new querystring; Streamlit
    // will pick up the new st.query_params on the next rerun.
    window.parent.location.search = qs;
  }} catch (e) {{
    $status.textContent = "Can't navigate the parent frame: " + e.message;
  }}
}}

function loadAll() {{
  try {{
    const raw = window.parent.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {{}};
    return JSON.parse(raw) || {{}};
  }} catch (e) {{
    return {{}};
  }}
}}

function saveAll(scenarios) {{
  try {{
    window.parent.localStorage.setItem(STORAGE_KEY, JSON.stringify(scenarios));
  }} catch (e) {{
    $status.textContent = "Couldn't save (localStorage blocked): " + e.message;
  }}
}}

function render() {{
  const scenarios = loadAll();
  const names = Object.keys(scenarios).sort();
  $list.innerHTML = "";
  if (names.length === 0) {{
    const empty = document.createElement("div");
    empty.className = "lb-scn-empty";
    empty.textContent = (
      "No saved scenarios yet. Enter a name above and click Save current " +
      "to capture the current inputs. Scenarios are stored in this browser only."
    );
    $list.appendChild(empty);
    return;
  }}
  for (const name of names) {{
    const row = document.createElement("div");
    row.className = "lb-scn-saved-item";
    const entry = scenarios[name];
    const qs = (typeof entry === "string") ? entry : entry.qs;
    const savedAt = (typeof entry === "object" && entry.savedAt) ? entry.savedAt : "";

    const link = document.createElement("a");
    link.className = "lb-scn-saved-link";
    link.textContent = name;
    link.title = "Load this scenario";
    link.onclick = (ev) => {{
      ev.preventDefault();
      setParentSearch(qs);
    }};
    row.appendChild(link);

    if (savedAt) {{
      const meta = document.createElement("span");
      meta.className = "lb-scn-saved-meta";
      meta.textContent = savedAt;
      row.appendChild(meta);
    }}

    const del = document.createElement("button");
    del.className = "lb-scn-btn lb-scn-btn-danger";
    del.textContent = "✕";
    del.title = "Delete this scenario";
    del.onclick = () => {{
      if (!window.confirm("Delete scenario \\"" + name + "\\"?")) return;
      const all = loadAll();
      delete all[name];
      saveAll(all);
      $status.textContent = "Deleted \\"" + name + "\\".";
      render();
    }};
    row.appendChild(del);
    $list.appendChild(row);
  }}
}}

$save.onclick = () => {{
  const name = ($name.value || "").trim();
  if (!name) {{
    $status.textContent = "Enter a name first.";
    return;
  }}
  const qs = getParentSearch();
  if (!qs) {{
    $status.textContent = (
      "Nothing to save yet — enter some inputs above (or pick a region) "
      + "and click 'Copy shareable link' first so the URL captures your state."
    );
    return;
  }}
  const all = loadAll();
  const savedAt = new Date().toISOString().slice(0, 10);
  all[name] = {{ qs: qs, savedAt: savedAt }};
  saveAll(all);
  $name.value = "";
  $status.textContent = "Saved \\"" + name + "\\".";
  render();
}};

$name.addEventListener("keydown", (ev) => {{
  if (ev.key === "Enter") $save.click();
}});

render();
</script>
</body>
</html>
"""


def render_scenarios_panel(
    *,
    accent: str = "#B4521E",
    parchment_deep: str = "#F0EBE1",
    storage_key: str = "lb_scenarios_v1",
    height: int = _DEFAULT_HEIGHT,
) -> None:
    """Render the save/load UI for named scenarios.

    Place this above the rest of the page; the panel is self-contained
    and writes its state to ``localStorage[storage_key]`` on the user's
    browser.
    """
    html = _HTML_TEMPLATE.format(
        accent=accent,
        parchment_deep=parchment_deep,
        storage_key=storage_key,
    )
    components.html(html, height=height, scrolling=False)


__all__ = ["render_scenarios_panel"]
