"""Streamlit Community Cloud entrypoint.

Streamlit Cloud points at a single file at the repo root. This thin shim
runs the dashboard's real entrypoint by importing it; that way local
``uv run streamlit run dashboard/app.py`` and the deployed app stay in sync.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Ensure both ``dashboard/`` (for the ``components.*`` imports inside the
# dashboard package) and ``src/`` (for ``usda_sandbox.*``) are importable
# whether running locally or on Streamlit Cloud.
ROOT = Path(__file__).resolve().parent
for sub in (ROOT / "dashboard", ROOT / "src"):
    p = str(sub)
    if p not in sys.path:
        sys.path.insert(0, p)

# Execute the real landing page in this process so Streamlit's session
# state and page-discovery (it scans ``pages/`` *relative to the entrypoint
# directory*) work correctly.
runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")
