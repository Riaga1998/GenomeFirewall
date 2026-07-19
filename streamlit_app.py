"""Root entry point for Streamlit Community Cloud.

Streamlit Cloud looks for `streamlit_app.py` at the repository root by default, so this
shim runs the real app in `app/streamlit_app.py` rather than duplicating it.

    streamlit run streamlit_app.py        # same as running app/streamlit_app.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app" / "streamlit_app.py"

exec(compile(APP.read_text(), str(APP), "exec"))
