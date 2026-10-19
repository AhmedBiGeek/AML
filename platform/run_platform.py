"""Cross-platform launcher for Professional Profile Explorer."""

from pathlib import Path
import subprocess
import sys


APP = Path(__file__).with_name("streamlit_app.py")

raise SystemExit(
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP)],
        check=False,
    ).returncode
)
