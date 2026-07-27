import sys
from pathlib import Path

# Make the repo root importable so `import scrub` works without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
