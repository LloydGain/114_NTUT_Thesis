import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))