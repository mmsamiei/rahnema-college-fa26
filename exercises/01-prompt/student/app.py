"""Start the student version of exercise 01."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from runtime import serve  # noqa: E402


if __name__ == "__main__":
    serve(HERE, 8765)
