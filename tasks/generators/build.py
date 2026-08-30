"""Regenerate every task whose answer key is computed rather than written.

uv run python tasks/generators/build.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import longcontext  # noqa: E402
import statetrack  # noqa: E402
import transformation  # noqa: E402


def main() -> None:
    for module in (longcontext, statetrack, transformation):
        module.build()
        print(f"built {module.__name__}")


if __name__ == "__main__":
    main()
