"""Resume checkpoints.

The state file is rewritten after every completed prompt and judge call, so a
run interrupted at any point can be resumed with `--resume <run_dir>`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATE_FILE = "state.json"

# 1: original, unversioned.
# 2: added schema_version, load_time, separated prefill from TTFT.
# 3: every prompt is answered N times, so results and scores became lists.
SCHEMA_VERSION = 3


def save_state(run_dir: Path, state: dict[str, Any]) -> None:
    """Atomically save resume state to state.json."""
    payload = {"schema_version": SCHEMA_VERSION, **state}
    path = run_dir / STATE_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _migrate_to_v3(state: dict[str, Any]) -> dict[str, Any]:
    """Wrap single results and scores into one-element repeat lists.

    Pre-v3 runs sampled each prompt exactly once, which is exactly a run with
    `QUALITY_REPEATS=1` — so the old data survives migration intact rather than
    being discarded.
    """
    for per_model in state.get("all_results", {}).values():
        for category, result in list(per_model.items()):
            if isinstance(result, dict):
                per_model[category] = [result]
    for per_model in state.get("judge_scores", {}).values():
        for category, verdict in list(per_model.items()):
            if isinstance(verdict, dict):
                per_model[category] = [verdict]
    state["schema_version"] = SCHEMA_VERSION
    return state


def load_state(run_dir: Path) -> dict[str, Any] | None:
    """Load resume state from state.json, or None if absent.

    Older state files are migrated forward in memory; the file on disk is only
    rewritten in the new format once the resumed run checkpoints.
    """
    path = run_dir / STATE_FILE
    if not path.exists():
        return None
    state: dict[str, Any] = json.loads(path.read_text())
    version = state.get("schema_version", 1)
    if version < 3:
        state = _migrate_to_v3(state)
    elif version > SCHEMA_VERSION:
        raise SystemExit(
            f"'{path}' was written by a newer version of this tool "
            f"(schema {version} > {SCHEMA_VERSION}). Upgrade, or start a fresh run."
        )
    return state
