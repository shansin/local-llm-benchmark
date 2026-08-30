"""Comparing runs.

Nine benchmark runs sitting in `output/` with no way to relate them is nine
isolated snapshots. What a benchmark is actually for is noticing change: a model
that regressed after an update, a quantisation that costs more quality than it
saves in memory.

Differences are read against the run's own measured noise floor, so a change is
only called a change when it exceeds what repeated sampling produces anyway.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llmbench.report.jsonout import load_json

BETTER = "improved"
WORSE = "regressed"
SAME = "unchanged"


def noise_floor(document: dict[str, Any]) -> float:
    """The widest within-task spread in a run — the bar a change must clear."""
    spreads = [m["overall"].get("repeat_std", 0.0) for m in document.get("models", [])]
    return max(spreads) if spreads else 0.0


# Settings that have to match before two runs' scores mean the same thing.
# The judge panel is the sharpest of them: swapping the judge moves every score
# in the table at once, in a direction no per-model reading can recover. The
# elicitation protocol itself is fixed in code, so it no longer appears here —
# except when one run predates the fixed protocol, which the answer_tags key
# from older documents still reveals.
_COMPARABILITY_KEYS = (
    ("judges", "judge panel"),
    ("answer_tags", "answer-tag prompting"),
)
_GENERATION_KEYS = ("num_ctx", "temperature", "top_p", "top_k", "seed")


def comparability(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Differences in how two runs were measured, not in what they measured.

    A benchmark's whole purpose is noticing change, which makes a silent change
    in the instrument the one failure it cannot survive. Two runs of this suite
    a week apart used different judge models; the leaderboard showed a model
    dropping half a point and offered no way to tell that from a regression.
    """
    old_config = before.get("config", {})
    new_config = after.get("config", {})
    notes = []

    for key, label in _COMPARABILITY_KEYS:
        old_value, new_value = old_config.get(key), new_config.get(key)
        if old_value != new_value:
            notes.append(f"{label}: `{old_value}` → `{new_value}`")

    old_gen = old_config.get("generation", {}) or {}
    new_gen = new_config.get("generation", {}) or {}
    for key in _GENERATION_KEYS:
        if key in old_gen and key in new_gen and old_gen[key] != new_gen[key]:
            notes.append(f"{key}: `{old_gen[key]}` → `{new_gen[key]}`")

    old_tasks = {t["id"] for t in before.get("tasks", [])}
    new_tasks = {t["id"] for t in after.get("tasks", [])}
    if old_tasks != new_tasks:
        added = sorted(new_tasks - old_tasks)
        removed = sorted(old_tasks - new_tasks)
        parts = []
        if added:
            parts.append(f"added {', '.join(added)}")
        if removed:
            parts.append(f"removed {', '.join(removed)}")
        notes.append(f"task set: {'; '.join(parts)}")

    return notes


def _verdict(delta: float, threshold: float) -> str:
    if abs(delta) <= threshold:
        return SAME
    return BETTER if delta > 0 else WORSE


def compare_documents(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Diff two result documents model by model and task by task."""
    # A difference has to beat the noise in either run to count.
    threshold = max(noise_floor(before), noise_floor(after))

    before_models = {m["name"]: m for m in before.get("models", [])}
    after_models = {m["name"]: m for m in after.get("models", [])}

    rows = []
    for name in sorted(set(before_models) | set(after_models)):
        old = before_models.get(name)
        new = after_models.get(name)
        if old is None:
            rows.append({"model": name, "status": "added"})
            continue
        if new is None:
            rows.append({"model": name, "status": "removed"})
            continue

        old_score = old["overall"].get("mean")
        new_score = new["overall"].get("mean")
        entry: dict[str, Any] = {
            "model": name,
            "status": "present",
            "before": old_score,
            "after": new_score,
        }
        if old_score is not None and new_score is not None:
            entry["delta"] = new_score - old_score
            entry["verdict"] = _verdict(entry["delta"], threshold)

        old_tps = old.get("throughput", {}).get("tps_median", 0.0)
        new_tps = new.get("throughput", {}).get("tps_median", 0.0)
        if old_tps and new_tps:
            entry["tps_before"] = old_tps
            entry["tps_after"] = new_tps
            entry["tps_delta_pct"] = (new_tps - old_tps) / old_tps * 100

        entry["tasks"] = _compare_tasks(old, new, threshold)
        rows.append(entry)

    return {
        "before": before.get("run"),
        "after": after.get("run"),
        "noise_threshold": threshold,
        "comparability": comparability(before, after),
        "models": rows,
    }


def _compare_tasks(
    old: dict[str, Any], new: dict[str, Any], threshold: float
) -> list[dict[str, Any]]:
    changes = []
    old_tasks = old.get("tasks", {})
    new_tasks = new.get("tasks", {})
    for key in sorted(set(old_tasks) & set(new_tasks)):
        before = old_tasks[key]["blended"].get("mean")
        after = new_tasks[key]["blended"].get("mean")
        if before is None or after is None:
            continue
        delta = after - before
        verdict = _verdict(delta, threshold)
        if verdict != SAME:
            changes.append(
                {
                    "task": key,
                    "category": new_tasks[key].get("category"),
                    "before": before,
                    "after": after,
                    "delta": delta,
                    "verdict": verdict,
                }
            )
    return sorted(changes, key=lambda c: c["delta"])


def render_comparison(diff: dict[str, Any]) -> str:
    """Render a comparison as markdown."""
    lines = [f"# Comparison — {diff['before']} → {diff['after']}", ""]

    changes = diff.get("comparability") or []
    if changes:
        lines += [
            "> **These runs were not measured the same way.** The differences below "
            "move scores on their own, so a delta in this comparison is not "
            "necessarily a change in the models:",
            "",
        ]
        lines += [f"> - {note}" for note in changes]
        lines.append("")

    threshold = diff["noise_threshold"]
    if threshold:
        lines.append(
            f"Differences of {threshold:.2f} points or less are within measured sampling "
            "noise and are reported as unchanged."
        )
    else:
        lines.append(
            "**No noise estimate** (each prompt is sampled once at temperature 0), so a "
            "small difference below could be sampling variation rather than change. "
            "Trust deltas that are large or consistent across tasks."
        )
    lines.append("")

    lines.append("| Model | Before | After | Δ | Verdict | Tok/s Δ |")
    lines.append("|-------|--------|-------|---|---------|---------|")
    for row in diff["models"]:
        if row["status"] != "present":
            lines.append(f"| {row['model']} | — | — | — | {row['status']} | — |")
            continue
        before = "—" if row.get("before") is None else f"{row['before']:.2f}"
        after = "—" if row.get("after") is None else f"{row['after']:.2f}"
        delta = f"{row['delta']:+.2f}" if "delta" in row else "—"
        tps = f"{row['tps_delta_pct']:+.1f}%" if "tps_delta_pct" in row else "—"
        lines.append(
            f"| {row['model']} | {before} | {after} | {delta} | {row.get('verdict', '—')} | {tps} |"
        )

    changed = [(r["model"], t) for r in diff["models"] for t in r.get("tasks", [])]
    if changed:
        lines += ["", "## Task-level changes beyond the noise floor", ""]
        lines.append("| Model | Task | Before | After | Δ |")
        lines.append("|-------|------|--------|-------|---|")
        for model, task in changed:
            lines.append(
                f"| {model} | {task['category']}/{task['task']} | {task['before']:.1f} "
                f"| {task['after']:.1f} | {task['delta']:+.1f} |"
            )
    else:
        lines += ["", "No task-level change exceeded the noise floor."]

    return "\n".join(lines) + "\n"


def leaderboard(documents: list[dict[str, Any]]) -> str:
    """Best score each model has achieved across every run, and where."""
    best: dict[str, dict[str, Any]] = {}
    for document in documents:
        for model in document.get("models", []):
            score = model["overall"].get("mean")
            if score is None:
                continue
            current = best.get(model["name"])
            if current is None or score > current["score"]:
                best[model["name"]] = {
                    "score": score,
                    "run": document.get("run"),
                    "tps": model.get("throughput", {}).get("tps_median", 0.0),
                    "vram": model.get("memory", {}).get("vram_mib", 0.0),
                }

    lines = [f"# Leaderboard across {len(documents)} run(s)", ""]

    panels = {
        ", ".join(document.get("config", {}).get("judges", []) or ["?"]) for document in documents
    }
    if len(panels) > 1:
        lines += [
            "> **Mixed judges.** These runs were scored by different judge panels "
            f"({'; '.join(sorted(panels))}), so a best-of across them partly ranks "
            "which judge was most generous. Compare runs sharing a panel instead.",
            "",
        ]

    lines.append("| Model | Best score | Tok/s | VRAM (MiB) | From run |")
    lines.append("|-------|------------|-------|------------|----------|")
    for name, entry in sorted(best.items(), key=lambda kv: kv[1]["score"], reverse=True):
        vram = f"{entry['vram']:.0f}" if entry["vram"] else "—"
        lines.append(
            f"| {name} | {entry['score']:.2f} | {entry['tps']:.1f} | {vram} | {entry['run']} |"
        )
    return "\n".join(lines) + "\n"


def load_run(path: Path) -> dict[str, Any] | None:
    """Load a run's canonical results, if it has any."""
    return load_json(path)
