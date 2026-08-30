"""A single-file HTML report.

The decision a local-model benchmark actually informs is a trade: quality
against speed against memory. A table makes you compute that in your head; a
scatter plot shows it. Everything here is inline — no CDN, no fonts, no scripts
— so the file works offline and can be handed to someone as-is.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

REPORT_FILE = "report.html"

CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #666; --line: #e0e0e0;
  --accent: #2a6fb5; --good: #2e7d4f; --bad: #b3402f; --panel: #f7f7f8;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16181c; --fg: #e8e8ea; --muted: #9aa0a6; --line: #2e3238;
    --accent: #6aa9e8; --good: #6cc48c; --bad: #e8836f; --panel: #1e2126;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem; background: var(--bg); color: var(--fg);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.7rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2.5rem 0 .5rem; padding-bottom: .3rem;
     border-bottom: 1px solid var(--line); }
p.note { color: var(--muted); margin: .25rem 0 1rem; }
.meta { color: var(--muted); font-size: .875rem; margin-bottom: 1.5rem; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: right; padding: .45rem .6rem; border-bottom: 1px solid var(--line);
         white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { font-weight: 600; color: var(--muted); font-size: .8rem;
           text-transform: uppercase; letter-spacing: .03em; }
tbody tr:hover { background: var(--panel); }
.best { font-weight: 700; color: var(--accent); }
.pos { color: var(--good); } .neg { color: var(--bad); }
figure { margin: 0 0 1rem; }
svg { max-width: 100%; height: auto; display: block; }
.legend { display: flex; flex-wrap: wrap; gap: .75rem; font-size: .82rem;
          color: var(--muted); margin-top: .5rem; }
.legend span { display: flex; align-items: center; gap: .35rem; }
.swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
footer { margin-top: 3rem; color: var(--muted); font-size: .8rem; }
"""

# Colour-blind-safe categorical palette, readable on both themes.
PALETTE = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#a463f2", "#9c9c9c", "#3ca951", "#ff8ab7"]


def _esc(text: Any) -> str:
    return html.escape(str(text))


def _table(columns: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{_esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return (
        '<div class="scroll"><table>'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody>"
        "</table></div>"
    )


def _scatter(models: list[dict[str, Any]]) -> str:
    """Quality against speed — the actual trade being made."""
    points = [
        (
            m["throughput"].get("tps_median", 0.0),
            m["overall"].get("mean"),
            m["name"],
            m.get("memory", {}).get("vram_mib", 0.0),
        )
        for m in models
    ]
    points = [p for p in points if p[1] is not None and p[0] > 0]
    if len(points) < 2:
        return ""

    width, height = 720, 380
    pad_l, pad_r, pad_t, pad_b = 56, 24, 20, 46
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_max = max(xs) * 1.12
    y_min, y_max = min(ys) - 0.6, max(ys) + 0.6
    y_min, y_max = max(0.0, y_min), min(10.0, y_max)
    if y_max - y_min < 1:
        y_min, y_max = max(0.0, y_min - 0.5), min(10.0, y_max + 0.5)

    def sx(value: float) -> float:
        return pad_l + (value / x_max) * (width - pad_l - pad_r)

    def sy(value: float) -> float:
        return height - pad_b - ((value - y_min) / (y_max - y_min)) * (height - pad_t - pad_b)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Quality against generation speed">'
    ]
    parts.append(
        f'<line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" '
        'stroke="var(--line)"/>'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" stroke="var(--line)"/>'
    )

    steps = 4
    for i in range(steps + 1):
        value = y_min + (y_max - y_min) * i / steps
        y = sy(value)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            'stroke="var(--line)" stroke-dasharray="2 4"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="var(--muted)">{value:.1f}</text>'
        )
        value_x = x_max * i / steps
        x = sx(value_x)
        parts.append(
            f'<text x="{x:.1f}" y="{height - pad_b + 18}" text-anchor="middle" font-size="11" '
            f'fill="var(--muted)">{value_x:.0f}</text>'
        )

    max_vram = max((p[3] for p in points), default=0.0)
    for i, (tps, score, name, vram) in enumerate(points):
        colour = PALETTE[i % len(PALETTE)]
        radius = 6 + (10 * (vram / max_vram) if max_vram else 0)
        parts.append(
            f'<circle cx="{sx(tps):.1f}" cy="{sy(score):.1f}" r="{radius:.1f}" fill="{colour}" '
            f'fill-opacity="0.75" stroke="{colour}"><title>{_esc(name)}: {score:.2f} score, '
            f"{tps:.0f} tok/s, {vram:.0f} MiB</title></circle>"
        )

    parts.append(
        f'<text x="{(width) / 2:.0f}" y="{height - 8}" text-anchor="middle" font-size="12" '
        'fill="var(--muted)">Generation speed (tok/s)</text>'
        f'<text x="14" y="{height / 2:.0f}" font-size="12" fill="var(--muted)" '
        f'transform="rotate(-90 14 {height / 2:.0f})" text-anchor="middle">Blended score</text>'
    )
    parts.append("</svg>")

    legend = "".join(
        f'<span><i class="swatch" style="background:{PALETTE[i % len(PALETTE)]}"></i>'
        f"{_esc(p[2])}</span>"
        for i, p in enumerate(points)
    )
    return (
        "<figure>"
        + "".join(parts)
        + f'<div class="legend">{legend}</div>'
        + "</figure>"
        + '<p class="note">Up and to the right is better. Circle size is VRAM footprint.</p>'
    )


def _category_bars(models: list[dict[str, Any]], categories: list[str]) -> str:
    """Per-category scores, grouped by category."""
    if not models or not categories:
        return ""

    row_height = 26
    group_gap = 14
    label_w = 120
    width = 720
    bar_area = width - label_w - 60
    height = len(categories) * (len(models) * row_height + group_gap) + 20

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Scores by category">']
    y = 10
    for category in categories:
        parts.append(
            f'<text x="0" y="{y + 14}" font-size="12" font-weight="600" '
            f'fill="var(--fg)">{_esc(category.title())}</text>'
        )
        for i, model in enumerate(models):
            score = model["by_category"].get(category, {}).get("mean")
            bar_y = y + i * row_height + 20
            colour = PALETTE[i % len(PALETTE)]
            if score is None:
                parts.append(
                    f'<text x="{label_w}" y="{bar_y + 12}" font-size="11" '
                    'fill="var(--muted)">not scored</text>'
                )
                continue
            bar_w = max((score / 10.0) * bar_area, 1)
            parts.append(
                f'<rect x="{label_w}" y="{bar_y}" width="{bar_w:.1f}" height="16" rx="3" '
                f'fill="{colour}" fill-opacity="0.85"><title>{_esc(model["name"])}: '
                f"{score:.2f}</title></rect>"
                f'<text x="{label_w + bar_w + 6:.1f}" y="{bar_y + 13}" font-size="11" '
                f'fill="var(--muted)">{score:.1f}</text>'
            )
        y += len(models) * row_height + group_gap
    parts.append("</svg>")

    legend = "".join(
        f'<span><i class="swatch" style="background:{PALETTE[i % len(PALETTE)]}"></i>'
        f"{_esc(m['name'])}</span>"
        for i, m in enumerate(models)
    )
    return "<figure>" + "".join(parts) + f'<div class="legend">{legend}</div></figure>'


def _calibration(models: list[dict[str, Any]]) -> str:
    """How far the judge sits from the measured result, where both exist."""
    rows = []
    for model in models:
        deltas = []
        for task in model["tasks"].values():
            judge = task["judge"].get("mean")
            objective = task["objective"].get("mean")
            if judge is not None and objective is not None:
                deltas.append(judge - objective)
        if not deltas:
            continue
        average = sum(deltas) / len(deltas)
        css = "pos" if average > 0.5 else ("neg" if average < -0.5 else "")
        rows.append(
            [
                _esc(model["name"]),
                f"{len(deltas)}",
                f'<span class="{css}">{average:+.2f}</span>',
            ]
        )
    if not rows:
        return ""
    return (
        "<h2>Judge calibration</h2>"
        '<p class="note">Mean judge score minus measured score, on tasks where both exist. '
        "Positive means the judge is more generous than the evidence supports.</p>"
        + _table(["Model", "Tasks compared", "Judge − measured"], rows)
    )


def render(document: dict[str, Any]) -> str:
    """Render a results document as a standalone HTML page."""
    models = document.get("models", [])
    categories = document.get("categories", [])
    config = document.get("config", {})
    host = document.get("host", {})

    scored = [m for m in models if m["overall"].get("mean") is not None]
    best = max((m["overall"]["mean"] for m in scored), default=None)
    fastest = max((m["throughput"].get("tps_median", 0.0) for m in models), default=0.0)

    summary_rows = []
    for model in models:
        score = model["overall"].get("mean")
        tps = model["throughput"].get("tps_median", 0.0)
        vram = model.get("memory", {}).get("vram_mib", 0.0)
        score_cell = "—" if score is None else f"{score:.2f}"
        if score is not None and best is not None and abs(score - best) < 1e-9:
            score_cell = f'<span class="best">{score:.2f}</span>'
        tps_cell = f"{tps:.1f}" if tps else "—"
        if tps and abs(tps - fastest) < 1e-9:
            tps_cell = f'<span class="best">{tps:.1f}</span>'
        summary_rows.append(
            [
                _esc(model["name"]),
                _esc(model["details"].get("parameter_size", "?")),
                _esc(model["details"].get("quantization_level", "?")),
                score_cell,
                tps_cell,
                f"{vram:.0f}" if vram else "—",
                f"{model.get('cold_load_seconds', 0.0):.1f}",
            ]
        )

    category_rows = []
    for model in models:
        cells = [_esc(model["name"])]
        for category in categories:
            value = model["by_category"].get(category, {}).get("mean")
            cells.append("—" if value is None else f"{value:.1f}")
        category_rows.append(cells)

    noise_note = (
        "Each prompt is sampled once at temperature 0, so treat small differences "
        "between models as ties."
    )

    gpus = "; ".join(host.get("gpus") or []) or "Not detected"
    generation = ", ".join(f"{k}={v}" for k, v in (config.get("generation") or {}).items())

    body = f"""<main>
<h1>Benchmark — {_esc(document.get("run", ""))}</h1>
<div class="meta">
  {_esc(host.get("cpu") or "Unknown CPU")} · {_esc(host.get("ram") or "?")} RAM · {_esc(gpus)}<br>
  {len(document.get("tasks", []))} tasks ·
  judges: {_esc(", ".join(config.get("judges") or []) or "none")}<br>
  <code>{_esc(generation)}</code>
</div>

<h2>Quality against speed</h2>
{_scatter(models)}

<h2>Summary</h2>
<p class="note">{_esc(noise_note)}</p>
{
        _table(
            ["Model", "Params", "Quant", "Score", "Tok/s", "VRAM (MiB)", "Load (s)"],
            summary_rows,
        )
    }

<h2>By category</h2>
{_category_bars(models, categories)}
{_table(["Model", *[c.title() for c in categories]], category_rows)}

{_calibration(models)}

<footer>Generated by local-llm-benchmark. Full data in <code>results.json</code>.</footer>
</main>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark — {_esc(document.get("run", ""))}</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


def write_html(run_dir: Path, document: dict[str, Any]) -> None:
    (run_dir / REPORT_FILE).write_text(render(document))
