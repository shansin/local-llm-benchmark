"""Summary statistics over repeated measurements.

Every helper tolerates empty and single-element inputs, because a model that
timed out or was only run once still has to appear in the report.
"""

from __future__ import annotations

import statistics


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def stdev(values: list[float]) -> float:
    """Sample standard deviation; 0.0 when there's nothing to spread."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile, p in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (p / 100.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def iqr(values: list[float]) -> float:
    """Interquartile range — the spread measure reported next to the median."""
    return percentile(values, 75) - percentile(values, 25)
