"""Sampling GPU state while a generation is in flight.

Ollama's response tells you how fast a model ran but nothing about what it cost
to run it. Peak VRAM is the number that decides whether a model fits on the
card at all, and it can only be observed while the work is happening.

Everything here degrades to a no-op when `nvidia-smi` is unavailable, so the
benchmark still runs on CPU-only and non-NVIDIA hosts.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from types import TracebackType

import requests

SAMPLE_INTERVAL = 0.5
QUERY = "--query-gpu=memory.used,utilization.gpu"


@dataclass
class GpuSample:
    memory_mib: float
    utilization: float


@dataclass
class GpuUsage:
    """What a generation cost, in GPU terms."""

    peak_memory_mib: float = 0.0
    baseline_memory_mib: float = 0.0
    mean_utilization: float = 0.0
    samples: int = 0

    @property
    def peak_delta_mib(self) -> float:
        """Memory attributable to this run, over what was already resident."""
        return max(self.peak_memory_mib - self.baseline_memory_mib, 0.0)

    def as_dict(self) -> dict[str, float]:
        return {
            "peak_memory_mib": self.peak_memory_mib,
            "baseline_memory_mib": self.baseline_memory_mib,
            "peak_delta_mib": self.peak_delta_mib,
            "mean_utilization": self.mean_utilization,
            "samples": float(self.samples),
        }


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def loaded_model_footprint(base_url: str, model_name: str) -> dict[str, float]:
    """Ask Ollama how much memory a loaded model is actually using.

    This is the honest source for per-model footprint. Sampling `nvidia-smi`
    around a load only yields a *global* delta, which attributes nothing when
    other models are already resident — and on a machine that keeps several
    models warm, that is the normal case. Ollama's `/api/ps` reports each
    loaded model's size and how much of it sits in VRAM.

    Returns an empty dict when the model is not resident or Ollama cannot be
    reached, so callers report "not measured" rather than a wrong zero.
    """
    try:
        resp = requests.get(f"{base_url}/api/ps", timeout=10)
        resp.raise_for_status()
        entries = resp.json().get("models", [])
    except (requests.exceptions.RequestException, ValueError):
        return {}

    for entry in entries:
        if entry.get("name") == model_name or entry.get("model") == model_name:
            total = float(entry.get("size", 0)) / 1024**2
            in_vram = float(entry.get("size_vram", 0)) / 1024**2
            return {
                "size_mib": total,
                "vram_mib": in_vram,
                # Anything not in VRAM has spilled to system RAM, which is the
                # difference between a model that fits and one that crawls.
                "offloaded_mib": max(total - in_vram, 0.0),
            }
    return {}


def read_gpus() -> list[GpuSample]:
    """One reading across every visible GPU. Empty when unavailable."""
    if not nvidia_smi_available():
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", QUERY, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []

    samples = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            samples.append(GpuSample(float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return samples


class GpuMonitor:
    """Context manager sampling GPU memory and utilisation in the background.

    Used as::

        with GpuMonitor() as monitor:
            ...generate...
        usage = monitor.usage
    """

    def __init__(self, interval: float = SAMPLE_INTERVAL) -> None:
        self.interval = interval
        self.usage = GpuUsage()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._utilizations: list[float] = []
        self._peak = 0.0

    def __enter__(self) -> GpuMonitor:
        if not nvidia_smi_available():
            return self
        baseline = read_gpus()
        self.usage.baseline_memory_mib = sum(s.memory_mib for s in baseline)
        self._peak = self.usage.baseline_memory_mib
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.usage.peak_memory_mib = self._peak
        self.usage.samples = len(self._utilizations)
        if self._utilizations:
            self.usage.mean_utilization = sum(self._utilizations) / len(self._utilizations)

    def _loop(self) -> None:
        while not self._stop.is_set():
            samples = read_gpus()
            if samples:
                total = sum(s.memory_mib for s in samples)
                self._peak = max(self._peak, total)
                self._utilizations.append(sum(s.utilization for s in samples) / len(samples))
            # Wait on the event rather than sleeping, so shutdown is immediate.
            self._stop.wait(self.interval)


def measure(interval: float = SAMPLE_INTERVAL) -> GpuMonitor:
    return GpuMonitor(interval)


def settle(seconds: float = 0.3) -> None:
    """Give the driver a moment so a baseline reading isn't taken mid-teardown."""
    time.sleep(seconds)
