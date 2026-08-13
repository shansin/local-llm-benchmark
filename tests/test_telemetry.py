from llmbench import telemetry
from llmbench.runner import build_probe_prompt
from llmbench.scoring.aggregate import peak_vram


def test_monitor_is_a_noop_without_nvidia_smi(monkeypatch):
    """The benchmark must still run on CPU-only and non-NVIDIA hosts."""
    monkeypatch.setattr(telemetry, "nvidia_smi_available", lambda: False)
    with telemetry.measure() as monitor:
        pass
    assert monitor.usage.samples == 0
    assert monitor.usage.peak_delta_mib == 0.0


def test_peak_delta_is_measured_over_the_baseline():
    usage = telemetry.GpuUsage(peak_memory_mib=9000.0, baseline_memory_mib=1000.0)
    assert usage.peak_delta_mib == 8000.0


def test_peak_delta_never_goes_negative():
    usage = telemetry.GpuUsage(peak_memory_mib=500.0, baseline_memory_mib=900.0)
    assert usage.peak_delta_mib == 0.0


def test_read_gpus_parses_nvidia_smi_output(monkeypatch):
    class _Result:
        returncode = 0
        stdout = "8192, 74\n4096, 30\n"

    monkeypatch.setattr(telemetry, "nvidia_smi_available", lambda: True)
    monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: _Result())
    samples = telemetry.read_gpus()
    assert [s.memory_mib for s in samples] == [8192.0, 4096.0]
    assert samples[0].utilization == 74.0


def test_read_gpus_survives_garbage_output(monkeypatch):
    class _Result:
        returncode = 0
        stdout = "not, numbers\n\n8192, 50\n"

    monkeypatch.setattr(telemetry, "nvidia_smi_available", lambda: True)
    monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: _Result())
    assert len(telemetry.read_gpus()) == 1


def test_read_gpus_survives_a_failing_nvidia_smi(monkeypatch):
    monkeypatch.setattr(telemetry, "nvidia_smi_available", lambda: True)

    def _boom(*a, **k):
        raise OSError("nvidia-smi exploded")

    monkeypatch.setattr(telemetry.subprocess, "run", _boom)
    assert telemetry.read_gpus() == []


def _ps_response(models):
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": models}

    return _Resp()


def test_footprint_is_attributed_per_model_not_by_global_delta(monkeypatch):
    """On a host keeping several models warm, a global delta attributes nothing."""
    monkeypatch.setattr(
        telemetry.requests,
        "get",
        lambda *a, **k: _ps_response(
            [
                {"name": "other:7b", "size": 5 * 1024**3, "size_vram": 5 * 1024**3},
                {"name": "llama3.1:8b", "size": 7 * 1024**3, "size_vram": 7 * 1024**3},
            ]
        ),
    )
    usage = telemetry.loaded_model_footprint("http://x", "llama3.1:8b")
    assert usage["vram_mib"] == 7168.0
    assert usage["offloaded_mib"] == 0.0


def test_footprint_reports_the_part_that_spilled_to_system_ram():
    """A model that does not fit runs, but slowly — that has to be visible."""

    def _get(*a, **k):
        return _ps_response([{"name": "big:70b", "size": 40 * 1024**3, "size_vram": 16 * 1024**3}])

    import llmbench.telemetry as t

    original = t.requests.get
    t.requests.get = _get
    try:
        usage = t.loaded_model_footprint("http://x", "big:70b")
    finally:
        t.requests.get = original
    assert usage["vram_mib"] == 16384.0
    assert usage["offloaded_mib"] == 24576.0


def test_footprint_is_empty_when_the_model_is_not_resident(monkeypatch):
    monkeypatch.setattr(telemetry.requests, "get", lambda *a, **k: _ps_response([]))
    assert telemetry.loaded_model_footprint("http://x", "absent") == {}


def test_footprint_is_empty_when_ollama_is_unreachable(monkeypatch):
    def _boom(*a, **k):
        raise telemetry.requests.exceptions.ConnectionError()

    monkeypatch.setattr(telemetry.requests, "get", _boom)
    assert telemetry.loaded_model_footprint("http://x", "m") == {}


def test_peak_vram_across_repeats_takes_the_maximum():
    results = [
        {"error": None, "gpu": {"peak_delta_mib": 4000.0}},
        {"error": None, "gpu": {"peak_delta_mib": 6000.0}},
    ]
    assert peak_vram(results) == 6000.0


def test_peak_vram_is_zero_when_never_measured():
    assert peak_vram([{"error": None, "gpu": None}]) == 0.0
    assert peak_vram([]) == 0.0


def test_peak_vram_ignores_failed_runs():
    results = [
        {"error": "timeout", "gpu": {"peak_delta_mib": 99999.0}},
        {"error": None, "gpu": {"peak_delta_mib": 1000.0}},
    ]
    assert peak_vram(results) == 1000.0


def test_probe_prompt_grows_with_the_requested_length():
    short = build_probe_prompt(100)
    long = build_probe_prompt(4000)
    assert len(long) > len(short) * 10


def test_probe_prompt_is_stable_within_a_run():
    """Samples in one run stay comparable with each other."""
    assert build_probe_prompt(1000, "a") == build_probe_prompt(1000, "a")


def test_probe_prompts_differ_per_sample_so_nothing_is_cached():
    """Reused text is served from Ollama's prompt cache and measures the cache."""
    assert build_probe_prompt(1000, "a") != build_probe_prompt(1000, "b")


def test_probe_prompt_body_is_not_repeated_filler():
    """A repeating body is partially reusable; unique word soup is not."""
    text = build_probe_prompt(2000, "x")
    words = text.split()
    assert len(set(words)) > len(words) * 0.7
