from llmbench.scoring.aggregate import (
    combined_perf,
    model_score_stats,
    perf_summary,
    successful,
    task_score_stats,
)

CATS = ["coding", "reasoning", "writing"]


def gen(tps=10.0, ttft=1.0, prefill=100.0, total=5.0, tokens=50, error=None):
    return {
        "tokens_per_sec": tps,
        "ttft": ttft,
        "prefill_time": 0.5,
        "load_time": 0.0,
        "prompt_eval_speed": prefill,
        "total_time": total,
        "eval_count": tokens,
        "error": error,
    }


def verdicts(*scores):
    return [{"score": s, "reason": "because"} for s in scores]


def test_successful_drops_failures():
    results = [gen(), gen(error="timeout"), gen()]
    assert len(successful(results)) == 2


def test_perf_summary_uses_median_not_mean():
    """One stalled sample must not drag the headline throughput down."""
    results = [gen(tps=100.0), gen(tps=101.0), gen(tps=99.0), gen(tps=1.0)]
    summary = perf_summary(results)
    assert summary["tps_median"] == 99.5
    assert summary["samples"] == 4


def test_perf_summary_excludes_errored_samples():
    summary = perf_summary([gen(tps=50.0), gen(error="timeout")])
    assert summary["tps_median"] == 50.0
    assert summary["failures"] == 1


def test_perf_summary_of_nothing_is_zero_not_a_crash():
    assert perf_summary([])["tps_median"] == 0.0
    assert perf_summary([gen(error="x")])["tps_median"] == 0.0


def test_ttft_percentiles():
    results = [gen(ttft=t) for t in (1.0, 2.0, 3.0, 4.0, 10.0)]
    summary = perf_summary(results)
    assert summary["ttft_p50"] == 3.0
    assert summary["ttft_p90"] > summary["ttft_p50"]


def test_combined_perf_pools_every_category():
    by_cat = {
        "coding": [gen(tokens=10)],
        "reasoning": [gen(tokens=20)],
        "writing": [gen(tokens=30)],
    }
    assert combined_perf(by_cat, CATS)["total_tokens"] == 60


def test_task_score_stats_reports_spread_across_repeats():
    stats = task_score_stats(verdicts(8, 10, 9))
    assert stats["mean"] == 9.0
    assert stats["std"] > 0
    assert stats["n"] == 3


def test_task_score_stats_ignores_unscored_repeats():
    stats = task_score_stats(verdicts(10, None, 8))
    assert stats["mean"] == 9.0
    assert stats["n"] == 2
    assert stats["n_attempted"] == 3


def test_task_never_scored_is_none_not_zero():
    stats = task_score_stats(verdicts(None, None))
    assert stats["mean"] is None


def test_single_repeat_has_zero_spread():
    assert task_score_stats(verdicts(7))["std"] == 0.0


def test_model_score_stats_excludes_unscored_categories():
    scores = {
        "coding": verdicts(10, 10),
        "reasoning": verdicts(None),
        "writing": verdicts(8, 8),
    }
    stats = model_score_stats(scores, CATS)
    assert stats["mean"] == 9.0
    assert stats["n_scored"] == 2
    assert stats["n_total"] == 3


def test_model_score_stats_reports_noise_floor():
    """repeat_std is the within-task spread that model gaps have to clear."""
    scores = {"coding": verdicts(6, 10), "reasoning": verdicts(9, 9), "writing": verdicts(8, 8)}
    stats = model_score_stats(scores, CATS)
    assert stats["repeat_std"] > 0


def test_model_with_no_scores_at_all():
    stats = model_score_stats({}, CATS)
    assert stats["mean"] is None
    assert stats["n_scored"] == 0
