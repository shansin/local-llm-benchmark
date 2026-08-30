import json

from llmbench.report.compare import (
    comparability,
    compare_documents,
    leaderboard,
    noise_floor,
    render_comparison,
)
from llmbench.report.html import render, write_html
from llmbench.report.jsonout import build_document, load_json, write_json
from llmbench.tasks import Task


def task(task_id="t1", category="coding", checks=None):
    return Task(id=task_id, category=category, prompt="p", criteria="c", checks=checks or [])


def gen(tps=100.0):
    return {
        "response": "answer",
        "tokens_per_sec": tps,
        "ttft": 0.2,
        "prefill_time": 0.1,
        "load_time": 0.0,
        "total_time": 3.0,
        "eval_count": 300,
        "prompt_eval_speed": 1500.0,
        "prompt_eval_count": 150,
        "seed": 0,
        "error": None,
        "gpu": None,
    }


def document(name="run-a", score=9.0, tps=100.0):
    tasks = [task("t1"), task("t2", "writing")]
    return build_document(
        name,
        all_results={"m1": {"t1": [gen(tps)], "t2": [gen(tps)]}},
        all_details={"m1": {"parameter_size": "8B", "quantization_level": "Q4_K_M"}},
        judge_scores={
            "m1": {
                "t1": [{"score": score, "reason": "ok"}],
                "t2": [{"score": score, "reason": "ok"}],
            }
        },
        judge_models=["judge1"],
        tasks=tasks,
        perf_results={"m1": [gen(tps), gen(tps)]},
        cold_loads={"m1": 4.0},
        gen_params={"temperature": 0.0, "seed": 0},
        total_runtime=120.0,
        model_vram={"m1": {"vram_mib": 5000.0}},
    )


# ---------- json ----------


def test_document_carries_summaries_and_the_raw_records_behind_them():
    doc = document()
    model = doc["models"][0]
    assert model["overall"]["mean"] == 9.0
    assert model["tasks"]["t1"]["repeats"][0]["tokens_per_sec"] == 100.0
    assert model["tasks"]["t1"]["judge_verdicts"][0]["score"] == 9.0


def test_document_records_the_config_needed_to_reproduce_the_run():
    config = document()["config"]
    assert config["generation"]["temperature"] == 0.0
    assert config["judges"] == ["judge1"]
    # The elicitation protocol is fixed, and recorded so old runs stay legible.
    assert config["answer_tags"] is True


def test_document_is_json_serialisable(tmp_path):
    write_json(tmp_path, document())
    reloaded = load_json(tmp_path)
    assert reloaded["models"][0]["name"] == "m1"


def test_load_json_returns_none_for_older_runs(tmp_path):
    assert load_json(tmp_path) is None


def test_blended_score_falls_back_to_the_judge_without_checks():
    doc = document(score=8.0)
    assert doc["models"][0]["tasks"]["t1"]["blended"]["mean"] == 8.0


def test_category_breakdown_is_present():
    doc = document()
    assert set(doc["models"][0]["by_category"]) == {"coding", "writing"}


# ---------- html ----------


def test_html_is_self_contained():
    page = render(document())
    assert "<style>" in page
    for forbidden in ("http://", "https://", "<script", "cdn"):
        assert forbidden not in page.lower()


def test_html_handles_a_single_model_without_a_scatter():
    """Two points are needed for a scatter; one model must still render."""
    page = render(document())
    assert "<h1>" in page
    assert "Summary" in page


def test_html_escapes_model_names():
    doc = document()
    doc["models"][0]["name"] = "<script>alert(1)</script>"
    assert "<script>alert(1)</script>" not in render(doc)
    assert "&lt;script&gt;" in render(doc)


def test_html_notes_that_single_samples_make_small_gaps_ties():
    assert "sampled once" in render(document())


def test_html_written_to_disk(tmp_path):
    write_html(tmp_path, document())
    assert (tmp_path / "report.html").read_text().startswith("<!doctype html>")


def test_html_survives_a_model_that_was_never_scored():
    doc = document()
    doc["models"][0]["overall"] = {"mean": None, "repeat_std": 0.0, "n_scored": 0, "n_total": 2}
    doc["models"][0]["by_category"] = {"coding": {"mean": None}, "writing": {"mean": None}}
    assert "—" in render(doc)


# ---------- compare ----------


def test_improvement_beyond_the_noise_floor_is_reported():
    before = document("a", score=7.0)
    after = document("b", score=9.0)
    diff = compare_documents(before, after)
    row = diff["models"][0]
    assert row["delta"] == 2.0
    assert row["verdict"] == "improved"


def test_regression_is_reported():
    diff = compare_documents(document("a", score=9.0), document("b", score=6.0))
    assert diff["models"][0]["verdict"] == "regressed"


def test_change_within_the_noise_floor_is_called_unchanged():
    """The whole point of measuring noise is to not report it as a result."""
    before = document("a", score=9.0)
    after = document("b", score=9.05)
    # Give the runs a noise floor wider than the difference.
    for doc in (before, after):
        doc["models"][0]["overall"]["repeat_std"] = 0.5
    diff = compare_documents(before, after)
    assert diff["models"][0]["verdict"] == "unchanged"
    assert diff["models"][0]["tasks"] == []


def test_throughput_change_is_reported_as_a_percentage():
    diff = compare_documents(document("a", tps=100.0), document("b", tps=150.0))
    assert diff["models"][0]["tps_delta_pct"] == 50.0


def test_added_and_removed_models_are_flagged():
    before = document("a")
    after = document("b")
    after["models"][0]["name"] = "m2"
    statuses = {r["model"]: r["status"] for r in compare_documents(before, after)["models"]}
    assert statuses == {"m1": "removed", "m2": "added"}


def test_noise_floor_reads_the_widest_spread():
    doc = document()
    doc["models"][0]["overall"]["repeat_std"] = 0.42
    assert noise_floor(doc) == 0.42


def test_rendered_comparison_flags_missing_noise_estimates():
    text = render_comparison(compare_documents(document("a"), document("b")))
    assert "no noise estimate" in text.lower()


def test_rendered_comparison_states_the_threshold_when_known():
    before, after = document("a", score=7.0), document("b", score=9.0)
    for doc in (before, after):
        doc["models"][0]["overall"]["repeat_std"] = 0.3
    text = render_comparison(compare_documents(before, after))
    assert "0.30 points or less" in text


def test_leaderboard_keeps_the_best_score_per_model():
    runs = [document("a", score=6.0), document("b", score=9.0), document("c", score=7.0)]
    text = leaderboard(runs)
    assert "9.00" in text
    assert "| b |" in text or "b |" in text


def test_leaderboard_over_a_single_run():
    assert "m1" in leaderboard([document()])


def test_comparison_json_round_trips():
    diff = compare_documents(document("a"), document("b"))
    assert json.loads(json.dumps(diff))["before"] == "a"


# ---------- comparability of runs ----------


def test_a_changed_judge_panel_is_reported_before_any_delta():
    """Swapping the judge moves every score at once; a delta then means nothing."""
    before = document("a", score=8.7)
    after = document("b", score=8.2)
    after["config"]["judges"] = ["a-different-judge"]
    diff = compare_documents(before, after)
    assert any("judge panel" in note for note in diff["comparability"])
    rendered = render_comparison(diff)
    assert "not measured the same way" in rendered


def test_a_changed_context_window_is_reported():
    before = document("a")
    after = document("b")
    before["config"]["generation"] = {"num_ctx": 8192}
    after["config"]["generation"] = {"num_ctx": 32768}
    assert any("num_ctx" in n for n in comparability(before, after))


def test_identical_settings_produce_no_comparability_warning():
    assert comparability(document("a"), document("b")) == []
    assert "not measured the same way" not in render_comparison(
        compare_documents(document("a"), document("b"))
    )


def test_a_changed_task_set_is_reported():
    before = document("a")
    after = document("b")
    after["tasks"].append({"id": "t3", "category": "writing"})
    assert any("task set" in n and "t3" in n for n in comparability(before, after))


def test_leaderboard_warns_when_runs_used_different_judges():
    first = document("a", score=8.0)
    second = document("b", score=9.0)
    second["config"]["judges"] = ["someone-else"]
    assert "Mixed judges" in leaderboard([first, second])


def test_leaderboard_is_quiet_when_the_judge_never_changed():
    assert "Mixed judges" not in leaderboard([document("a"), document("b")])


# ---------- completeness ----------


def test_document_records_how_many_generations_were_scorable():
    doc = build_document(
        "run",
        all_results={
            "m1": {"t1": [gen(), {**gen(), "response": "<think>cut off", "truncated": True}]}
        },
        all_details={},
        judge_scores={},
        judge_models=["j"],
        tasks=[task("t1")],
    )
    stats = doc["models"][0]["completeness"]
    assert stats["total"] == 2
    assert stats["truncated"] == 1
    assert stats["empty"] == 1
