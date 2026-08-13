import json

import pytest

from llmbench.state import SCHEMA_VERSION, load_state, save_state
from llmbench.util import format_duration, safe_dirname


def test_state_round_trip(tmp_path):
    state = {"selected": [{"name": "m"}], "all_results": {"m": {"coding": {}}}}
    save_state(tmp_path, state)
    loaded = load_state(tmp_path)
    assert loaded["selected"] == state["selected"]
    assert loaded["schema_version"] == SCHEMA_VERSION


def test_state_absent_returns_none(tmp_path):
    assert load_state(tmp_path) is None


def test_legacy_state_is_migrated_to_repeat_lists(tmp_path):
    """A pre-repeats run is exactly a run with one repeat, so it migrates intact."""
    legacy = {
        "selected": [{"name": "m"}],
        "all_results": {"m": {"coding": {"tokens_per_sec": 5.0, "response": "hi"}}},
        "judge_scores": {"m": {"coding": {"score": 8, "reason": "ok"}}},
    }
    (tmp_path / "state.json").write_text(json.dumps(legacy))

    state = load_state(tmp_path)
    assert state["schema_version"] == SCHEMA_VERSION
    assert state["all_results"]["m"]["coding"] == [{"tokens_per_sec": 5.0, "response": "hi"}]
    assert state["judge_scores"]["m"]["coding"] == [{"score": 8, "reason": "ok"}]


def test_migration_leaves_already_migrated_state_alone(tmp_path):
    save_state(tmp_path, {"all_results": {"m": {"coding": [{"a": 1}, {"a": 2}]}}})
    assert len(load_state(tmp_path)["all_results"]["m"]["coding"]) == 2


def test_state_from_a_newer_tool_version_is_refused(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"schema_version": 999}))
    with pytest.raises(SystemExit):
        load_state(tmp_path)


def test_save_leaves_no_temp_file(tmp_path):
    save_state(tmp_path, {})
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_format_duration():
    assert format_duration(45) == "45s"
    assert format_duration(3661) == "1h 1m 1s"
    assert format_duration(120) == "2m 0s"


def test_safe_dirname():
    assert safe_dirname("qwen3.5:27b") == "qwen3.5_27b"
    assert safe_dirname("hf.co/user/model:q4") == "hf.co_user_model_q4"
