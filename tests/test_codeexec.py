"""Tests for the code-execution sandbox.

These deliberately include hostile inputs — an infinite loop, a memory bomb, an
attempted network call — because the sandbox's containment is the feature.
"""

from pathlib import Path

import pytest

from llmbench.scoring.codeexec import extract_code, run_code_check, run_suite

SUITE = """
from candidate import add


def test_adds():
    assert add(1, 2) == 3


def test_adds_negative():
    assert add(-1, -1) == -2


def test_adds_zero():
    assert add(0, 0) == 0
"""


@pytest.fixture
def suite(tmp_path):
    path = tmp_path / "suite.py"
    path.write_text(SUITE)
    return path


# ---------- extraction ----------


def test_extracts_a_labelled_python_fence():
    response = (
        "Here you go:\n\n```python\ndef add(a, b):\n    return a + b\n```\n\nHope that helps!"
    )
    assert extract_code(response) == "def add(a, b):\n    return a + b"


def test_extracts_an_unlabelled_fence():
    assert "def add" in extract_code("```\ndef add(a, b):\n    return a + b\n```")


def test_extracts_bare_code_with_no_fence():
    assert extract_code("def add(a, b):\n    return a + b") is not None


def test_joins_multiple_python_fences():
    """Solutions are often split into a code block and a tests block."""
    response = "```python\ndef add(a, b):\n    return a + b\n```\ntests:\n```python\nX = 1\n```"
    code = extract_code(response)
    assert "def add" in code
    assert "X = 1" in code


def test_prose_only_response_yields_no_code():
    assert extract_code("I'm sorry, I can't help with that.") is None


def test_syntactically_invalid_code_yields_none():
    assert extract_code("```python\ndef add(a, b:\n    return\n```") is None


def test_prose_around_a_broken_fence_does_not_parse_as_code():
    assert extract_code("Sure! Here is the answer, which is 42.") is None


# ---------- execution ----------


def test_correct_code_passes_every_test(suite):
    result = run_code_check("```python\ndef add(a, b):\n    return a + b\n```", suite)
    assert result.passed == 3
    assert result.total == 3
    assert result.fraction == 1.0


def test_partially_correct_code_gets_partial_credit(suite):
    """Graded credit is the point — a near miss is not the same as no answer."""
    code = "```python\ndef add(a, b):\n    return abs(a) + abs(b)\n```"
    result = run_code_check(code, suite)
    assert 0 < result.passed < result.total
    assert "first failure" in result.detail


def test_wrong_function_name_scores_zero(suite):
    result = run_code_check("```python\ndef plus(a, b):\n    return a + b\n```", suite)
    assert result.fraction == 0.0
    assert "could not run" in result.detail


def test_response_without_code_scores_zero(suite):
    result = run_code_check("I cannot write that function.", suite)
    assert result.fraction == 0.0
    assert "no valid Python" in result.detail


def test_code_that_raises_on_import_scores_zero(suite):
    result = run_suite("raise RuntimeError('boom')", suite)
    assert result.fraction == 0.0


# ---------- containment ----------


def test_infinite_loop_is_killed_and_does_not_hang(suite):
    result = run_suite("while True:\n    pass\n", suite, timeout=10)
    assert result.fraction == 0.0
    assert "timed out" in result.detail


def test_memory_bomb_is_contained(suite):
    """RLIMIT_AS must stop this before it exhausts the host's RAM."""
    code = "x = bytearray(8 * 1024 * 1024 * 1024)\n\n\ndef add(a, b):\n    return a + b\n"
    result = run_suite(code, suite, timeout=30)
    assert result.fraction < 1.0


def test_fork_bomb_is_contained(suite):
    code = "import os\n\nwhile True:\n    os.fork()\n"
    result = run_suite(code, suite, timeout=15)
    assert result.fraction == 0.0


def test_candidate_cannot_see_the_projects_files(suite, tmp_path):
    """The sandbox cwd is a fresh temp dir, not the repository."""
    code = (
        "import os\n"
        "assert not os.path.exists('pyproject.toml'), 'sandbox can see the repo'\n\n\n"
        "def add(a, b):\n"
        "    return a + b\n"
    )
    assert run_suite(code, suite).fraction == 1.0


def test_secrets_in_the_environment_are_not_inherited(suite, monkeypatch):
    monkeypatch.setenv("MY_API_TOKEN", "super-secret-value")
    code = (
        "import os\n"
        "assert 'MY_API_TOKEN' not in os.environ, 'environment leaked into the sandbox'\n\n\n"
        "def add(a, b):\n"
        "    return a + b\n"
    )
    assert run_suite(code, suite).fraction == 1.0


def test_writes_land_in_the_temp_dir_not_the_repo(suite):
    code = "open('scratch.txt', 'w').write('x')\n\n\ndef add(a, b):\n    return a + b\n"
    assert run_suite(code, suite).fraction == 1.0
    assert not Path("scratch.txt").exists()
