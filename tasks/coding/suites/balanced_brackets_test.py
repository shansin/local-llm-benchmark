"""Reference tests for the is_balanced task."""

import pytest

from candidate import is_balanced


def test_examples_from_the_prompt():
    assert is_balanced("(a[b]{c})") is True
    assert is_balanced("([)]") is False
    assert is_balanced("(") is False


def test_empty_string_is_balanced():
    assert is_balanced("") is True


def test_simple_pairs():
    assert is_balanced("()") is True
    assert is_balanced("[]") is True
    assert is_balanced("{}") is True


def test_nested_pairs():
    assert is_balanced("({[]})") is True


def test_sequential_pairs():
    assert is_balanced("()[]{}") is True


def test_unopened_bracket():
    assert is_balanced(")") is False
    assert is_balanced("()) ") is False


def test_non_bracket_characters_are_ignored():
    assert is_balanced("hello world") is True
    assert is_balanced("if (x) { return y[0]; }") is True


def test_wrong_closing_type():
    assert is_balanced("(]") is False


def test_counts_alone_are_not_enough():
    """Equal counts in the wrong order must still fail."""
    assert is_balanced("[(])") is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
