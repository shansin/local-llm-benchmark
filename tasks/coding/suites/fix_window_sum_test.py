"""Reference tests for the max_window_sum repair task.

The bug is invisible on the inputs a model is most likely to try by hand
(small positive lists), so the tests lean on the cases that expose it: windows
of all-negative values, and a maximum that falls in the very first window.
"""

import pytest

from candidate import max_window_sum


def test_the_obvious_case_still_works():
    assert max_window_sum([1, 2, 3, 4, 5], 2) == 9


def test_maximum_window_in_the_middle():
    assert max_window_sum([1, 9, 9, 1, 1], 2) == 18


def test_all_negative_values():
    """The original returns 0 here, which is not the sum of any window."""
    assert max_window_sum([-5, -2, -8, -1], 2) == -7


def test_maximum_window_is_the_first_one():
    """The original never compares the first window against the best."""
    assert max_window_sum([10, 10, 1, 1, 1], 2) == 20


def test_single_element_windows():
    assert max_window_sum([-3, -1, -7], 1) == -1


def test_window_spanning_the_whole_list():
    assert max_window_sum([2, -4, 6], 3) == 4


def test_mixed_signs():
    assert max_window_sum([3, -1, -1, 3, 3], 3) == 5


def test_k_of_zero_is_rejected():
    with pytest.raises(ValueError):
        max_window_sum([1, 2, 3], 0)


def test_k_larger_than_the_list_is_rejected():
    with pytest.raises(ValueError):
        max_window_sum([1, 2, 3], 4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
