"""Reference tests for the merge_intervals task.

These are the benchmark's tests, not the model's — a model cannot raise its own
score by writing weak tests. The model's own tests are still judged qualitatively.
"""

import pytest

from candidate import merge_intervals


def norm(result):
    """Accept tuples or lists, since the prompt does not mandate one."""
    return [tuple(item) for item in result]


def test_example_from_the_prompt():
    assert norm(merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)])) == [(1, 6), (8, 10), (15, 18)]


def test_empty_list():
    assert norm(merge_intervals([])) == []


def test_single_interval():
    assert norm(merge_intervals([(1, 5)])) == [(1, 5)]


def test_non_overlapping_stay_separate():
    assert norm(merge_intervals([(1, 2), (5, 6), (9, 10)])) == [(1, 2), (5, 6), (9, 10)]


def test_unsorted_input_is_sorted_and_merged():
    assert norm(merge_intervals([(8, 10), (1, 3), (15, 18), (2, 6)])) == [(1, 6), (8, 10), (15, 18)]


def test_fully_contained_interval_is_absorbed():
    assert norm(merge_intervals([(1, 10), (3, 5)])) == [(1, 10)]


def test_all_intervals_merge_into_one():
    assert norm(merge_intervals([(1, 4), (2, 6), (5, 9), (8, 12)])) == [(1, 12)]


def test_touching_intervals():
    """(1,3) and (3,5) touch at a point — either answer is defensible."""
    result = norm(merge_intervals([(1, 3), (3, 5)]))
    assert result in ([(1, 5)], [(1, 3), (3, 5)])


def test_duplicate_intervals():
    assert norm(merge_intervals([(1, 5), (1, 5), (1, 5)])) == [(1, 5)]


def test_negative_coordinates():
    assert norm(merge_intervals([(-10, -5), (-6, -1), (0, 3)])) == [(-10, -1), (0, 3)]


def test_does_not_mutate_its_input():
    original = [(8, 10), (1, 3), (2, 6)]
    snapshot = list(original)
    merge_intervals(original)
    assert original == snapshot


def test_large_input_completes_quickly():
    """A quadratic solution will time out here; an O(n log n) one will not."""
    intervals = [(i * 3, i * 3 + 1) for i in range(20000)]
    assert len(norm(merge_intervals(intervals))) == 20000


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
