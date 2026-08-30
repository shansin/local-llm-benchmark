"""Reference tests for the date-overlap repair task.

Every case here is one where an off-by-one is the difference between a right
and a wrong answer, including the single-day overlap that the original reports
as no overlap at all.
"""

from datetime import date

import pytest

from candidate import overlap_days


def d(day: int) -> date:
    return date(2025, 3, day)


def test_partial_overlap():
    assert overlap_days(d(1), d(5), d(3), d(9)) == 3


def test_identical_ranges():
    assert overlap_days(d(1), d(5), d(1), d(5)) == 5


def test_one_range_inside_the_other():
    assert overlap_days(d(1), d(10), d(4), d(6)) == 3


def test_overlap_of_exactly_one_day():
    """The original returns 0 here, which reads as no overlap at all."""
    assert overlap_days(d(1), d(5), d(5), d(9)) == 1


def test_single_day_ranges_that_coincide():
    assert overlap_days(d(7), d(7), d(7), d(7)) == 1


def test_adjacent_ranges_do_not_overlap():
    assert overlap_days(d(1), d(4), d(5), d(9)) == 0


def test_disjoint_ranges():
    assert overlap_days(d(1), d(3), d(20), d(25)) == 0


def test_disjoint_ranges_in_the_other_order():
    assert overlap_days(d(20), d(25), d(1), d(3)) == 0


def test_reversed_range_is_rejected():
    with pytest.raises(ValueError):
        overlap_days(d(5), d(1), d(1), d(5))
    with pytest.raises(ValueError):
        overlap_days(d(1), d(5), d(9), d(2))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
