from llmbench.stats import iqr, mean, median, percentile, stdev


def test_empty_inputs_return_zero_rather_than_raising():
    assert mean([]) == 0.0
    assert median([]) == 0.0
    assert stdev([]) == 0.0
    assert percentile([], 50) == 0.0
    assert iqr([]) == 0.0


def test_single_sample_has_no_spread():
    assert stdev([5.0]) == 0.0
    assert iqr([5.0]) == 0.0
    assert percentile([5.0], 90) == 5.0


def test_percentile_interpolates():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 50) == 2.5


def test_percentile_is_order_independent():
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == percentile([1.0, 2.0, 3.0, 4.0], 50)


def test_iqr_is_the_middle_spread():
    assert iqr([1.0, 2.0, 3.0, 4.0, 5.0]) == 2.0
