"""Tests for the comparisons against the fixed-period frontier."""
import numpy as np
import pytest

from analysis.pareto import frontier_ticks_at, frontier_value_at, matched_error_saving, matched_ticks_gain, non_dominated

# A fixed-period curve: more rounds -> less error (as in the experiments).
TICKS = np.array([4000.0, 5000.0, 6700.0, 10000.0, 19900.0])
ERROR = np.array([40.0, 33.0, 25.0, 16.0, 8.0])


def test_frontier_value_interpolates_linearly_in_ticks() -> None:
    assert frontier_value_at(TICKS, ERROR, 8350.0) == pytest.approx((25.0 + 16.0) / 2)


def test_frontier_value_accepts_unsorted_reference() -> None:
    order = [3, 0, 4, 1, 2]
    assert frontier_value_at(TICKS[order], ERROR[order], 4500.0) == pytest.approx(36.5)


def test_frontier_value_does_not_extrapolate() -> None:
    assert np.isnan(frontier_value_at(TICKS, ERROR, 3000.0))
    assert np.isnan(frontier_value_at(TICKS, ERROR, 25000.0))


def test_frontier_ticks_inverts_the_curve() -> None:
    assert frontier_ticks_at(TICKS, ERROR, 20.5) == pytest.approx(8350.0)


def test_frontier_ticks_is_nan_beyond_the_best_baseline() -> None:
    assert np.isnan(frontier_ticks_at(TICKS, ERROR, 6.0))


def test_matched_ticks_gain_is_relative_reduction_against_frontier() -> None:
    # at 8350 ticks the frontier error is 20.5 -> 10.25 halves it
    assert matched_ticks_gain(8350.0, 10.25, TICKS, ERROR) == pytest.approx(0.5)


def test_matched_error_saving_is_fraction_of_rounds_saved() -> None:
    # error 20.5 costs 8350 ticks with fixed periods; 4175 ticks saves 50 %
    assert matched_error_saving(4175.0, 20.5, TICKS, ERROR) == pytest.approx(0.5)


def test_non_dominated_minimises_both_objectives() -> None:
    x = np.array([1.0, 2.0, 3.0, 2.5])
    y = np.array([5.0, 3.0, 1.0, 4.0])
    assert non_dominated(x, y).tolist() == [True, True, True, False]
