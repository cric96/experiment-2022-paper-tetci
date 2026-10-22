"""Tests for confidence intervals, significance tests and effect sizes."""
import numpy as np
import pandas as pd
import pytest

from analysis.stats import (
    bootstrap_ci,
    bootstrap_two_sample,
    cliffs_delta,
    compare,
    hedges_g,
    holm,
    relative_change,
    summarize,
    t_interval,
)


def test_t_interval_matches_student_t_with_n_minus_one_dof() -> None:
    ci = t_interval([1.0, 2.0, 3.0, 4.0])
    assert ci.mean == pytest.approx(2.5)
    assert ci.n == 4
    # half width = t(0.975, 3) * s / sqrt(n) = 3.182446 * 1.290994 / 2
    assert ci.low == pytest.approx(2.5 - 2.054260, abs=1e-5)
    assert ci.high == pytest.approx(2.5 + 2.054260, abs=1e-5)
    assert ci.half_width == pytest.approx(2.054260, abs=1e-5)


def test_t_interval_of_a_single_value_has_no_bounds() -> None:
    ci = t_interval([3.0])
    assert ci.mean == 3.0 and ci.n == 1
    assert np.isnan(ci.low) and np.isnan(ci.high)


def test_t_interval_ignores_nan() -> None:
    assert t_interval([1.0, np.nan, 3.0]).n == 2


def test_bootstrap_ci_of_constant_sample_is_degenerate() -> None:
    ci = bootstrap_ci([2.0, 2.0, 2.0, 2.0])
    assert (ci.mean, ci.low, ci.high) == (2.0, 2.0, 2.0)


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible() -> None:
    sample = np.arange(10.0)
    first, second = bootstrap_ci(sample), bootstrap_ci(sample)
    assert first.low < first.mean < first.high
    assert (first.low, first.high) == (second.low, second.high)
    assert 1.5 < first.low < 4.5 and 4.5 < first.high < 7.5


def test_bootstrap_with_undefined_statistic_has_no_bounds() -> None:
    ci = bootstrap_two_sample([1.0, 2.0], [3.0, 4.0], lambda a, b: float("nan"))
    assert np.isnan(ci.low) and np.isnan(ci.high)


def test_relative_change_is_fraction_saved_against_reference() -> None:
    assert relative_change([50.0, 50.0], [100.0, 100.0]) == pytest.approx(0.5)
    assert relative_change([150.0], [100.0]) == pytest.approx(-0.5)


def test_bootstrap_two_sample_resamples_both_groups() -> None:
    ci = bootstrap_two_sample([50.0, 50.0, 50.0], [100.0, 100.0, 100.0], relative_change)
    assert (ci.mean, ci.low, ci.high) == pytest.approx((0.5, 0.5, 0.5))
    spread = bootstrap_two_sample([40.0, 60.0, 50.0], [90.0, 110.0, 100.0], relative_change)
    assert spread.low < 0.5 < spread.high


def test_holm_step_down_adjustment() -> None:
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_keeps_nan_and_caps_at_one() -> None:
    adjusted = holm([0.6, np.nan, 0.7])
    assert np.isnan(adjusted[1])
    assert adjusted[0] == pytest.approx(1.0) and adjusted[2] == pytest.approx(1.0)


def test_hedges_g_is_bias_corrected_cohens_d() -> None:
    g = hedges_g([1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0])
    # d = -2 / 1.290994 = -1.549193 ; J = 1 - 3 / (4 * 8 - 9)
    assert g == pytest.approx(-1.549193 * (1 - 3 / 23), abs=1e-5)


def test_hedges_g_is_nan_without_variance() -> None:
    assert np.isnan(hedges_g([1.0, 1.0], [2.0, 2.0]))


def test_cliffs_delta_counts_dominance() -> None:
    assert cliffs_delta([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == -1.0
    assert cliffs_delta([4.0, 5.0], [1.0, 2.0]) == 1.0
    assert cliffs_delta([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_compare_reports_difference_tests_and_effect_sizes() -> None:
    result = compare([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
    assert result.difference == pytest.approx(-4.0)
    assert result.n_a == 4 and result.n_b == 4
    assert result.mwu_p == pytest.approx(2 / 70)  # exact, complete separation
    assert 0.0 < result.welch_p < 0.01
    assert result.cliffs_delta == -1.0
    assert result.hedges_g < -2.0


def test_compare_against_constant_reference_still_tests() -> None:
    result = compare([4.0, 5.0, 6.0, 5.5], [10.0, 10.0, 10.0, 10.0])
    assert result.welch_p < 0.01


def test_summarize_groups_and_reports_n() -> None:
    frame = pd.DataFrame({"m": ["a", "a", "a", "b", "b"], "v": [1.0, 2.0, 3.0, 5.0, 7.0]})
    table = summarize(frame, ["m"], "v")
    assert table.set_index("m").loc["a", "mean"] == pytest.approx(2.0)
    assert table.set_index("m").loc["b", "n"] == 2
    assert {"low", "high", "sd"}.issubset(table.columns)
