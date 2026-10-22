"""Tests for the derived results (best baseline, collapse detection, switch response)."""
import numpy as np
import pandas as pd
import pytest

from analysis import config
from analysis.results import best_fixed_period, collapsed_seeds, switch_response


def _fixed_summary() -> pd.DataFrame:
    rows = []
    for weight, rewards in {0.5: {1.0: -78.0, 3.0: -6.0, 5.0: -8.0}, 1.0: {1.0: -1.0, 3.0: -4.0, 5.0: -12.0}}.items():
        for period, reward in rewards.items():
            for seed in (1, 2):
                rows.append({"method": "fixed", "weight": weight, "period": period, "seed": seed, "reward": reward + seed * 0.1})
    return pd.DataFrame(rows)


def test_best_fixed_period_maximises_mean_reward_at_the_weight() -> None:
    summary = _fixed_summary()
    assert best_fixed_period(summary, 0.5) == 3.0
    assert best_fixed_period(summary, 1.0) == 1.0


def test_collapsed_seeds_flags_runs_close_to_the_fastest_period() -> None:
    ticks = np.array([10_500.0, 18_500.0, 10_600.0, 19_800.0])
    assert collapsed_seeds(ticks, fastest_ticks=19_900.0).tolist() == [False, True, False, True]


def _trace() -> pd.DataFrame:
    t = np.arange(2.0, 202.0, 2.0)
    wakeup = np.where((t >= 100) & (t < 120), 1.0, 3.0)
    rate = np.where((t > 100) & (t <= 140), 2.0, 1.0)  # rounds / s in the 2 s before each sample
    ticks = np.cumsum(rate * 2.0)
    deviation = np.where((t >= 100) & (t <= 130), 5.0, 0.0)
    return pd.DataFrame({"t": t, "wakeup": wakeup, "ticks": ticks, "deviation": deviation})


def test_switch_response_measures_adaptation_around_the_switch() -> None:
    response = switch_response(_trace())
    assert response["steady_period"] == pytest.approx(3.0)
    assert response["min_period"] == pytest.approx(1.0)
    assert response["rate_ratio"] == pytest.approx(2.0)
    assert response["recovery_time"] == pytest.approx(132.0 - config.SOURCE_SWITCH_S)


def test_switch_response_without_recovery_is_nan() -> None:
    trace = _trace()
    trace.loc[trace.t > 150, "deviation"] = 5.0
    assert np.isnan(switch_response(trace)["recovery_time"])
