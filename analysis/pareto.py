"""Comparisons against the fixed-period frontier.

The fixed-period baselines (1..5 s) trace a cost/quality curve: rounds per
episode (x, energy proxy) against error or unstable time (y). Linear
interpolation between consecutive periods gives the frontier a fixed schedule
can reach; no extrapolation is done beyond the 1 s and 5 s periods.
"""
from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray | list[float]


def _sorted_by_ticks(ticks: ArrayLike, values: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    ticks, values = np.asarray(ticks, dtype=float), np.asarray(values, dtype=float)
    order = np.argsort(ticks)
    return ticks[order], values[order]


def frontier_value_at(ref_ticks: ArrayLike, ref_values: ArrayLike, ticks: float) -> float:
    """Frontier quality at the same number of rounds (NaN outside the baseline range)."""
    x, y = _sorted_by_ticks(ref_ticks, ref_values)
    if not x[0] <= ticks <= x[-1]:
        return float("nan")
    return float(np.interp(ticks, x, y))


def frontier_ticks_at(ref_ticks: ArrayLike, ref_values: ArrayLike, value: float) -> float:
    """Rounds a fixed schedule needs to reach ``value`` (NaN outside the baseline range).

    The frontier is monotone: more rounds give lower error / unstable time.
    """
    x, y = _sorted_by_ticks(ref_ticks, ref_values)
    if not y.min() <= value <= y.max():
        return float("nan")
    decreasing = np.argsort(y)  # np.interp needs increasing abscissae
    return float(np.interp(value, y[decreasing], x[decreasing]))


def matched_ticks_gain(ticks: float, value: float, ref_ticks: ArrayLike, ref_values: ArrayLike) -> float:
    """Relative reduction of ``value`` w.r.t. the frontier at the same rounds."""
    return 1.0 - value / frontier_value_at(ref_ticks, ref_values, ticks)


def matched_error_saving(ticks: float, value: float, ref_ticks: ArrayLike, ref_values: ArrayLike) -> float:
    """Fraction of rounds saved w.r.t. a fixed schedule reaching the same ``value``."""
    return 1.0 - ticks / frontier_ticks_at(ref_ticks, ref_values, value)


def non_dominated(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Mask of the points not dominated by any other (both objectives minimised)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        dominated_by = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        mask[i] = not dominated_by.any()
    return mask
