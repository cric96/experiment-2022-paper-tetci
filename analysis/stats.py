"""Confidence intervals, significance tests and effect sizes.

Only four seeds per configuration exist, so:

* means are reported with Student-t 95 % intervals over the per-seed values
  (a percentile bootstrap of 4 values has only 35 distinct resamples and
  badly under-covers);
* ratios such as "% rounds saved" use a two-sample percentile bootstrap,
  resampling the seeds of both groups independently;
* group comparisons report Welch's t-test (primary), the exact Mann-Whitney U
  test (robustness check; with 4 vs 4 seeds its smallest two-sided p is
  2/70 = 0.029), Hedges' g and Cliff's delta.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config

ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2.0


@dataclass(frozen=True)
class Comparison:
    difference: float  # mean(a) - mean(b)
    welch_p: float
    mwu_p: float
    hedges_g: float
    cliffs_delta: float
    n_a: int
    n_b: int


def _clean(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[~np.isnan(array)]


def t_interval(values: ArrayLike, confidence: float = config.CONFIDENCE) -> Interval:
    """Mean with a Student-t confidence interval (NaN bounds when n < 2)."""
    sample = _clean(values)
    n = len(sample)
    mean = float(np.mean(sample)) if n else float("nan")
    if n < 2:
        return Interval(mean, float("nan"), float("nan"), n)
    half = sps.t.ppf(0.5 + confidence / 2.0, df=n - 1) * np.std(sample, ddof=1) / np.sqrt(n)
    return Interval(mean, mean - half, mean + half, n)


def _percentile_interval(estimate: float, replicates: np.ndarray, confidence: float, n: int) -> Interval:
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.nanpercentile(replicates, [tail, 100.0 - tail])
    return Interval(estimate, float(low), float(high), n)


def bootstrap_ci(
    values: ArrayLike,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = config.CONFIDENCE,
    n_resamples: int = config.BOOTSTRAP_RESAMPLES,
    seed: int = config.RANDOM_SEED,
) -> Interval:
    """Percentile-bootstrap interval of ``statistic`` (reproducible)."""
    sample = _clean(values)
    rng = np.random.default_rng(seed)
    draws = sample[rng.integers(0, len(sample), size=(n_resamples, len(sample)))]
    replicates = np.apply_along_axis(statistic, 1, draws)
    return _percentile_interval(float(statistic(sample)), replicates, confidence, len(sample))


def bootstrap_two_sample(
    a: ArrayLike,
    b: ArrayLike,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    confidence: float = config.CONFIDENCE,
    n_resamples: int = config.BOOTSTRAP_RESAMPLES,
    seed: int = config.RANDOM_SEED,
) -> Interval:
    """Percentile-bootstrap interval of ``statistic(a, b)`` resampling both groups."""
    sample_a, sample_b = _clean(a), _clean(b)
    rng = np.random.default_rng(seed)
    index_a = rng.integers(0, len(sample_a), size=(n_resamples, len(sample_a)))
    index_b = rng.integers(0, len(sample_b), size=(n_resamples, len(sample_b)))
    replicates = np.array([statistic(sample_a[i], sample_b[j]) for i, j in zip(index_a, index_b)])
    return _percentile_interval(float(statistic(sample_a, sample_b)), replicates, confidence, len(sample_a))


def relative_change(treatment: ArrayLike, reference: ArrayLike) -> float:
    """Fraction saved by ``treatment`` w.r.t. ``reference``: 1 - mean(t) / mean(r)."""
    return 1.0 - float(np.mean(treatment)) / float(np.mean(reference))


def holm(pvalues: ArrayLike) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN entries are left out and kept)."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def hedges_g(a: ArrayLike, b: ArrayLike) -> float:
    """Standardised mean difference with the small-sample correction."""
    x, y = _clean(a), _clean(b)
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled == 0.0:
        return float("nan")
    correction = 1.0 - 3.0 / (4.0 * (nx + ny) - 9.0)
    return float((np.mean(x) - np.mean(y)) / pooled * correction)


def cliffs_delta(a: ArrayLike, b: ArrayLike) -> float:
    """P(a > b) - P(a < b) over all pairs (non-parametric effect size)."""
    x, y = _clean(a)[:, None], _clean(b)[None, :]
    return float((np.sum(x > y) - np.sum(x < y)) / (x.size * y.size))


def _welch_p(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided Welch t-test (written out: scipy warns on a constant group)."""
    se_x, se_y = np.var(x, ddof=1) / len(x), np.var(y, ddof=1) / len(y)
    if se_x + se_y == 0.0:
        return 0.0 if np.mean(x) != np.mean(y) else 1.0
    t = (np.mean(x) - np.mean(y)) / np.sqrt(se_x + se_y)
    dof = (se_x + se_y) ** 2 / (se_x**2 / (len(x) - 1) + se_y**2 / (len(y) - 1))
    return float(2.0 * sps.t.sf(abs(t), dof))


def compare(a: ArrayLike, b: ArrayLike) -> Comparison:
    """Two-sided comparison of two independent samples (e.g. per-seed values)."""
    x, y = _clean(a), _clean(b)
    mwu = sps.mannwhitneyu(x, y, alternative="two-sided", method="auto").pvalue
    return Comparison(
        difference=float(np.mean(x) - np.mean(y)),
        welch_p=_welch_p(x, y),
        mwu_p=float(mwu),
        hedges_g=hedges_g(x, y),
        cliffs_delta=cliffs_delta(x, y),
        n_a=len(x),
        n_b=len(y),
    )


def summarize(frame: pd.DataFrame, by: Sequence[str], value: str, confidence: float = config.CONFIDENCE) -> pd.DataFrame:
    """Per-group mean, t-interval bounds, standard deviation and n of ``value``."""
    rows = []
    for key, group in frame.groupby(list(by), dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        ci = t_interval(group[value], confidence)
        sd = float(np.std(_clean(group[value]), ddof=1)) if ci.n > 1 else float("nan")
        rows.append({**dict(zip(by, key)), "mean": ci.mean, "low": ci.low, "high": ci.high, "sd": sd, "n": ci.n})
    return pd.DataFrame(rows)
