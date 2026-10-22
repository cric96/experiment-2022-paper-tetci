"""Tidy tables built from the raw Alchemist runs.

Two granularities are used throughout the analysis:

* episode tables -- one row per (run, episode) with the per-episode metrics;
* final-episode traces -- the 2 s time series of the last episode, used for the
  intra-episode dynamics around the source switch.

Metric names (Alchemist column -> tidy name):
    ticks[sum]                     -> ticks     rounds executed by all nodes in the episode (energy proxy)
    accumulatedReward[mean]        -> reward    per-node return of the episode
    accumulatedUnstableTime[mean]  -> unstable  per-node time (s) spent with a changing output
    accumulatedError[mean]         -> error     per-node time-integral of the normalised error (s)
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from . import config
from .io import AlchemistRun
from .methods import EXPERIMENT_TO_METHOD, FIXED, GENERALIZATION_ALGORITHMS

METRIC_COLUMNS = {
    "ticks[sum]": "ticks",
    "accumulatedReward[mean]": "reward",
    "accumulatedUnstableTime[mean]": "unstable",
    "accumulatedError[mean]": "error",
}
METRICS = tuple(METRIC_COLUMNS.values())
RUN_KEYS = ["method", "weight", "period", "seed"]


def _episode_rows(run: AlchemistRun) -> pd.DataFrame:
    """Per-episode metrics of one run, indexed by the episode number (1-based)."""
    frame = run.data.rename(columns=METRIC_COLUMNS)
    episode = np.rint(frame["time"] / config.EPISODE_LENGTH_S).astype(int)
    frame = frame.assign(episode=episode)
    frame = frame[(frame.episode >= 1) & (frame.episode <= config.N_EPISODES)]
    return frame[["episode", *METRICS]].reset_index(drop=True)


def episode_table(runs: Iterable[AlchemistRun]) -> pd.DataFrame:
    """Episode table of the main experiment: method, weight, period, seed, episode, metrics."""
    frames = []
    for run in runs:
        method = EXPERIMENT_TO_METHOD.get(run.experiment)
        if method is None:
            continue
        rows = _episode_rows(run)
        rows.insert(0, "method", method)
        rows.insert(1, "weight", run.param("weight"))
        rows.insert(2, "period", run.param("actions") if method == FIXED.key else np.nan)
        rows.insert(3, "seed", int(run.param("random")))
        frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*RUN_KEYS, "episode", *METRICS])


def evaluation_summary(
    episodes: pd.DataFrame,
    eval_episodes: Sequence[int] = config.EVAL_EPISODES,
    keys: Sequence[str] = RUN_KEYS,
) -> pd.DataFrame:
    """Mean of each metric over the evaluation episodes, one row per run (seed)."""
    evaluation = episodes[episodes.episode.isin(eval_episodes)]
    return evaluation.groupby(list(keys), dropna=False)[list(METRICS)].mean().reset_index()


def generalization_beta(episode: int) -> float:
    """beta_in given to beta-conditioned policies in a generalization episode (NaN while training)."""
    if episode <= config.TRAINING_EPISODES:
        return float("nan")
    index = (episode - config.TRAINING_EPISODES - 1) // config.EPISODES_PER_GENERALIZATION_BETA
    return config.WEIGHTS[min(index, len(config.WEIGHTS) - 1)]


def generalization_table(runs: Iterable[AlchemistRun]) -> pd.DataFrame:
    """Episode table of the generalization experiment.

    Columns: method, period, program, seed, episode, beta_in, metrics.
    """
    frames = []
    for run in runs:
        method, period = GENERALIZATION_ALGORITHMS[int(run.param("algoId"))]
        rows = _episode_rows(run)
        rows.insert(0, "method", method)
        rows.insert(1, "period", np.nan if period is None else period)
        rows.insert(2, "program", config.PROGRAMS[int(run.param("progId"))])
        rows.insert(3, "seed", int(run.param("random")))
        rows.insert(5, "beta_in", [generalization_beta(e) for e in rows.episode])
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def final_episode_trace(run: AlchemistRun) -> pd.DataFrame:
    """Time series of the last episode, with time relative to the episode start.

    ``deviation`` is |mean(g) - mean(g*)| over the network: the exports only
    contain sums of the gradient and of the ground truth, so this is the
    deviation of the *network-mean* gradient (not the mean per-node error).
    """
    data = run.data
    start = config.FINAL_EPISODE_START_S
    window = data[(data.time > start) & (data.time <= start + config.EPISODE_LENGTH_S + 0.5)]
    gradient_mean = window["localComputation[sum]"].to_numpy() / config.N_NODES
    truth_mean = window["groundTruth[sum]"].to_numpy() / config.N_NODES
    return pd.DataFrame(
        {
            "t": window.time.to_numpy() - start,
            "wakeup": window["nextWakeUp[mean]"].to_numpy(),
            "ticks": window["ticks[sum]"].to_numpy(),
            "gradient_mean": gradient_mean,
            "truth_mean": truth_mean,
            "deviation": np.abs(gradient_mean - truth_mean),
        }
    )


def resample(traces: Sequence[pd.DataFrame], column: str, grid: np.ndarray, time: str = "t") -> np.ndarray:
    """Interpolate ``column`` of every trace on ``grid``: one row per trace."""
    return np.vstack([np.interp(grid, trace[time].to_numpy(), trace[column].to_numpy()) for trace in traces])


def final_episode_traces(runs: Iterable[AlchemistRun]) -> pd.DataFrame:
    """Long table of the final-episode traces of the main experiment (all runs)."""
    frames = []
    for run in runs:
        method = EXPERIMENT_TO_METHOD.get(run.experiment)
        if method is None:
            continue
        trace = final_episode_trace(run)
        trace.insert(0, "method", method)
        trace.insert(1, "weight", run.param("weight"))
        trace.insert(2, "period", run.param("actions") if method == FIXED.key else np.nan)
        trace.insert(3, "seed", int(run.param("random")))
        frames.append(trace)
    return pd.concat(frames, ignore_index=True)


def select(frame: pd.DataFrame, method: str, weight: float | None = None, period: float | None = None) -> pd.DataFrame:
    """Rows of one configuration (``weight``/``period`` matched with a tolerance)."""
    mask = frame.method == method
    if weight is not None:
        mask &= np.isclose(frame.weight, weight)
    if period is not None:
        mask &= np.isclose(frame.period, period)
    return frame[mask]


def traces_by_seed(traces: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a long trace table into one frame per seed."""
    return [group for _, group in traces.groupby("seed", sort=True)]

