"""Learning curves: per-episode metrics during training and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from .. import config, style
from ..data import select
from ..dataset import Dataset
from ..methods import FIXED, LEARNED
from ..results import best_fixed_period
from .common import METRIC_LABELS, episode_matrix, method_band, thousands

EPISODES = np.arange(1, config.N_EPISODES + 1)
REWARD_WEIGHTS = (0.5, 0.8, 0.95, 1.0)
METRIC_WEIGHTS = (0.95, 0.8)
REFERENCE_PERIODS = (1.0, 3.0, 5.0)
MARK_EVERY = 3


def _episode_axis(ax) -> None:
    ax.set_xlim(0.5, config.N_EPISODES + 0.5)
    ax.set_xticks([1, 5, 10, 15, 20])


def _fixed_line(ax, data: Dataset, weight: float, period: float, metric: str, label: str) -> None:
    """A fixed period's mean curve in grey with a direct label at the right edge."""
    matrix = episode_matrix(select(data.episodes, FIXED.key, weight=weight, period=period), metric)
    mean = matrix.mean(axis=0)
    color = style.fixed_period_color(period)
    ax.plot(EPISODES, mean, color=color, linestyle="--", linewidth=0.8, zorder=1)
    style.direct_label(ax, EPISODES[-1], mean[-1], label, dx=2, color=style.INK_SECONDARY, fontsize=5.5)


def learning_curves_reward(data: Dataset, out: Path) -> None:
    """Return per episode for four betas, against the best fixed period of each beta."""
    fig, axes = style.figure("double", 1.95, 1, len(REWARD_WEIGHTS))
    for index, (ax, weight) in enumerate(zip(axes, REWARD_WEIGHTS)):
        style.shade_evaluation(ax, label=index == 0)
        for method in LEARNED:
            matrix = episode_matrix(select(data.episodes, method.key, weight=weight), "reward")
            method_band(ax, EPISODES, matrix, method, markevery=MARK_EVERY)
        period = best_fixed_period(data.evaluation, weight)
        _fixed_line(ax, data, weight, period, "reward", f"{period:g} s")
        ax.set_title(f"$\\beta$ = {weight:g}")
        ax.set_xlabel("Episode")
        _episode_axis(ax)
    axes[0].set_ylabel(METRIC_LABELS["reward"])
    handles = [style.legend_handle(m) for m in LEARNED]
    handles.append(
        Line2D([], [], color=style.fixed_period_color(3.0), linestyle="--", linewidth=0.8, label="Best fixed period")
    )
    style.figure_legend(fig, handles, ncol=len(handles))
    style.save(fig, out, "learning_curves_reward")


def learning_curves_metrics(data: Dataset, out: Path) -> None:
    """Rounds, unstable time and error per episode (two betas), against fixed periods."""
    metrics = ("ticks", "unstable", "error")
    fig, axes = style.figure("double", 3.5, len(METRIC_WEIGHTS), len(metrics), sharex=True)
    for row, weight in enumerate(METRIC_WEIGHTS):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            style.shade_evaluation(ax, label=(row == 0 and col == 0))
            for method in LEARNED:
                matrix = episode_matrix(select(data.episodes, method.key, weight=weight), metric)
                method_band(ax, EPISODES, matrix, method, markevery=MARK_EVERY)
            for period in REFERENCE_PERIODS:
                _fixed_line(ax, data, weight, period, metric, f"{period:g} s")
            ax.set_title(f"$\\beta$ = {weight:g}")
            ax.set_ylabel(METRIC_LABELS[metric])
            if metric == "ticks":
                thousands(ax, "y")
            _episode_axis(ax)
            if row == len(METRIC_WEIGHTS) - 1:
                ax.set_xlabel("Episode")
    handles = [style.legend_handle(m) for m in LEARNED]
    handles.append(
        Line2D(
            [], [], color=style.fixed_period_color(3.0), linestyle="--", linewidth=0.8, label="Fixed period (1, 3, 5 s)"
        )
    )
    style.figure_legend(fig, handles, ncol=len(handles))
    style.save(fig, out, "learning_curves_metrics")


FIGURES = (learning_curves_reward, learning_curves_metrics)
