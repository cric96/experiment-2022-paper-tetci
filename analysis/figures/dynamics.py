"""Intra-episode dynamics of the final (evaluation) episode around the source switch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from .. import config, style
from ..data import resample, select, traces_by_seed
from ..dataset import Dataset
from ..methods import FIXED, LEARNED, TD3BC
from .common import mean_band, method_band

ZOOM_S = (80.0, 165.0)  # window around the source switch
ZOOM_GRID = np.arange(ZOOM_S[0], ZOOM_S[1] + 0.5, 1.0)
EPISODE_GRID = np.arange(4.0, config.EPISODE_LENGTH_S + 0.5, 2.0)  # skips the start-up transient
REFERENCE_PERIODS = (3.0, 5.0)  # fixed periods drawn for comparison
SHOWN_SEED = config.SPATIAL_SEED  # seed whose raw mean gradient is drawn
PERIOD_LABEL = "Mean wake-up period (s)"
DEVIATION_LABEL = "$|\\bar g - \\bar g^*|$ (m)"
EPISODE_TIME_LABEL = "Time within the final episode (s)"


def _matrix(traces, column: str, grid: np.ndarray) -> np.ndarray:
    return resample(traces_by_seed(traces), column, grid)


def _fixed_style(period: float) -> dict:
    return {"color": style.fixed_period_color(period), "linestyle": "--", "linewidth": 0.9}


def source_switch_dynamics(data: Dataset, out: Path) -> None:
    """Mean gradient vs. ground truth, its deviation and the wake-up period around the switch."""
    weight = config.FOCUS_WEIGHT
    fig, axes = style.figure("double", 3.9, 3, 1, sharex=True)
    gradient_ax, deviation_ax, period_ax = axes
    configurations = [(m, select(data.traces, m.key, weight=weight)) for m in LEARNED]
    for method, traces in configurations:
        seed = traces[traces.seed == SHOWN_SEED]
        gradient_ax.plot(ZOOM_GRID, np.interp(ZOOM_GRID, seed.t, seed.gradient_mean), color=method.color, linewidth=1.0)
        method_band(
            deviation_ax, ZOOM_GRID, _matrix(traces, "deviation", ZOOM_GRID), method, markevery=6, nonnegative=True
        )
        method_band(period_ax, ZOOM_GRID, _matrix(traces, "wakeup", ZOOM_GRID), method, markevery=6)
    for period in REFERENCE_PERIODS:
        traces = select(data.traces, FIXED.key, weight=weight, period=period)
        seed = traces[traces.seed == SHOWN_SEED]
        gradient_ax.plot(ZOOM_GRID, np.interp(ZOOM_GRID, seed.t, seed.gradient_mean), **_fixed_style(period))
        mean_band(deviation_ax, ZOOM_GRID, _matrix(traces, "deviation", ZOOM_GRID), band=False, **_fixed_style(period))
        period_ax.plot(ZOOM_GRID, np.full_like(ZOOM_GRID, period), **_fixed_style(period))
    truth = select(data.traces, TD3BC.key, weight=weight)
    truth = truth[truth.seed == SHOWN_SEED]
    gradient_ax.plot(
        ZOOM_GRID,
        np.interp(ZOOM_GRID, truth.t, truth.truth_mean),
        color=style.INK,
        linestyle=":",
        linewidth=1.1,
        zorder=5,
    )
    gradient_ax.set_ylabel(f"Mean gradient (m)\nseed {SHOWN_SEED}")
    deviation_ax.set_ylabel(DEVIATION_LABEL)
    period_ax.set_ylabel(PERIOD_LABEL)
    period_ax.set_ylim(0.8, 5.4)
    period_ax.set_xlabel(EPISODE_TIME_LABEL)
    period_ax.set_xlim(*ZOOM_S)
    for index, ax in enumerate(axes):
        style.mark_source_switch(ax, label=ax is period_ax, label_y=0.03)
        style.panel_label(ax, "abc"[index])
    handles = [style.legend_handle(m) for m in LEARNED]
    handles += [Line2D([], [], label=f"Fixed {p:g} s", **_fixed_style(p)) for p in REFERENCE_PERIODS]
    handles.append(Line2D([], [], color=style.INK, linestyle=":", linewidth=1.1, label="Ground truth $\\bar g^*$"))
    style.figure_legend(fig, handles, ncol=4)
    style.save(fig, out, "source_switch_dynamics")


def beta_effect_intra_episode(data: Dataset, out: Path) -> None:
    """How beta shapes the wake-up schedule and the deviation of TD3+BC during the final episode."""
    fig, axes = style.figure("single", 3.3, 2, 1, sharex=True)
    period_ax, deviation_ax = axes
    for weight in config.WEIGHTS:
        traces = select(data.traces, TD3BC.key, weight=weight)
        color = style.BETA_COLORS[weight]
        mean_band(period_ax, EPISODE_GRID, _matrix(traces, "wakeup", EPISODE_GRID), color, f"{weight:g}", linewidth=1.0)
        mean_band(
            deviation_ax,
            EPISODE_GRID,
            _matrix(traces, "deviation", EPISODE_GRID),
            color,
            linewidth=1.0,
            nonnegative=True,
        )
    period_ax.set_ylabel(PERIOD_LABEL)
    period_ax.set_ylim(0.8, 5.2)
    deviation_ax.set_ylabel(DEVIATION_LABEL)
    deviation_ax.set_xlabel(EPISODE_TIME_LABEL)
    deviation_ax.set_xlim(0, config.EPISODE_LENGTH_S)
    for index, ax in enumerate(axes):
        style.mark_source_switch(ax, label=ax is deviation_ax)
        style.panel_label(ax, "ab"[index])
    handles, labels = period_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), title=f"{TD3BC.short}, $\\beta$ =")
    style.save(fig, out, "beta_effect_intra_episode")


FIGURES = (source_switch_dynamics, beta_effect_intra_episode)
