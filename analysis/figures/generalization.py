"""Zero-shot generalization to other self-stabilising building blocks.

All policies are trained for 10 episodes on the Euclidean gradient only; the
evaluation episodes then run another program (gradient, hop-count gradient,
broadcast, collect). beta-conditioned policies are queried with beta_in
= 0.5 ... 1 (two episodes each). Only one seed was run for this experiment,
so there are no across-seed intervals: whiskers show the range over the
evaluation episodes of each point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from .. import config, style
from ..dataset import Dataset
from ..methods import BCQ, FIXED, GNN, HEURISTIC, LEARNED, QLEARNING, TD3BC, Method
from .common import METRIC_LABELS

PROGRAMS = tuple(config.PROGRAMS.values())
LEARNED_BETA = 0.95  # beta_in of the beta-conditioned policies in the overview
HEURISTIC_BETA = 0.8  # beta_in of the heuristic: its rounds then match the learned policies' (~7k)
OVERVIEW_PERIODS = (5.0, 3.0)
SWEPT = (HEURISTIC, BCQ, TD3BC, GNN)  # drawn as beta_in-swept curves
LOG_TICKS = (4000, 5000, 7000, 10000, 15000, 20000)


def _evaluation(data: Dataset) -> pd.DataFrame:
    table = data.generalization
    return table[table.episode.isin(config.EVAL_EPISODES)]


def _episodes(
    data: Dataset, program: str, method: str, period: float | None = None, beta_in: float | None = None
) -> pd.DataFrame:
    """Evaluation episodes of one configuration (beta_in None: all of them)."""
    rows = _evaluation(data)
    rows = rows[(rows.program == program) & (rows.method == method)]
    if period is not None:
        rows = rows[np.isclose(rows.period, period)]
    if beta_in is not None:
        rows = rows[np.isclose(rows.beta_in, beta_in)]
    return rows


def _overview_entries() -> list[tuple[str, str, str, str, float | None, float | None]]:
    """(label, color, marker, method key, period, beta_in) of each overview point, left to right."""
    entries = [
        (f"Fixed {p:g} s", style.fixed_period_color(p), FIXED.marker, FIXED.key, p, None) for p in OVERVIEW_PERIODS
    ]
    entries.append(
        (
            f"{HEURISTIC.label} ($\\beta_{{in}}$ = {HEURISTIC_BETA:g})",
            HEURISTIC.color,
            HEURISTIC.marker,
            HEURISTIC.key,
            None,
            HEURISTIC_BETA,
        )
    )
    for method in LEARNED:
        beta = None if method is QLEARNING else LEARNED_BETA
        suffix = "" if beta is None else f" ($\\beta_{{in}}$ = {beta:g})"
        entries.append((method.label + suffix, method.color, method.marker, method.key, None, beta))
    return entries


def generalization_overview(data: Dataset, out: Path) -> None:
    """Unstable time, error (relative to the 5 s period) and rounds on each building block."""
    entries = _overview_entries()
    metrics = ("unstable", "relative_error", "ticks")
    labels = {**METRIC_LABELS, "relative_error": "Error relative to fixed 5 s (%)"}
    width = 0.8 / len(entries)
    fig, axes = style.figure("double", 2.3, 1, len(metrics))
    for ax, metric in zip(axes, metrics):
        for index, (label, color, marker, method, period, beta) in enumerate(entries):
            for position, program in enumerate(PROGRAMS):
                rows = _episodes(data, program, method, period, beta)
                values = rows.error.to_numpy() if metric == "relative_error" else rows[metric].to_numpy()
                if metric == "relative_error":
                    reference = _episodes(data, program, FIXED.key, period=max(config.FIXED_PERIODS_S)).error.mean()
                    values = values / reference * 100.0
                x = position + (index - (len(entries) - 1) / 2) * width
                ax.plot([x, x], [values.min(), values.max()], color=color, linewidth=0.6)
                ax.plot(
                    x,
                    values.mean(),
                    marker=marker,
                    color=color,
                    markersize=3.6 * (1.3 if marker == "*" else 1),
                    markeredgecolor="white",
                    markeredgewidth=0.3,
                    linestyle="none",
                    label=label if position == 0 else None,
                )
        ax.set_xticks(range(len(PROGRAMS)), [config.PROGRAM_SHORT_LABELS[p] for p in PROGRAMS])
        ax.set_ylabel(labels[metric])
        ax.grid(axis="x", visible=False)
        if metric == "ticks":
            ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:g}k")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside upper center", ncol=4)
    style.save(fig, out, "generalization_overview")


def _swept_curve(data: Dataset, program: str, method: Method, metric: str) -> tuple[np.ndarray, np.ndarray]:
    rows = _episodes(data, program, method.key)
    points = rows.groupby("beta_in")[["ticks", metric]].mean().sort_values("ticks")
    return points.ticks.to_numpy(), points[metric].to_numpy()


def _generalization_pareto(data: Dataset, out: Path, metric: str, name: str) -> None:
    fig, axes = style.figure("double", 2.15, 1, len(PROGRAMS))
    for ax, program in zip(axes, PROGRAMS):
        fixed = _evaluation(data)
        fixed = fixed[(fixed.program == program) & (fixed.method == FIXED.key)]
        frontier = fixed.groupby("period")[["ticks", metric]].mean().reset_index()
        ax.plot(frontier.ticks, frontier[metric], **style.fixed_period_style())
        style.label_fixed_periods(ax, frontier.ticks, frontier[metric], frontier.period, dx=2, dy=1)
        for method in SWEPT:
            ticks, values = _swept_curve(data, program, method, metric)
            ax.plot(ticks, values, linewidth=0.8, **style.line_style(method))
        q = _episodes(data, program, QLEARNING.key)
        ax.plot(q.ticks.mean(), q[metric].mean(), linestyle="none", **style.line_style(QLEARNING, markersize=4.5))
        ax.set_title(config.PROGRAM_LABELS[program])
        ax.set_xscale("log")
        ax.set_xticks(LOG_TICKS, [f"{t / 1000:g}k" for t in LOG_TICKS])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Rounds per episode")
        ax.margins(y=0.12)
    axes[0].set_ylabel(METRIC_LABELS[metric])
    handles = [Line2D([], [], **style.fixed_period_style())]
    handles += [style.legend_handle(m, linewidth=0.8, label=f"{m.label} ($\\beta_{{in}}$ swept)") for m in SWEPT]
    handles.append(style.legend_handle(QLEARNING, linestyle="none", markersize=4.5))
    style.figure_legend(fig, handles, ncol=3)
    style.save(fig, out, name)


def generalization_pareto_error(data: Dataset, out: Path) -> None:
    """Rounds vs. error on each building block, with beta_in swept for the conditioned policies."""
    _generalization_pareto(data, out, "error", "generalization_pareto_error")


def generalization_pareto_unstable(data: Dataset, out: Path) -> None:
    """Rounds vs. unstable time on each building block (companion of the error figure)."""
    _generalization_pareto(data, out, "unstable", "generalization_pareto_unstable")


FIGURES = (generalization_overview, generalization_pareto_error, generalization_pareto_unstable)
