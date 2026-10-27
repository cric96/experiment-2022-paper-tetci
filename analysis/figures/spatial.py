"""Spatial view of the final episode (TD3, seed 2): filmstrip, computation map, animation.

Data: ``spatial/spatial_snapshots_weight-<beta>.csv`` (one row per node every
2 s of the final episode: position, source flag, gradient, ground truth,
wake-up period, error, cumulative rounds) and ``spatial_edges_weight-<beta>.csv``.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image
from scipy import stats as sps

from .. import config, style
from ..dataset import Dataset

SNAPSHOT_BETAS = (0.8, 0.95)  # first available one is used
FILM_TIMES_S = (95.0, 103.0, 108.0, 116.0, 138.0)
BEFORE_SWITCH_S, AFTER_SWITCH_S = 95.0, 150.0  # instants used to locate the two sources
UNREACHABLE_GRADIENT_M = 600.0  # value exported for an infinite gradient
GRADIENT_RANGE_M = (0.0, 520.0)
RATE_RANGE_HZ = (0.2, 1.0)
ERROR_RANGE_M = (0.0, 180.0)
MIN_PERIOD_S = 0.5
ANIMATION_WINDOW_S = (88.0, 148.0)
ANIMATION_FRAME_MS = 220
ANIMATION_DPI = 110


class Snapshots:
    """Node snapshots of the final episode, accessed by (nearest) time."""

    def __init__(self, frame: pd.DataFrame, edges: pd.DataFrame, beta: float) -> None:
        self.frame = frame
        self.edges = edges
        self.beta = beta
        self.times = np.unique(frame.tEp.to_numpy())
        first = self.at(self.times[0])
        self.triangulation = mtri.Triangulation(first.x, first.y)

    @classmethod
    def load(cls, data: Dataset) -> Snapshots | None:
        for beta in SNAPSHOT_BETAS:
            path = data.spatial_dir / f"spatial_snapshots_weight-{beta}.csv"
            if path.exists():
                edges = pd.read_csv(data.spatial_dir / f"spatial_edges_weight-{beta}.csv")
                return cls(pd.read_csv(path), edges, beta)
        return None

    def at(self, t: float) -> pd.DataFrame:
        """Nodes (one row each, sorted by id) at the snapshot closest to ``t``."""
        nearest = self.times[np.argmin(np.abs(self.times - t))]
        rows = self.frame[np.isclose(self.frame.tEp, nearest)]
        return rows.drop_duplicates("nodeId").sort_values("nodeId").reset_index(drop=True)

    def degrees(self, node_ids: np.ndarray) -> np.ndarray:
        counts = pd.concat([self.edges.u, self.edges.v]).value_counts()
        return counts.reindex(node_ids, fill_value=0).to_numpy()


def _draw_edges(ax: Axes, edges: pd.DataFrame, color: str = "white", alpha: float = 0.35) -> None:
    segments = np.stack([edges[["xU", "yU"]].to_numpy(), edges[["xV", "yV"]].to_numpy()], axis=1)
    ax.add_collection(LineCollection(segments, colors=color, linewidths=0.3, alpha=alpha, zorder=2))


def _source_style(filled: bool) -> dict:
    return {
        "marker": "*",
        "markersize": 8,
        "markeredgewidth": 0.6,
        "linestyle": "none",
        "markerfacecolor": style.INK if filled else "white",
        "markeredgecolor": "white" if filled else style.INK,
    }


def _draw_sources(ax: Axes, nodes: pd.DataFrame, filled: bool = True) -> None:
    sources = nodes[nodes.isSource > 0.5]
    ax.plot(sources.x, sources.y, zorder=5, clip_on=False, **_source_style(filled))


def _field(ax: Axes, snapshots: Snapshots, nodes: pd.DataFrame, values: np.ndarray, cmap, value_range) -> object:
    low, high = value_range
    levels = np.linspace(low, high, 15)
    # Clip just inside the top level so saturated nodes (e.g. unreachable ones) stay coloured.
    clipped = np.clip(values, low, high - 1e-6 * (high - low))
    filled = ax.tricontourf(snapshots.triangulation, clipped, levels=levels, cmap=cmap)
    _draw_edges(ax, snapshots.edges)
    ax.scatter(nodes.x, nodes.y, s=2, color=style.INK, linewidths=0, zorder=3)
    _draw_sources(ax, nodes)
    ax.set_aspect("equal")
    ax.grid(False)
    return filled


def _rows(nodes: pd.DataFrame) -> list[tuple[np.ndarray, object, tuple[float, float], str]]:
    rate = 1.0 / np.maximum(MIN_PERIOD_S, nodes.nextWakeUp.to_numpy())
    return [
        (nodes.gradient.to_numpy(), style.SEQUENTIAL, GRADIENT_RANGE_M, "Gradient $g$ (m)"),
        (rate, style.SEQUENTIAL_2, RATE_RANGE_HZ, "Round rate $1/\\Delta t$ (Hz)"),
        (nodes.error.to_numpy(), style.SEQUENTIAL_3, ERROR_RANGE_M, "Error $|g - g^*|$ (m)"),
    ]


def spatial_filmstrip(data: Dataset, out: Path) -> None:
    """Gradient, round rate and error over the network at five instants around the switch."""
    snapshots = Snapshots.load(data)
    if snapshots is None:
        return
    fig, axes = style.figure("double", 4.6, 3, len(FILM_TIMES_S), sharex=True, sharey=True)
    for col, t in enumerate(FILM_TIMES_S):
        nodes = snapshots.at(t)
        for row, (values, cmap, value_range, label) in enumerate(_rows(nodes)):
            ax = axes[row, col]
            filled = _field(ax, snapshots, nodes, values, cmap, value_range)
            if col == len(FILM_TIMES_S) - 1:
                bar = fig.colorbar(
                    filled,
                    ax=axes[row, :].tolist(),
                    fraction=0.02,
                    pad=0.01,
                    ticks=np.linspace(*value_range, 5),
                    format="%g",
                )
                bar.set_label(label, fontsize=7)
                bar.outline.set_visible(False)
        axes[0, col].set_title(f"$t$ = {t:g} s")
        axes[-1, col].set_xlabel("$x$ (m)")
    for row in range(3):
        axes[row, 0].set_ylabel("$y$ (m)")
    style.save(fig, out, "spatial_filmstrip")


def rounds_per_node(snapshots: Snapshots) -> pd.DataFrame:
    """Rounds of each node over the final episode, with its distance to the nearest source and degree."""
    start, end = snapshots.at(snapshots.times[0]), snapshots.at(snapshots.times[-1])
    before, after = snapshots.at(BEFORE_SWITCH_S), snapshots.at(AFTER_SWITCH_S)
    return pd.DataFrame(
        {
            "x": end.x.to_numpy(),
            "y": end.y.to_numpy(),
            "rounds": end.ticks.to_numpy() - start.ticks.to_numpy(),
            # the ground truth is the shortest-path distance to the source active at that time
            "distance": np.minimum(before.groundTruth.to_numpy(), after.groundTruth.to_numpy()),
            "degree": snapshots.degrees(end.nodeId.to_numpy()),
        }
    )


def rounds_distance_correlation(data: Dataset) -> tuple[float, float] | None:
    """Spearman correlation (rho, p) between rounds per node and distance to the nearest source."""
    snapshots = Snapshots.load(data)
    if snapshots is None:
        return None
    nodes = rounds_per_node(snapshots)
    result = sps.spearmanr(nodes.distance, nodes.rounds)
    return float(result.statistic), float(result.pvalue)


def spatial_computation(data: Dataset, out: Path) -> None:
    """Rounds per node over the final episode: map, and against distance to the sources."""
    snapshots = Snapshots.load(data)
    if snapshots is None:
        return
    before, after = snapshots.at(BEFORE_SWITCH_S), snapshots.at(AFTER_SWITCH_S)
    nodes = rounds_per_node(snapshots)
    rounds, distance, degree = nodes.rounds.to_numpy(), nodes.distance.to_numpy(), nodes.degree.to_numpy()

    fig, (map_ax, scatter_ax) = style.figure("double", 2.6, 1, 2)
    filled = map_ax.tricontourf(snapshots.triangulation, rounds, levels=14, cmap=style.SEQUENTIAL)
    _draw_edges(map_ax, snapshots.edges, color=style.INK, alpha=0.25)
    map_ax.scatter(nodes.x, nodes.y, s=3, color=style.INK, linewidths=0, zorder=3)
    _draw_sources(map_ax, before, filled=True)
    _draw_sources(map_ax, after, filled=False)
    map_ax.set_aspect("equal")
    map_ax.grid(False)
    map_ax.set_xlabel("$x$ (m)")
    map_ax.set_ylabel("$y$ (m)")
    bar = fig.colorbar(filled, ax=map_ax, fraction=0.046, pad=0.02)
    bar.set_label("Rounds per node in the episode", fontsize=7)
    bar.outline.set_visible(False)
    handles = [
        Line2D([], [], label="source, $t$ < 100 s", **_source_style(True)),
        Line2D([], [], label="source, $t$ > 100 s", **_source_style(False)),
    ]
    map_ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    points = scatter_ax.scatter(distance, rounds, c=degree, cmap=style.SEQUENTIAL_3, s=10, linewidths=0)
    rho, p_value = rounds_distance_correlation(data)
    scatter_ax.text(
        0.98,
        0.97,
        f"Spearman $\\rho = {rho:.2f}$ ($p = {p_value:.2f}$)",
        transform=scatter_ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=style.INK_SECONDARY,
    )
    scatter_ax.set_xlabel("Distance to the nearest source (m)")
    scatter_ax.set_ylabel("Rounds per node in the episode")
    bar = fig.colorbar(points, ax=scatter_ax, fraction=0.046, pad=0.02)
    bar.set_label("Node degree", fontsize=7)
    bar.outline.set_visible(False)
    style.panel_label(map_ax, "a")
    style.panel_label(scatter_ax, "b")
    style.save(fig, out, "spatial_computation")


def spatial_wave_animation(data: Dataset, out: Path) -> None:
    """GIF of the three fields around the switch (for presentations, not the paper)."""
    snapshots = Snapshots.load(data)
    if snapshots is None:
        return
    window = snapshots.times[(snapshots.times >= ANIMATION_WINDOW_S[0]) & (snapshots.times <= ANIMATION_WINDOW_S[1])]
    frames = []
    for t in window[::2]:
        nodes = snapshots.at(t)
        fig, axes = style.figure("double", 2.6, 1, 3)
        for ax, (values, cmap, value_range, label) in zip(axes, _rows(nodes)):
            filled = _field(ax, snapshots, nodes, values, cmap, value_range)
            bar = fig.colorbar(filled, ax=ax, fraction=0.046, pad=0.02, orientation="horizontal", location="bottom")
            bar.set_label(label, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"$t$ = {t:.0f} s (source moves at {config.SOURCE_SWITCH_S:g} s)", fontsize=8)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=ANIMATION_DPI)
        plt.close(fig)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE))
    out.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out / "spatial_wave_animation.gif", save_all=True, append_images=frames[1:], duration=ANIMATION_FRAME_MS, loop=0
    )


FIGURES = (spatial_filmstrip, spatial_computation, spatial_wave_animation)
