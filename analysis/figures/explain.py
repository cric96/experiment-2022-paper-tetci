"""Explainability of the TD3 policy: hidden-layer embeddings, attribution, response surfaces.

Data: ``embeddings/embeddings_weight-<beta>.csv`` (800 sampled states with the
actor's 64-d penultimate activations ``h_*``, predicted period ``predDt``,
critic value ``qValue``, a physical ``regime`` label and finite-difference
sensitivities ``sens_*``) and ``embeddings/surface_weight-<beta>.csv`` (policy
output on a grid of self / neighbour change intensities).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, style
from ..dataset import Dataset

FOCUS_BETA = config.FOCUS_WEIGHT
ATTRIBUTION_BETAS = (0.8, 0.95, 0.99)
# Lower bound on a feature's spread when scaling its sensitivity (avoids zero weight for rare features).
MIN_FEATURE_STD = 0.15

FEATURES = {
    "deltaLatest": "Latest trend $\\Delta g(t)$",
    "deltaPrev": "Previous trend $\\Delta g(t-1)$",
    "selfDeltaMag": "Self change $|\\Delta g_i|$",
    "maxNeighDeltaMag": "Max neighbour change $|\\Delta g_j|$",
    "neighChangedRatio": "Changed-neighbour ratio",
    "bfResidual": "Bellman-Ford residual",
    "childrenRatio": "Children ratio (backbone)",
    "minNeighAction": "Min neighbour period",
}
BIPLOT_FEATURES = {
    "selfDeltaMag": "self change",
    "maxNeighDeltaMag": "neigh. change",
    "bfResidual": "BF residual",
    "childrenRatio": "children ratio",
    "minNeighAction": "min neigh. period",
}
# Physical regimes, ordered by how involved the node is in the wave (ordinal grey ramp).
REGIMES = {
    "Steady_State": ("Steady state", "o"),
    "Neighbor_EarlyWarning": ("Neighbour changing", "v"),
    "Leaf_Wave": ("Wave, leaf node", "<"),
    "Backbone_Wave": ("Wave, backbone node", ">"),
}
LEAF_CHILDREN_RATIO = 0.0
BACKBONE_CHILDREN_RATIO = 0.5


def _embeddings(data: Dataset, beta: float) -> pd.DataFrame:
    return pd.read_csv(data.embeddings_dir / f"embeddings_weight-{beta}.csv")


def _surface(data: Dataset, beta: float) -> pd.DataFrame:
    return pd.read_csv(data.embeddings_dir / f"surface_weight-{beta}.csv")


def _hidden(frame: pd.DataFrame) -> np.ndarray:
    return frame[[c for c in frame.columns if c.startswith("h_")]].to_numpy()


def pca(matrix: np.ndarray, components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Scores on the first principal components and the explained-variance ratios."""
    centred = matrix - matrix.mean(axis=0)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    return u[:, :components] * s[:components], (s**2 / np.sum(s**2))[:components]


def rbf_kernel_pca(matrix: np.ndarray, components: int = 2) -> np.ndarray:
    """Kernel PCA with a Gaussian kernel (median-distance bandwidth)."""
    centred = matrix - matrix.mean(axis=0)
    norms = np.sum(centred**2, axis=1, keepdims=True)
    distances = np.maximum(0.0, norms + norms.T - 2.0 * centred @ centred.T)
    bandwidth = np.median(distances[distances > 0]) if np.any(distances > 0) else 1.0
    kernel = np.exp(-distances / (2.0 * bandwidth))
    n = len(kernel)
    ones = np.full((n, n), 1.0 / n)
    centred_kernel = kernel - ones @ kernel - kernel @ ones + ones @ kernel @ ones
    values, vectors = np.linalg.eigh(centred_kernel)
    top = np.argsort(values)[::-1][:components]
    return vectors[:, top] * np.sqrt(np.maximum(values[top], 1e-12))


def _scatter_colorbar(fig, ax, mappable, label: str) -> None:
    bar = fig.colorbar(mappable, ax=ax, fraction=0.05, pad=0.02)
    bar.set_label(label, fontsize=7)
    bar.outline.set_visible(False)


def policy_embeddings(data: Dataset, out: Path) -> None:
    """PCA of the actor's hidden layer by regime (biplot), by predicted period; kernel PCA by Q."""
    frame = _embeddings(data, FOCUS_BETA)
    scores, explained = pca(_hidden(frame))
    fig, axes = style.figure("double", 2.3, 1, 3)
    fig.get_layout_engine().set(wspace=0.06)
    regime_ax, period_ax, value_ax = axes
    for (key, (label, marker)), color in zip(REGIMES.items(), style.REGIME_GREYS):
        rows = frame.regime.to_numpy() == key
        regime_ax.scatter(
            scores[rows, 0], scores[rows, 1], s=6, marker=marker, color=color, label=label, linewidths=0, alpha=0.8
        )
    # Biplot: one arrow per feature, pointing along its correlation with (PC1, PC2).
    loadings = {
        key: [np.corrcoef(frame[key], scores[:, k])[0, 1] for k in (0, 1)]
        for key in BIPLOT_FEATURES
        if frame[key].std() > 1e-6
    }
    longest = max(np.hypot(*v) for v in loadings.values())
    reach = 0.9 * np.abs(scores).max(axis=0) / longest
    for key, (cx, cy) in loadings.items():
        dx, dy = cx * reach[0], cy * reach[1]
        regime_ax.annotate(
            "", xy=(dx, dy), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": style.INK, "linewidth": 0.7}
        )
        regime_ax.text(
            dx,
            dy,
            f" {BIPLOT_FEATURES[key]} ",
            fontsize=5.5,
            color=style.INK,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            bbox={"boxstyle": "round,pad=0.05", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    regime_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=6,
        markerscale=1.8,
        handletextpad=0.2,
        columnspacing=0.8,
    )
    for ax in (regime_ax, period_ax):
        ax.set_xlabel(f"PC1 ({explained[0]:.0%} var.)")
        ax.set_ylabel(f"PC2 ({explained[1]:.0%} var.)")
    points = period_ax.scatter(scores[:, 0], scores[:, 1], c=frame.predDt, cmap=style.SEQUENTIAL, s=5, linewidths=0)
    _scatter_colorbar(fig, period_ax, points, "Predicted period (s)")
    kernel_scores = rbf_kernel_pca(_hidden(frame))
    points = value_ax.scatter(
        kernel_scores[:, 0], kernel_scores[:, 1], c=frame.qValue, cmap=style.SEQUENTIAL_3, s=5, linewidths=0
    )
    _scatter_colorbar(fig, value_ax, points, "Critic value $Q(s, \\pi(s))$")
    value_ax.set_xlabel("RBF kernel PC1")
    value_ax.set_ylabel("RBF kernel PC2")
    for index, ax in enumerate(axes):
        style.panel_label(ax, "abc"[index])
        ax.grid(False)
    style.save(fig, out, "policy_embeddings")


def feature_attribution(frame: pd.DataFrame) -> pd.Series:
    """Share (%) of each input in the policy output: mean |d pi / d x| scaled by the feature spread."""
    impact = pd.Series({key: frame[f"sens_{key}"].mean() * max(MIN_FEATURE_STD, frame[key].std()) for key in FEATURES})
    return impact / impact.sum() * 100.0


def _surface_grid(surface: pd.DataFrame, children_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = surface[np.isclose(surface.childrenRatio, children_ratio)]
    table = rows.pivot(index="maxNeighDeltaMag", columns="selfDeltaMag", values="predDt")
    return table.columns.to_numpy(), table.index.to_numpy(), table.to_numpy()


def policy_explanation(data: Dataset, out: Path) -> None:
    """Input attribution for three betas, response curves and decision surfaces at beta = 0.95."""
    fig = style.canvas("double", 4.3)
    grid = fig.add_gridspec(2, 2)
    attribution_ax = fig.add_subplot(grid[0, 0])
    response_ax = fig.add_subplot(grid[0, 1])
    surface_axes = (fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]))

    positions = np.arange(len(FEATURES))[::-1]
    height = 0.8 / len(ATTRIBUTION_BETAS)
    for index, beta in enumerate(ATTRIBUTION_BETAS):
        shares = feature_attribution(_embeddings(data, beta))
        attribution_ax.barh(
            positions + (1 - index) * height,
            shares.to_numpy(),
            height=height * 0.9,
            color=style.BETA_COLORS[beta],
            label=f"$\\beta$ = {beta:g}",
        )
    attribution_ax.set_yticks(positions, list(FEATURES.values()))
    attribution_ax.set_xlabel("Share of policy sensitivity (%)")
    attribution_ax.grid(axis="y", visible=False)
    attribution_ax.legend(loc="lower right")

    surface = _surface(data, FOCUS_BETA)
    curves = (
        (BACKBONE_CHILDREN_RATIO, "self", "-", style.INK, "self, backbone"),
        (LEAF_CHILDREN_RATIO, "self", "--", style.INK, "self, leaf"),
        (BACKBONE_CHILDREN_RATIO, "neigh", "-", style.INK_MUTED, "neighbour, backbone"),
        (LEAF_CHILDREN_RATIO, "neigh", "--", style.INK_MUTED, "neighbour, leaf"),
    )
    for children_ratio, varied, linestyle, color, label in curves:
        x, y, z = _surface_grid(surface, children_ratio)
        values = z[0, :] if varied == "self" else z[:, 0]  # the other intensity held at zero
        intensities = x if varied == "self" else y
        response_ax.plot(intensities, values, linestyle=linestyle, color=color, linewidth=1.0, label=label)
    response_ax.set_xlabel("Change intensity $\\tanh(|\\Delta g| / 50)$ (other one = 0)")
    response_ax.set_ylabel("Chosen period (s)")
    response_ax.legend(loc="upper right", title="varied input, node")

    grids = [_surface_grid(surface, ratio) for ratio in (LEAF_CHILDREN_RATIO, BACKBONE_CHILDREN_RATIO)]
    low = min(g[2].min() for g in grids)
    high = max(g[2].max() for g in grids)
    levels = np.linspace(low, high, 15)
    titles = ("Leaf node (children ratio 0)", "Backbone node (children ratio 0.5)")
    for ax, (x, y, z), title in zip(surface_axes, grids, titles):
        filled = ax.contourf(x, y, z, levels=levels, cmap=style.SEQUENTIAL)
        lines = ax.contour(x, y, z, levels=levels[1::4], colors="white", linewidths=0.4)
        ax.clabel(lines, inline=True, fontsize=5.5, fmt="%.1f s")
        ax.set_title(title)
        ax.set_xlabel("Self change intensity")
        ax.set_ylabel("Neighbour change intensity")
        ax.grid(False)
    bar = fig.colorbar(filled, ax=list(surface_axes), fraction=0.03, pad=0.02, format="%.1f")
    bar.set_label("Chosen period (s)", fontsize=7)
    bar.outline.set_visible(False)
    for ax, letter in zip((attribution_ax, response_ax, *surface_axes), "abcd"):
        style.panel_label(ax, letter)
    style.save(fig, out, "policy_explanation")


FIGURES = (policy_embeddings, policy_explanation)
