"""Drawing helpers shared by several figure families."""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from .. import config, style
from ..methods import Method
from ..stats import Interval, t_band, t_interval

BAND_ALPHA = 0.15

METRIC_LABELS = {
    "ticks": "Rounds per episode",
    "reward": "Return per node",
    "unstable": "Unstable time per node (s)",
    "error": "Integrated error per node (s)",
}


def mean_band(ax: Axes, x: np.ndarray, matrix: np.ndarray, color: str, label: str | None = None,
              band: bool = True, **line_kwargs) -> None:
    """Mean across seeds (rows of ``matrix``) with its 95 % t-interval as a band."""
    mean, low, high = t_band(matrix)
    ax.plot(x, mean, color=color, label=label, **line_kwargs)
    if band:
        ax.fill_between(x, low, high, color=color, alpha=BAND_ALPHA, linewidth=0)


def method_band(ax: Axes, x: np.ndarray, matrix: np.ndarray, method: Method, markevery: int | None = None,
                band: bool = True, **overrides) -> None:
    kwargs = style.line_style(method, **overrides)
    if markevery is not None:
        kwargs["markevery"] = markevery
    color = kwargs.pop("color")
    label = kwargs.pop("label")
    mean_band(ax, x, matrix, color, label, band=band, **kwargs)


def episode_matrix(episodes, metric: str) -> np.ndarray:
    """Seeds x episodes matrix of ``metric`` for one configuration."""
    table = episodes.pivot(index="seed", columns="episode", values=metric)
    return table.reindex(columns=range(1, config.N_EPISODES + 1)).to_numpy()


def ci_of(values) -> Interval:
    return t_interval(values)


def errorbar(ax: Axes, x: Interval, y: Interval, method: Method, **overrides) -> None:
    """A (x, y) point with 95 % intervals on both axes, in the method's style."""
    kwargs = style.line_style(method, linestyle="none", elinewidth=0.6, **overrides)
    ax.errorbar(x.mean, y.mean, xerr=[[x.mean - x.low], [x.high - x.mean]],
                yerr=[[y.mean - y.low], [y.high - y.mean]], **kwargs)


def thousands(ax: Axes, axis: str = "x") -> None:
    """Format an axis in thousands (e.g. rounds) with a k suffix."""
    from matplotlib.ticker import FuncFormatter

    formatter = FuncFormatter(lambda v, _: f"{v / 1000:g}k")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(formatter)
