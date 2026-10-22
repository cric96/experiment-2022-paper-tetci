"""Shared paper style for IEEE two-column figures.

* sizes: single column 3.5 in, double column 7.16 in;
* serif text matching a Times body (STIX, bundled with matplotlib: no TeX needed),
  8 pt labels, 7 pt ticks and legends;
* vector PDF with embedded TrueType fonts (``pdf.fonttype = 42``) plus a PNG
  copy for quick viewing;
* recessive solid hairline grid, no top/right spines, no figure titles
  (captions live in LaTeX);
* colours: one colour + marker per method (``analysis.methods``); ordinal
  single-hue ramps for beta and the fixed periods; single-hue sequential maps
  for magnitudes and a blue-red diverging map for signed quantities.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from . import config  # noqa: E402
from .methods import FIXED, FIXED_PERIOD_GREYS, Method  # noqa: E402

SINGLE_COLUMN_IN = 3.5
DOUBLE_COLUMN_IN = 7.16

INK = "#1a1a1a"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
PHASE_SHADE = "#f2f2ef"

# Ordinal ramps (validated: monotone lightness, >= 2:1 light end on white).
BETA_COLORS = dict(zip(config.WEIGHTS, ("#fd9344", "#f36f1a", "#dd4d04", "#ae3903", "#852904")))
REGIME_GREYS = ("#afafaf", "#7b7b7b", "#4a4a4a", "#0e0e0e")


def _single_hue(name: str, start: float = 0.22) -> mpl.colors.Colormap:
    """A matplotlib single-hue map without its near-white end (small marks stay visible)."""
    base = mpl.colormaps[name]
    return mpl.colors.LinearSegmentedColormap.from_list(f"{name}_paper", base(np.linspace(start, 1.0, 256)))


# Continuous maps: one hue per magnitude (a second / third magnitude in the
# same figure takes the next hue); blue-red diverging for signed values.
SEQUENTIAL = _single_hue("Blues")
SEQUENTIAL_2 = _single_hue("Oranges")
SEQUENTIAL_3 = _single_hue("Purples")
DIVERGING = "RdBu_r"

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.titleweight": "normal",
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.borderaxespad": 0.3,
    "legend.columnspacing": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK_SECONDARY,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "errorbar.capsize": 1.5,
}


def apply() -> None:
    """Install the paper style globally."""
    mpl.rcParams.update(PAPER_RC)


def figure(
    width: str = "double", height: float = 2.2, nrows: int = 1, ncols: int = 1, **kwargs
) -> tuple[Figure, object]:
    """Create a figure ``single`` (3.5 in) or ``double`` (7.16 in) column wide."""
    inches = {"single": SINGLE_COLUMN_IN, "double": DOUBLE_COLUMN_IN}[width]
    return plt.subplots(nrows, ncols, figsize=(inches, height), **kwargs)


def canvas(width: str = "double", height: float = 2.2) -> Figure:
    """An empty figure (for custom grids) ``single`` or ``double`` column wide."""
    inches = {"single": SINGLE_COLUMN_IN, "double": DOUBLE_COLUMN_IN}[width]
    return plt.figure(figsize=(inches, height))


def save(fig: Figure, out_dir: Path, name: str) -> list[Path]:
    """Write ``name``.pdf (vector, for the paper) and ``name``.png, then close the figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


# --- method-level drawing helpers --------------------------------------------


def line_style(method: Method, **overrides) -> dict:
    """Keyword arguments for ``plot``/``errorbar`` lines of a method."""
    style = {
        "color": method.color,
        "marker": method.marker,
        "markersize": PAPER_RC["lines.markersize"] * method.marker_scale,
        "markeredgecolor": "white",
        "markeredgewidth": 0.4,
        "label": method.label,
    }
    style.update(overrides)
    return style


def fixed_period_style(**overrides) -> dict:
    """Keyword arguments for the fixed-period frontier."""
    style = {
        "color": FIXED.color,
        "marker": FIXED.marker,
        "markersize": PAPER_RC["lines.markersize"],
        "linestyle": "--",
        "linewidth": 0.9,
        "label": "Fixed period (1–5 s)",
    }
    style.update(overrides)
    return style


def fixed_period_color(period: float) -> str:
    return FIXED_PERIOD_GREYS[period]


def legend_handle(method: Method, **overrides) -> Line2D:
    kwargs = line_style(method, **overrides)
    return Line2D([], [], **kwargs)


def figure_legend(fig: Figure, handles: Sequence, labels: Sequence[str] | None = None, ncol: int = 4) -> None:
    """One legend for the whole figure, above the panels (never over the data)."""
    labels = labels if labels is not None else [h.get_label() for h in handles]
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol, frameon=False)


def direct_label(ax: Axes, x: float, y: float, text: str, dx: float = 3, dy: float = 0, **kwargs) -> None:
    """Small label next to a data point, in secondary ink."""
    kwargs.setdefault("fontsize", 6)
    kwargs.setdefault("color", INK_SECONDARY)
    kwargs.setdefault("va", "center")
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", **kwargs)


def label_fixed_periods(
    ax: Axes, xs: Iterable[float], ys: Iterable[float], periods: Iterable[float], dx: float = 3, dy: float = 2
) -> None:
    for x, y, period in zip(xs, ys, periods):
        direct_label(ax, x, y, f"{period:g} s", dx=dx, dy=dy, va="bottom")


def shade_evaluation(ax: Axes, label: bool = False) -> None:
    """Shade the evaluation episodes (policy frozen) behind the learning curves."""
    start = config.TRAINING_EPISODES + 0.5
    ax.axvspan(start, config.N_EPISODES + 0.5, color=PHASE_SHADE, zorder=0, linewidth=0)
    if label:  # above the plotting area, so it never covers data
        ax.text(0.0, 1.01, "training", transform=ax.transAxes, fontsize=6, color=INK_MUTED, va="bottom")
        ax.text(1.0, 1.01, "evaluation", transform=ax.transAxes, fontsize=6, color=INK_MUTED, va="bottom", ha="right")


def mark_source_switch(ax: Axes, label: bool = False, label_y: float = 0.97) -> None:
    """Dotted vertical line at the source switch; ``label_y`` in axes fraction."""
    ax.axvline(config.SOURCE_SWITCH_S, color=INK_MUTED, linewidth=0.7, linestyle=":", zorder=1)
    if label:
        ax.text(
            config.SOURCE_SWITCH_S,
            label_y,
            " source moves",
            transform=ax.get_xaxis_transform(),
            fontsize=6,
            color=INK_MUTED,
            va="top" if label_y > 0.5 else "bottom",
            ha="left",
        )


def panel_label(ax: Axes, letter: str) -> None:
    """(a), (b), ... in the top-left corner, outside the plotting area."""
    ax.set_title(f"({letter})", loc="left", fontsize=8, pad=2)
