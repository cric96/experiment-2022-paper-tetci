"""Figure families. Each module exposes ``FIGURES``: functions ``(data, out_dir) -> None``."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..dataset import Dataset
from . import dynamics, explain, generalization, learning, spatial, tradeoff

FigureFunction = Callable[[Dataset, Path], None]

MODULES = (learning, tradeoff, dynamics, explain, spatial, generalization)


def all_figures() -> list[FigureFunction]:
    return [figure for module in MODULES for figure in module.FIGURES]
