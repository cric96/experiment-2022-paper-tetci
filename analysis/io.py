"""Parsing of Alchemist CSV exports.

An Alchemist export is a whitespace-separated table preceded by ``#`` comment
lines. Comment lines hold the values of the simulation variables
(e.g. ``# random = 2.0, weight = 0.95``) and the last comment line before the
data holds the column names. The file name prefix (before the first ``_``)
identifies the experiment, e.g. ``experimentDeepOfflineGNN``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_VARIABLE = re.compile(r"([A-Za-z0-9._-]+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_BANNERS = ("started at", "finished at")


@dataclass
class AlchemistRun:
    """One simulation run: its variables and its exported time series."""

    path: Path
    experiment: str
    params: dict[str, float] = field(default_factory=dict)
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def param(self, name: str, default: float = float("nan")) -> float:
        return self.params.get(name, default)


def experiment_name(path: Path) -> str:
    """Experiment identifier encoded as the file-name prefix."""
    return Path(path).name.split("_")[0]


def parse_alchemist_csv(path: Path | str) -> AlchemistRun:
    """Parse one Alchemist CSV export into an :class:`AlchemistRun`."""
    path = Path(path)
    params: dict[str, float] = {}
    columns: list[str] | None = None
    last_comment = ""
    rows: list[list[float]] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if not any(banner in line for banner in _BANNERS):
                    params.update({k: float(v) for k, v in _VARIABLE.findall(line)})
                last_comment = line.lstrip("#").strip()
                continue
            if columns is None:
                columns = last_comment.split()
            rows.append([float(token) for token in line.split()])
    data = pd.DataFrame(rows, columns=columns or [])
    return AlchemistRun(path=path, experiment=experiment_name(path), params=params, data=data)


def load_runs(directory: Path | str, pattern: str = "*.csv") -> list[AlchemistRun]:
    """Parse every export in ``directory`` (sorted by name); empty if missing."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    runs = [parse_alchemist_csv(p) for p in sorted(directory.glob(pattern))]
    return [run for run in runs if not run.data.empty]
