#!/usr/bin/env python3
"""Generate every figure, LaTeX table and the summary of the paper from the Alchemist exports.

Usage:
    python plotter.py [--exports build/exports] [--out charts] [--only NAME ...] [--no-animation] [--strict]

Outputs (in --out): <figure>.pdf/.png, spatial_wave_animation.gif, tables/*.tex (+ CSV), summary.md.
The analysis code lives in the ``analysis`` package.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

from analysis import config, report, style, tables
from analysis.dataset import load_dataset
from analysis.figures import all_figures

ANIMATIONS = {"spatial_wave_animation"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    figures = [f.__name__ for f in all_figures()]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exports", type=Path, default=config.DEFAULT_EXPORTS_DIR,
                        help="Alchemist export directory (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=config.DEFAULT_OUTPUT_DIR,
                        help="output directory (default: %(default)s)")
    parser.add_argument("--only", nargs="+", choices=figures, metavar="NAME",
                        help=f"render only these figures (skips tables and summary); one of: {', '.join(figures)}")
    parser.add_argument("--no-animation", action="store_true", help="skip the (slow) GIF animation")
    parser.add_argument("--strict", action="store_true", help="turn every warning into an error")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.strict:
        warnings.simplefilter("error")
    style.apply()
    data = load_dataset(args.exports)
    if data.episodes.empty:
        print(f"No episode exports found in {args.exports}", file=sys.stderr)
        return 1
    traces = data.traces.groupby(["method", "weight", "period", "seed"], dropna=False).ngroups
    generalization = data.generalization.groupby(["program", "method", "period", "seed"], dropna=False).ngroups
    print(f"Loaded {len(data.evaluation)} runs, {traces} final-episode traces, {generalization} generalization runs.")
    for figure in all_figures():
        name = figure.__name__
        if (args.only and name not in args.only) or (args.no_animation and name in ANIMATIONS):
            continue
        start = time.perf_counter()
        figure(data, args.out)
        print(f"  figure {name:<34} {time.perf_counter() - start:5.1f} s")
    if not args.only:
        for path in tables.write_tables(data, args.out):
            print(f"  table  {path}")
        print(f"  summary {report.write_summary(data, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
