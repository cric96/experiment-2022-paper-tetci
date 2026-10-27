"""Machine-generated ``summary.md`` with the key numbers of the analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .dataset import Dataset
from .figures.gnn import load_embeddings, wave_front
from .figures.spatial import rounds_distance_correlation
from .methods import FIXED, LEARNED, get
from .pareto import non_dominated
from .results import COLLAPSE_FRACTION, RECOVERY_TOLERANCE_M, fixed_frontier, fixed_metrics_independent_of_weight
from .stats import t_interval

ALPHA = 0.05


def _pct(value: float, digits: int = 0) -> str:
    return f"{value * 100:.{digits}f}%"


def _ci_pct(row: pd.Series) -> str:
    if np.isnan(row.low):
        return _pct(row["mean"])
    return f"{_pct(row['mean'])} [{_pct(row.low)}, {_pct(row.high)}]"


def _config(method: str, weight: float) -> str:
    return f"{get(method).short} (beta={weight:g})"


def _data_section(data: Dataset) -> list[str]:
    evaluation = data.evaluation
    seeds = sorted(int(seed) for seed in evaluation.seed.unique())
    per_config = evaluation.groupby(["method", "weight", "period"], dropna=False).seed.nunique()
    n_seeds = (
        f"{per_config.min()}" if per_config.min() == per_config.max() else f"{per_config.min()}-{per_config.max()}"
    )
    general_seeds = sorted(int(seed) for seed in data.generalization.seed.unique())
    lines = [
        "## Data and protocol",
        "",
        f"- Main experiment: {len(evaluation)} runs; seeds {seeds}; n = {n_seeds} seeds "
        f"per (method, beta) configuration. {config.N_EPISODES} episodes of {config.EPISODE_LENGTH_S:g} s: episodes "
        f"1-{config.TRAINING_EPISODES} train, {config.EVAL_EPISODES[0]}-{config.EVAL_EPISODES[-1]} evaluate "
        "(all main numbers are means over the evaluation episodes, then over seeds).",
        f"- Fixed periods: rounds, error and unstable time identical for every beta: "
        f"{'yes' if fixed_metrics_independent_of_weight(evaluation) else 'NO (check the exports)'} "
        "(only the return depends on beta).",
        f"- Generalization experiment: seeds {general_seeds} only, so it has no across-seed intervals; "
        "beta-conditioned policies are evaluated with 2 episodes per beta_in.",
        f"- Spatial snapshots and embeddings: seed {config.SPATIAL_SEED}, final episode only (illustrative, n = 1).",
        "",
        "## Statistics",
        "",
        "- Means: 95% Student-t intervals over seeds (a percentile bootstrap of 4 values has 35 distinct "
        "resamples and under-covers).",
        "- Ratios (gains, % saved): 95% percentile bootstrap (10,000 resamples) over the seeds of both groups.",
        "- Learned vs. best fixed period of the same beta (highest return): Welch t-test, Holm-Bonferroni over the "
        "20 comparisons of each metric, Hedges' g; Mann-Whitney U as a robustness check.",
        "",
    ]
    return lines


def _energy_section(data: Dataset) -> list[str]:
    gains = data.gains.set_index(["method", "weight", "gain"])
    lines = ["## Energy (rounds) at matched error", ""]
    fastest = fixed_frontier(data.evaluation, "error").iloc[0]
    lines.append(f"Fastest fixed schedule (1 s): {fastest.ticks:.0f} rounds/episode, error {fastest.error:.2f} s.")
    lines.append("")
    as_good = []
    for method in LEARNED:
        for weight in config.WEIGHTS:
            error = gains.loc[(method.key, weight, "error_vs_fastest")]
            if error.high >= 0.0:  # error not significantly worse than the 1 s schedule
                rounds = gains.loc[(method.key, weight, "rounds_vs_fastest")]
                as_good.append((method.key, weight, rounds, error))
    if as_good:
        lines.append(
            "Configurations whose error is not significantly higher than the 1 s schedule's "
            "(upper 95% bound of the error reduction >= 0):"
        )
        lines.append("")
        for method, weight, rounds, error in as_good:
            lines.append(
                f"- {_config(method, weight)}: {_ci_pct(rounds)} fewer rounds, error change "
                f"{_ci_pct(error)} (positive = lower error)."
            )
    else:
        lines.append("No learned configuration matches the error of the 1 s schedule.")
    lines.append("")
    saved = data.gains[data.gains.gain == "rounds_at_error"].dropna(subset=["mean"])
    best = saved.loc[saved["mean"].idxmax()]
    lines.append(
        f"Against the whole fixed-period frontier (interpolated 1-5 s), the largest saving at equal error is "
        f"{_ci_pct(best)} for {_config(best.method, best.weight)}; "
        f"range over the {len(saved)} configurations inside the frontier's error range: "
        f"{_pct(saved['mean'].min())}-{_pct(saved['mean'].max())}."
    )
    lines.append("")
    return lines


def _pareto_section(data: Dataset) -> list[str]:
    gains = data.gains
    error_gain = gains[gains.gain == "error_at_ticks"]
    dominating = error_gain[error_gain.low > 0]
    not_dominating = error_gain[~(error_gain.low > 0)]
    lines = [
        "## Pareto front (rounds vs. error)",
        "",
        f"- {len(dominating)}/{len(error_gain)} learned configurations lie below the fixed-period frontier "
        "(95% bootstrap interval of the error reduction at equal rounds entirely > 0); "
        f"median reduction {_pct(error_gain['mean'].median())}.",
    ]
    if len(not_dominating):
        names = ", ".join(f"{_config(r.method, r.weight)} {_ci_pct(r)}" for _, r in not_dominating.iterrows())
        lines.append(f"- Not distinguishable from the frontier: {names}.")
    learned_means = data.evaluation[data.evaluation.method != FIXED.key].groupby(["method", "weight"]).ticks.mean()
    top_method, top_weight = learned_means.idxmax()
    lines.append(
        f"- Highest-budget learned configuration: {_config(top_method, top_weight)} with "
        f"{learned_means.max():.0f} rounds; between that and the 1 s schedule "
        f"({fixed_frontier(data.evaluation, 'error').ticks.max():.0f}) only fixed periods exist."
    )
    points = data.evaluation.groupby(["method", "weight", "period"], dropna=False)[["ticks", "error"]].mean()
    points = points.reset_index()
    points = points[(points.method != FIXED.key) | np.isclose(points.weight, config.FOCUS_WEIGHT)]
    front = points[non_dominated(points.ticks.to_numpy(), points.error.to_numpy())].sort_values("ticks")
    names = []
    for row in front.itertuples():
        names.append(f"fixed {row.period:g} s" if row.method == FIXED.key else _config(row.method, row.weight))
    lines.append(
        f"- Global non-dominated set (means, all methods and fixed periods), by increasing rounds: {'; '.join(names)}."
    )
    lines.append("")
    return lines


def _significance_section(data: Dataset) -> list[str]:
    table = data.significance
    lines = ["## Learned policies vs. best fixed period (same beta)", ""]
    for metric, better in (("reward", "higher"), ("error", "lower")):
        subset = table[table.metric == metric]
        improved = subset[(subset.welch_p_holm < ALPHA) & ((subset.difference > 0) == (better == "higher"))]
        worse = subset[(subset.welch_p_holm < ALPHA) & ((subset.difference > 0) != (better == "higher"))]
        lines.append(
            f"- {metric}: significantly {better} in {len(improved)}/{len(subset)} comparisons, "
            f"significantly worse in {len(worse)} (Welch, Holm-adjusted p < {ALPHA})."
        )
    by_beta = (
        table[table.metric == "reward"]
        .groupby("weight")
        .apply(lambda g: int(((g.welch_p_holm < ALPHA) & (g.difference > 0)).sum()), include_groups=False)
    )
    lines.append("- Return wins per beta: " + ", ".join(f"beta={w:g}: {n}/4" for w, n in by_beta.items()) + ".")
    lines.append(
        f"- Mann-Whitney U: smallest two-sided p = {table.mwu_p.min():.3f} (complete separation of 4 vs. 4 "
        f"seeds), smallest Holm-adjusted p = {table.mwu_p_holm.min():.2f}: with 4 seeds a rank test cannot "
        "reach significance after correction; more seeds are needed for non-parametric claims."
    )
    lines.append("")
    return lines


def _stability_section(data: Dataset) -> list[str]:
    stability = data.stability
    collapsed = stability[stability.n_collapsed > 0]
    lines = ["## Seed stability", ""]
    if len(collapsed):
        names = ", ".join(f"{_config(r.method, r.weight)}: {r.n_collapsed}/{r.n}" for r in collapsed.itertuples())
        lines.append(
            f"- Seeds collapsed to the fastest schedule (>= {COLLAPSE_FRACTION:.0%} of the 1 s rounds): {names}."
        )
    stable = stability.groupby("method").ticks_cv.max()
    lines.append(
        "- Largest across-seed CV of the rounds per method: "
        + ", ".join(f"{get(m).short} {_pct(cv)}" for m, cv in stable.items())
        + "."
    )
    lines.append("")
    return lines


def _switch_section(data: Dataset) -> list[str]:
    switch = data.switch
    lines = [f"## Reaction to the source switch (final episode, beta = {config.FOCUS_WEIGHT:g})", ""]
    for method in LEARNED:
        runs = switch[switch.method == method.key]
        ratio, recovery = t_interval(runs.rate_ratio), t_interval(runs.recovery_time)
        steady, low = t_interval(runs.steady_period), t_interval(runs.min_period)
        lines.append(
            f"- {method.short}: period {steady.mean:.2f} s -> {low.mean:.2f} s after the switch, "
            f"round rate x{ratio.mean:.2f} [{ratio.low:.2f}, {ratio.high:.2f}], re-converged after "
            f"{recovery.mean:.1f} s [{recovery.low:.1f}, {recovery.high:.1f}]."
        )
    fixed = switch[switch.method == FIXED.key].groupby("period").recovery_time.agg(["mean", "count"])
    lines.append(
        "- Fixed periods, re-convergence time: "
        + ", ".join(f"{p:g} s: {r['mean']:.1f} s" for p, r in fixed.iterrows())
        + f" (tolerance |mean g - mean g*| < {RECOVERY_TOLERANCE_M:g} m; unrecovered seeds excluded)."
    )
    lines.append("")
    return lines


def _generalization_section(data: Dataset) -> list[str]:
    gains = data.generalization_gains
    programs = tuple(config.PROGRAMS.values())
    per = gains.groupby(["method", "program"]).error_at_ticks.mean().unstack()
    lines = [
        "## Zero-shot generalization (1 seed)",
        "",
        "S followed by G broadcasts the nearest elected leader's id (categorical mismatch error), "
        "using grain 400 m and removal of an elected leader at 100 s; the other programs use a source move. "
        "Failed-node counters are retained and cumulative means use the initial population.",
        "",
        "Error reduction at equal rounds w.r.t. each program's own fixed-period frontier (mean over beta_in):",
        "",
    ]
    for method, row in per.iterrows():
        gap = row[programs[0]] - np.nanmean([row[p] for p in programs[1:]])
        cells = ", ".join(f"{config.PROGRAM_SHORT_LABELS[p]} {_pct(row[p])}" for p in programs)
        lines.append(f"- {get(method).short}: {cells}; generalization gap {gap * 100:.0f} points.")
    points = data.generalization_points
    fastest = points[(points.method != FIXED.key) & (points.ticks >= COLLAPSE_FRACTION * points.ticks.max())]
    for method, group in fastest.groupby("method"):
        cases = "; ".join(
            f"{program} (beta_in {', '.join(f'{b:g}' for b in sorted(g.beta_in))})"
            for program, g in group.groupby("program")
        )
        lines.append(f"- {get(method).short} degenerates to (almost) the 1 s period on: {cases}.")
    lines.append("")
    return lines


def _spatial_section(data: Dataset) -> list[str]:
    lines = ["## Spatial and embedding analyses (seed 2, illustrative)", ""]
    correlation = rounds_distance_correlation(data)
    if correlation is not None:
        rho, p = correlation
        lines.append(
            f"- Rounds per node vs. distance to the nearest source (TD3, final episode): Spearman rho = "
            f"{rho:.2f} (p = {p:.2g}); no evidence that rounds concentrate near the sources over a whole "
            "episode (the extra rounds are transient, see the filmstrip)."
        )
    frame = load_embeddings(data)
    if frame is not None:
        front = wave_front(frame)
        lines.append(
            f"- GNN embedding wave front after the switch: {front.speed:.1f} m/s "
            f"[{front.low:.1f}, {front.high:.1f}] (r = {front.r:.2f}, onset {front.onset:.1f} s), "
            "fitted on the per-node time of peak PC1 (replaces the hand-drawn 21 m/s line)."
        )
    lines.append("")
    return lines


def write_summary(data: Dataset, out_dir: Path) -> Path:
    """Write ``out_dir/summary.md``."""
    sections = [
        ["# Analysis summary (generated by plotter.py)", ""],
        _data_section(data),
        _energy_section(data),
        _pareto_section(data),
        _significance_section(data),
        _stability_section(data),
        _switch_section(data),
        _generalization_section(data),
        _spatial_section(data),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.md"
    path.write_text("\n".join(line for section in sections for line in section))
    return path
