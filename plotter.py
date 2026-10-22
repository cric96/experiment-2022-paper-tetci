#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 220,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.2,
})

EPISODE_DIR = "build/exports/simulation/episode"
FULL_DIRS = [
    "build/exports/simulation/full",
    "build/exports/simulationDeepOffline/full",
    "build/exports/simulationDeepOfflineGNN/full",
]
OUTPUT_DIR = "charts"


def parse_alchemist_csv(filepath):
    coords = {}
    columns = []
    rows = []
    coord_regex = re.compile(r"([a-zA-Z0-9._-]+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    with open(filepath, "r") as f:
        last_comment = ""
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                matches = coord_regex.findall(stripped)
                if matches and " started at" not in stripped and " finished at" not in stripped:
                    for k, v in matches:
                        coords[k] = float(v)
                last_comment = stripped.lstrip("#").strip()
            elif stripped[0].isdigit() or stripped[0] == "-":
                if not columns and last_comment:
                    columns = last_comment.split()
                vals = [float(x) for x in stripped.split()]
                rows.append(vals)
    data = np.array(rows, dtype=float) if rows else np.empty((0, len(columns)))
    return coords, columns, data


def load_all_exports(directory):
    records = []
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        fname = os.path.basename(path)
        prefix = fname.split("_")[0]
        coords, cols, data = parse_alchemist_csv(path)
        if data.size > 0 and len(cols) > 0:
            records.append({
                "path": path,
                "prefix": prefix,
                "coords": coords,
                "cols": cols,
                "data": data,
            })
    return records


def get_col(rec, name):
    idx = rec["cols"].index(name)
    return rec["data"][:, idx]


def aggregate_episode_series(records, col_name):
    ep_grid = np.arange(1, 21)
    curves = []
    for rec in records:
        if col_name not in rec["cols"]:
            continue
        t = get_col(rec, "time")
        vals = get_col(rec, col_name)
        mask = t >= 190.0
        eps = np.round(t[mask] / 200.0).astype(int)
        v = vals[mask]
        aligned = np.interp(ep_grid, eps, v)
        curves.append(aligned)
    if not curves:
        return ep_grid, None, None, None
    mat = np.vstack(curves)
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    sem = std / np.sqrt(mat.shape[0])
    return ep_grid, mean, std, sem


def plot_learning_curves(episode_records):
    weights_to_show = [1.0, 0.95, 0.8, 0.5]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), sharex=True)
    axes = axes.flatten()

    method_styles = [
        ("experimentDeepOfflineGNN", "EdgeGAT GNN + Vec-MORL", "#d62728", "-*"),
        ("experimentDeepOfflineContinuous", "Continuous MLP RL (TD3+BC)", "#ff7f0e", "-^"),
        ("experimentDeepOffline", "Discrete BCQ (Offline RL)", "#2ca02c", "-s"),
        ("experimentTable", "Tabular Q-Learning", "#1f77b4", "-o"),
    ]
    baseline_styles = [
        (1.0, "Baseline (Δt = 1.0s, fast)", "#666666", "--"),
        (3.0, "Baseline (Δt = 3.0s, medium)", "#9467bd", ":"),
        (5.0, "Baseline (Δt = 5.0s, slow)", "#8c564b", "-."),
    ]

    for ax, w in zip(axes, weights_to_show):
        ax.axvspan(10.5, 20.5, color="#f0f4f8", alpha=0.55, zorder=0)
        for prefix, label, color, style in method_styles:
            matching = [
                r for r in episode_records
                if r["prefix"] == prefix and abs(r["coords"].get("weight", -1) - w) < 1e-6
            ]
            eps, mean, std, sem = aggregate_episode_series(matching, "accumulatedReward[mean]")
            if mean is None:
                continue
            ax.plot(eps, mean, style, color=color, label=label, markersize=5, zorder=3)
            ax.fill_between(eps, mean - sem, mean + sem, color=color, alpha=0.18, zorder=2)

        for act_val, bl_label, color, ls in baseline_styles:
            matching = [
                r for r in episode_records
                if r["prefix"] == "experimentBaseline"
                and abs(r["coords"].get("weight", -1) - w) < 1e-6
                and abs(r["coords"].get("actions", -1) - act_val) < 1e-6
            ]
            eps, mean, std, sem = aggregate_episode_series(matching, "accumulatedReward[mean]")
            if mean is None:
                continue
            ax.plot(eps, mean, linestyle=ls, color=color, label=bl_label, linewidth=1.7, alpha=0.85, zorder=2)

        ax.axvline(10.5, color="#333333", linestyle="--", alpha=0.7, linewidth=1.3)
        ax.set_title(f"Trade-off Weight β = {w}")
        ax.set_ylabel("Accumulated Reward (mean ± SEM)")
        ax.set_xlim(1, 20)
        ax.set_xticks([1, 5, 10, 15, 20])

    axes[2].set_xlabel("Episode (200s each)")
    axes[3].set_xlabel("Episode (200s each)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Multi-Seed Learning Curves Across Episodes (Shaded Area = Exploitation Phase, Ep. 11–20)",
        y=0.99,
        fontsize=13.5,
        fontweight="bold",
    )
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.945))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"learning_curves_reward.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_ticks_and_unstable_curves(episode_records):
    weights_to_show = [0.95, 0.8]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.8), sharex=True)

    method_styles = [
        ("experimentDeepOfflineGNN", "EdgeGAT GNN + Vec-MORL", "#d62728", "-*"),
        ("experimentDeepOfflineContinuous", "Continuous MLP RL (TD3+BC)", "#ff7f0e", "-^"),
        ("experimentDeepOffline", "Discrete BCQ (Offline RL)", "#2ca02c", "-s"),
        ("experimentTable", "Tabular Q-Learning", "#1f77b4", "-o"),
    ]
    baseline_styles = [
        (1.0, "Baseline (Δt = 1.0s)", "#666666", "--"),
        (3.0, "Baseline (Δt = 3.0s)", "#9467bd", ":"),
        (5.0, "Baseline (Δt = 5.0s)", "#8c564b", "-."),
    ]

    metrics = [
        ("ticks[sum]", "Total Executions (Ticks)", "Ticks per Episode (↓)"),
        ("accumulatedUnstableTime[mean]", "Unstable Time (s)", "Unstable Time [s] (↓)"),
        ("accumulatedError[mean]", "Integrated Error (AUEC)", "Integrated Error AUEC (↓)"),
    ]

    for row_idx, w in enumerate(weights_to_show):
        for col_idx, (col_name, title_prefix, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            ax.axvspan(10.5, 20.5, color="#f0f4f8", alpha=0.55, zorder=0)

            for prefix, label, color, style in method_styles:
                matching = [
                    r for r in episode_records
                    if r["prefix"] == prefix and abs(r["coords"].get("weight", -1) - w) < 1e-6
                ]
                eps, mean, std, sem = aggregate_episode_series(matching, col_name)
                if mean is None:
                    continue
                ax.plot(eps, mean, style, color=color, label=label, markersize=4.5, zorder=3)
                ax.fill_between(eps, mean - sem, mean + sem, color=color, alpha=0.18, zorder=2)

            for act_val, bl_label, color, ls in baseline_styles:
                matching = [
                    r for r in episode_records
                    if r["prefix"] == "experimentBaseline"
                    and abs(r["coords"].get("weight", -1) - w) < 1e-6
                    and abs(r["coords"].get("actions", -1) - act_val) < 1e-6
                ]
                eps, mean, std, sem = aggregate_episode_series(matching, col_name)
                if mean is None:
                    continue
                ax.plot(eps, mean, linestyle=ls, color=color, label=bl_label, linewidth=1.6, alpha=0.85, zorder=2)

            ax.axvline(10.5, color="#333333", linestyle="--", alpha=0.7, linewidth=1.2)
            ax.set_title(f"{title_prefix} — β = {w}")
            ax.set_ylabel(ylabel)
            ax.set_xlim(1, 20)
            ax.set_xticks([1, 5, 10, 15, 20])
            if row_idx == 1:
                ax.set_xlabel("Episode")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Multi-Seed Evolution of Ticks, Unstable Time, and Integrated Error (Mean ± SEM Over 4 Seeds)",
        y=0.99,
        fontsize=13.5,
        fontweight="bold",
    )
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.945))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"learning_curves_ticks_unstable.{ext}"), bbox_inches="tight")
    plt.close(fig)


def collect_method_stats(episode_records, prefix, weights):
    w_vals = []
    tick_m, tick_s = [], []
    unst_m, unst_s = [], []
    err_m, err_s = [], []
    rew_m, rew_s = [], []
    for w in weights:
        matching = [
            r for r in episode_records
            if r["prefix"] == prefix and abs(r["coords"].get("weight", -1) - w) < 1e-6
        ]
        if not matching:
            continue
        t_l, u_l, e_l, r_l = [], [], [], []
        for rec in matching:
            t = get_col(rec, "time")
            mask = t >= 2200.0
            t_l.append(np.mean(get_col(rec, "ticks[sum]")[mask]))
            u_l.append(np.mean(get_col(rec, "accumulatedUnstableTime[mean]")[mask]))
            e_l.append(np.mean(get_col(rec, "accumulatedError[mean]")[mask]))
            r_l.append(np.mean(get_col(rec, "accumulatedReward[mean]")[mask]))
        n = max(1, len(matching))
        w_vals.append(w)
        tick_m.append(np.mean(t_l))
        tick_s.append(np.std(t_l) / np.sqrt(n))
        unst_m.append(np.mean(u_l))
        unst_s.append(np.std(u_l) / np.sqrt(n))
        err_m.append(np.mean(e_l))
        err_s.append(np.std(e_l) / np.sqrt(n))
        rew_m.append(np.mean(r_l))
        rew_s.append(np.std(r_l) / np.sqrt(n))
    return (
        np.array(w_vals),
        np.array(tick_m),
        np.array(tick_s),
        np.array(unst_m),
        np.array(unst_s),
        np.array(err_m),
        np.array(err_s),
        np.array(rew_m),
        np.array(rew_s),
    )


def collect_baseline_stats(episode_records, actions=(1.0, 2.0, 3.0, 4.0, 5.0), ref_weight=0.8):
    a_vals = []
    tick_m, tick_s = [], []
    unst_m, unst_s = [], []
    err_m, err_s = [], []
    for a in actions:
        matching = [
            r for r in episode_records
            if r["prefix"] == "experimentBaseline"
            and abs(r["coords"].get("weight", -1) - ref_weight) < 1e-6
            and abs(r["coords"].get("actions", -1) - a) < 1e-6
        ]
        if not matching:
            continue
        t_l, u_l, e_l = [], [], []
        for rec in matching:
            t = get_col(rec, "time")
            mask = t >= 2200.0
            t_l.append(np.mean(get_col(rec, "ticks[sum]")[mask]))
            u_l.append(np.mean(get_col(rec, "accumulatedUnstableTime[mean]")[mask]))
            e_l.append(np.mean(get_col(rec, "accumulatedError[mean]")[mask]))
        n = max(1, len(matching))
        a_vals.append(a)
        tick_m.append(np.mean(t_l))
        tick_s.append(np.std(t_l) / np.sqrt(n))
        unst_m.append(np.mean(u_l))
        unst_s.append(np.std(u_l) / np.sqrt(n))
        err_m.append(np.mean(e_l))
        err_s.append(np.std(e_l) / np.sqrt(n))
    return (
        np.array(a_vals),
        np.array(tick_m),
        np.array(tick_s),
        np.array(unst_m),
        np.array(unst_s),
        np.array(err_m),
        np.array(err_s),
    )


def plot_tradeoff_and_pareto(episode_records):
    weights = [0.5, 0.8, 0.95, 0.99, 1.0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    axes = axes.flatten()

    methods = [
        ("experimentDeepOfflineGNN", "EdgeGAT GNN + Vec-MORL", "#d62728", "-*"),
        ("experimentDeepOfflineContinuous", "Continuous MLP RL (TD3+BC)", "#ff7f0e", "-^"),
        ("experimentDeepOffline", "Discrete BCQ (Offline RL)", "#2ca02c", "-s"),
        ("experimentTable", "Tabular Q-Learning", "#1f77b4", "-o"),
    ]

    for prefix, label, color, style in methods:
        w_vals, tick_m, tick_s, unst_m, unst_s, err_m, err_s, _, _ = collect_method_stats(
            episode_records, prefix, weights
        )
        axes[0].errorbar(w_vals, tick_m, yerr=tick_s, fmt=style, color=color, label=label, markersize=6, capsize=3)
        axes[1].errorbar(w_vals, unst_m, yerr=unst_s, fmt=style, color=color, label=label, markersize=6, capsize=3)
        axes[2].errorbar(w_vals, err_m, yerr=err_s, fmt=style, color=color, label=label, markersize=6, capsize=3)
        axes[3].errorbar(tick_m, err_m, xerr=tick_s, yerr=err_s, fmt=style, color=color, label=label, markersize=6, capsize=3)

    a_vals, bl_ticks, bl_ticks_s, _, _, bl_err, bl_err_s = collect_baseline_stats(episode_records)
    for a, tx, ey in zip(a_vals, bl_ticks, bl_err):
        axes[3].annotate(f"Δt={int(a)}s", (tx, ey), textcoords="offset points", xytext=(7, 3), fontsize=9, fontweight="bold", color="#333333")

    if len(bl_ticks) > 0:
        axes[3].errorbar(
            bl_ticks,
            bl_err,
            xerr=bl_ticks_s,
            yerr=bl_err_s,
            fmt="--D",
            color="#555555",
            label="Fixed-Period Baselines (Δt=1..5s)",
            markersize=5.5,
            capsize=3,
        )

    axes[0].set_title("Communication Cost (Ticks) vs Weight β")
    axes[0].set_xlabel("Weight β (weightForConvergence)")
    axes[0].set_ylabel("Mean Ticks per Episode (± SEM)")

    axes[1].set_title("Unstable Time vs Weight β")
    axes[1].set_xlabel("Weight β (weightForConvergence)")
    axes[1].set_ylabel("Mean Unstable Time [s] (± SEM)")

    axes[2].set_title("Integrated Gradient Error (AUEC) vs Weight β")
    axes[2].set_xlabel("Weight β (weightForConvergence)")
    axes[2].set_ylabel("Mean Integrated Error AUEC (± SEM)")

    axes[3].set_title("Pareto Frontier: Ticks vs Integrated Error (AUEC)")
    axes[3].set_xlabel("Mean Ticks per Episode (↓ Less Energy)")
    axes[3].set_ylabel("Mean Integrated Error AUEC (↓ More Accurate)")

    handles, labels = axes[3].get_legend_handles_labels()
    fig.suptitle(
        "Multi-Seed Exploitation Trade-off & Pareto Frontier Analysis (4 Seeds, Episodes 11–20)",
        y=0.99,
        fontsize=13.5,
        fontweight="bold",
    )
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.945))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"tradeoff_pareto_weight.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_dual_pareto_frontiers(episode_records):
    """Detailed side-by-side Pareto Frontiers (Ticks vs Unstable Time & Ticks vs AUEC) with annotations."""
    weights = [0.5, 0.8, 0.95, 0.99, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    a_vals, bl_ticks, bl_ticks_s, bl_unst, bl_unst_s, bl_err, bl_err_s = collect_baseline_stats(episode_records)

    methods = [
        ("experimentDeepOfflineGNN", "EdgeGAT GNN + Vec-MORL", "#d62728", "-*"),
        ("experimentDeepOfflineContinuous", "Continuous MLP RL (TD3+BC)", "#ff7f0e", "-^"),
        ("experimentDeepOffline", "Discrete BCQ (Offline RL)", "#2ca02c", "-s"),
        ("experimentTable", "Tabular Q-Learning", "#1f77b4", "-o"),
    ]

    for prefix, label, color, style in methods:
        w_vals, tick_m, tick_s, unst_m, unst_s, err_m, err_s, _, _ = collect_method_stats(
            episode_records, prefix, weights
        )
        axes[0].errorbar(tick_m, unst_m, xerr=tick_s, yerr=unst_s, fmt=style, color=color, label=label, markersize=7, capsize=3, zorder=4)
        axes[1].errorbar(tick_m, err_m, xerr=tick_s, yerr=err_s, fmt=style, color=color, label=label, markersize=7, capsize=3, zorder=4)

        if prefix == "experimentDeepOfflineContinuous":
            for w, tx, uy, ey in zip(w_vals, tick_m, unst_m, err_m):
                axes[0].annotate(
                    f"β={w}",
                    (tx, uy),
                    textcoords="offset points",
                    xytext=(-12, -15),
                    fontsize=8.5,
                    fontweight="bold",
                    color=color,
                )
                axes[1].annotate(
                    f"β={w}",
                    (tx, ey),
                    textcoords="offset points",
                    xytext=(-12, -15),
                    fontsize=8.5,
                    fontweight="bold",
                    color=color,
                )

    if len(bl_ticks) > 0:
        axes[0].errorbar(
            bl_ticks, bl_unst, xerr=bl_ticks_s, yerr=bl_unst_s,
            fmt="--D", color="#444444", label="Fixed-Period Baselines (Δt=1..5s)", markersize=6, capsize=3, zorder=3
        )
        axes[1].errorbar(
            bl_ticks, bl_err, xerr=bl_ticks_s, yerr=bl_err_s,
            fmt="--D", color="#444444", label="Fixed-Period Baselines (Δt=1..5s)", markersize=6, capsize=3, zorder=3
        )
        for a, tx, uy, ey in zip(a_vals, bl_ticks, bl_unst, bl_err):
            axes[0].annotate(f"Δt={int(a)}s", (tx, uy), textcoords="offset points", xytext=(8, 3), fontsize=9, fontweight="bold", color="#333333")
            axes[1].annotate(f"Δt={int(a)}s", (tx, ey), textcoords="offset points", xytext=(8, 3), fontsize=9, fontweight="bold", color="#333333")

        # Highlight the β=0.95 sweet spot gain with callout arrows
        _, c_ticks, _, c_unst, _, c_err, _, _, _ = collect_method_stats(
            episode_records, "experimentDeepOfflineContinuous", [0.95]
        )
        if len(c_ticks) > 0:
            tx95, uy95, ey95 = c_ticks[0], c_unst[0], c_err[0]
            # Interpolate baseline at same ticks (Note: bl_ticks is descending from 19900 to 4000)
            order = np.argsort(bl_ticks)
            bl_u_at_tx = np.interp(tx95, bl_ticks[order], bl_unst[order])
            bl_e_at_tx = np.interp(tx95, bl_ticks[order], bl_err[order])
            axes[0].annotate(
                "", xy=(tx95, uy95 + 0.3), xytext=(tx95, bl_u_at_tx - 0.3),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.8, ls="--")
            )
            axes[0].text(
                tx95 + 650, (uy95 + bl_u_at_tx) / 2.0 + 0.8,
                f"−{int(round((1 - uy95 / bl_u_at_tx) * 100))}% Unstable Time\nat same Ticks",
                color="#b01010", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#d62728", alpha=0.88)
            )
            axes[1].annotate(
                "", xy=(tx95, ey95 + 0.5), xytext=(tx95, bl_e_at_tx - 0.5),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.8, ls="--")
            )
            axes[1].text(
                tx95 + 650, (ey95 + bl_e_at_tx) / 2.0 + 1.2,
                f"−{int(round((1 - ey95 / bl_e_at_tx) * 100))}% Error (AUEC)\nat same Ticks",
                color="#b01010", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#d62728", alpha=0.88)
            )

    axes[0].set_title("Pareto Frontier 1: Communication Cost vs Unstable Time")
    axes[0].set_xlabel("Mean Ticks per Episode (↓ Lower Energy / Messages)")
    axes[0].set_ylabel("Mean Unstable Time [s] (↓ Faster Convergence)")

    axes[1].set_title("Pareto Frontier 2: Communication Cost vs Integrated Error (AUEC)")
    axes[1].set_xlabel("Mean Ticks per Episode (↓ Lower Energy / Messages)")
    axes[1].set_ylabel("Mean Integrated Gradient Error AUEC (↓ Higher Accuracy)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Pareto Dominance of Adaptive RL Scheduling Over Fixed-Period Baselines (4 Seeds, Mean ± SEM)",
        y=1.01,
        fontsize=13.5,
        fontweight="bold",
    )
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"01_pareto_frontiers_dual.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_instantaneous_error_zoom(full_records):
    """Zoom into t in [80, 165] around the t=100s Source Switch showing instantaneous error |G(t) - G*(t)| and Δt(t)."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.8), sharex=True)
    t_grid = np.linspace(80.0, 165.0, 171)

    configs = [
        ("experimentBaseline", 5.0, 0.95, "Baseline (Δt = 5.0s, slow)", "#8c564b", "-."),
        ("experimentBaseline", 3.0, 0.95, "Baseline (Δt = 3.0s, medium)", "#9467bd", ":"),
        ("experimentTable", None, 0.95, "Tabular Q-Learning (β = 0.95)", "#1f77b4", "--"),
        ("experimentDeepOffline", None, 0.95, "Discrete BCQ (β = 0.95)", "#2ca02c", "-."),
        ("experimentDeepOfflineContinuous", None, 0.95, "Continuous MLP RL (β = 0.95)", "#ff7f0e", "--"),
        ("experimentDeepOfflineGNN", None, 0.95, "EdgeGAT GNN + Vec-MORL (β = 0.95)", "#d62728", "-"),
    ]

    for prefix, act_val, w, label, color, ls in configs:
        if act_val is not None:
            matching = [
                r for r in full_records
                if r["prefix"] == prefix
                and abs(r["coords"].get("weight", -1) - w) < 1e-6
                and abs(r["coords"].get("actions", -1) - act_val) < 1e-6
            ]
        else:
            matching = [
                r for r in full_records
                if r["prefix"] == prefix
                and abs(r["coords"].get("weight", -1) - w) < 1e-6
            ]
        if not matching:
            continue

        err_curves, wake_curves = [], []
        for rec in matching:
            t = get_col(rec, "time")
            wakeup = get_col(rec, "nextWakeUp[mean]")
            grad_sum = get_col(rec, "localComputation[sum]")
            gt_sum = get_col(rec, "groundTruth[sum]")
            mask = (t >= 3800.0) & (t <= 4000.5)
            t_ep = t[mask] - 3800.0
            inst_err = np.abs(grad_sum[mask] - gt_sum[mask]) / 100.0
            err_curves.append(np.interp(t_grid, t_ep, inst_err))
            wake_curves.append(np.interp(t_grid, t_ep, wakeup[mask]))

        e_mat = np.vstack(err_curves)
        w_mat = np.vstack(wake_curves)
        e_mean, e_sem = np.mean(e_mat, axis=0), np.std(e_mat, axis=0) / np.sqrt(e_mat.shape[0])
        w_mean, w_sem = np.mean(w_mat, axis=0), np.std(w_mat, axis=0) / np.sqrt(w_mat.shape[0])

        axes[0].plot(t_grid, e_mean, linestyle=ls, color=color, label=label, linewidth=2.2)
        axes[0].fill_between(t_grid, np.maximum(0, e_mean - e_sem), e_mean + e_sem, color=color, alpha=0.15)

        axes[1].plot(t_grid, w_mean, linestyle=ls, color=color, label=label, linewidth=2.2)
        axes[1].fill_between(t_grid, w_mean - w_sem, w_mean + w_sem, color=color, alpha=0.15)

    for ax in axes:
        ax.axvline(100.0, color="#222222", linestyle="--", linewidth=1.6, label="Source Switch (t = 100s)")
        ax.set_xlim(80, 165)

    axes[0].set_title("Transient Network Gradient Error |g(t) − g*(t)| After Source Switch at t = 100s (Mean ± SEM, 4 Seeds)")
    axes[0].set_ylabel("Mean Error per Node (m)")
    axes[0].legend(loc="upper right", frameon=True)

    axes[1].set_title("Corresponding Adaptive Wake-Up Period Δt(t) Around the Source Switch (β = 0.95)")
    axes[1].set_ylabel("Mean Next Wake-Up Δt (s)")
    axes[1].set_xlabel("Time Within Episode (s)")
    axes[1].set_ylim(0.5, 5.3)
    axes[1].legend(loc="lower right", ncol=2, frameon=True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"02_instantaneous_error_and_wakeup.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_multi_beta_intra_episode(full_records):
    """Shows how Continuous Offline RL modulates Δt(t) within an episode as β varies from 0.5 to 1.0."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.6), sharex=True)
    t_grid = np.linspace(0.0, 200.0, 101)

    beta_styles = [
        (0.5, "β = 0.50 (Energy Saving Priority)", "#1f77b4", "-"),
        (0.8, "β = 0.80 (Balanced)", "#2ca02c", "--"),
        (0.95, "β = 0.95 (High Reactivity)", "#d62728", "-"),
        (0.99, "β = 0.99 (Very High Reactivity)", "#9467bd", "-."),
        (1.0, "β = 1.00 (Max Convergence Speed)", "#ff7f0e", ":"),
    ]

    for w, label, color, ls in beta_styles:
        matching = [
            r for r in full_records
            if r["prefix"] == "experimentDeepOfflineContinuous"
            and abs(r["coords"].get("weight", -1) - w) < 1e-6
        ]
        if not matching:
            continue
        wake_curves, err_curves = [], []
        for rec in matching:
            t = get_col(rec, "time")
            wakeup = get_col(rec, "nextWakeUp[mean]")
            grad_sum = get_col(rec, "localComputation[sum]")
            gt_sum = get_col(rec, "groundTruth[sum]")
            mask = (t >= 3800.0) & (t <= 4000.5)
            t_ep = t[mask] - 3800.0
            inst_err = np.abs(grad_sum[mask] - gt_sum[mask]) / 100.0
            wake_curves.append(np.interp(t_grid, t_ep, wakeup[mask]))
            err_curves.append(np.interp(t_grid, t_ep, inst_err))

        w_mat = np.vstack(wake_curves)
        e_mat = np.vstack(err_curves)
        w_mean, w_sem = np.mean(w_mat, axis=0), np.std(w_mat, axis=0) / np.sqrt(w_mat.shape[0])
        e_mean, e_sem = np.mean(e_mat, axis=0), np.std(e_mat, axis=0) / np.sqrt(e_mat.shape[0])

        axes[0].plot(t_grid, w_mean, linestyle=ls, color=color, label=label, linewidth=2.2)
        axes[0].fill_between(t_grid, w_mean - w_sem, w_mean + w_sem, color=color, alpha=0.14)

        axes[1].plot(t_grid, e_mean, linestyle=ls, color=color, label=label, linewidth=2.2)
        axes[1].fill_between(t_grid, np.maximum(0, e_mean - e_sem), e_mean + e_sem, color=color, alpha=0.14)

    for ax in axes:
        ax.axvline(100.0, color="#222222", linestyle="--", linewidth=1.5)
        ax.set_xlim(0, 200)

    axes[0].set_title("Effect of Preference Weight β on Continuous Offline RL Wake-Up Schedule Δt(t) (4 Seeds ± SEM)")
    axes[0].set_ylabel("Mean Next Wake-Up Δt (s)")
    axes[0].set_ylim(0.5, 5.2)
    axes[0].legend(loc="lower right", ncol=2, frameon=True)

    axes[1].set_title("Resulting Instantaneous Gradient Error |g(t) − g*(t)| Across Preference Weights β")
    axes[1].set_ylabel("Mean Error per Node (m)")
    axes[1].set_xlabel("Time Within Episode (s)")
    axes[1].legend(loc="upper right", ncol=2, frameon=True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"03_multi_beta_intra_episode.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_gains(episode_records):
    """2-panel Heatmap: (Left) % Error Reduction vs Equal-Tick Fixed Baseline, (Right) % Unstable Time Reduction."""
    weights = [0.5, 0.8, 0.95, 0.99, 1.0]
    methods = [
        ("experimentTable", "Tabular Q-Learning"),
        ("experimentDeepOffline", "Discrete BCQ\n(Offline RL)"),
        ("experimentDeepOfflineContinuous", "Continuous MLP RL\n(TD3+BC)"),
        ("experimentDeepOfflineGNN", "EdgeGAT GNN\n+ Vec-MORL"),
    ]

    a_vals, bl_ticks, _, bl_unst, _, bl_err, _ = collect_baseline_stats(episode_records)
    order = np.argsort(bl_ticks)
    bl_t_sorted = bl_ticks[order]
    bl_u_sorted = bl_unst[order]
    bl_e_sorted = bl_err[order]

    err_gain_mat = np.zeros((len(methods), len(weights)))
    unst_gain_mat = np.zeros((len(methods), len(weights)))
    annot_err = [["" for _ in weights] for _ in methods]
    annot_unst = [["" for _ in weights] for _ in methods]

    for i, (prefix, _) in enumerate(methods):
        _, tick_m, _, unst_m, _, err_m, _, _, _ = collect_method_stats(episode_records, prefix, weights)
        for j, (tx, uy, ey) in enumerate(zip(tick_m, unst_m, err_m)):
            ref_u = np.interp(tx, bl_t_sorted, bl_u_sorted)
            ref_e = np.interp(tx, bl_t_sorted, bl_e_sorted)
            g_u = (1.0 - uy / ref_u) * 100.0
            g_e = (1.0 - ey / ref_e) * 100.0
            unst_gain_mat[i, j] = g_u
            err_gain_mat[i, j] = g_e
            sign_u = "−" if g_u >= 0 else "+"
            sign_e = "−" if g_e >= 0 else "+"
            annot_unst[i][j] = f"{sign_u}{abs(g_u):.1f}%\n({uy:.1f}s)"
            annot_err[i][j] = f"{sign_e}{abs(g_e):.1f}%\n({ey:.1f})"

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))

    for ax, mat, annot, title in [
        (axes[0], unst_gain_mat, annot_unst, "Unstable Time Reduction (%) vs Equal-Tick Fixed Baseline"),
        (axes[1], err_gain_mat, annot_err, "Integrated Error (AUEC) Reduction (%) vs Equal-Tick Baseline"),
    ]:
        im = ax.imshow(mat, cmap="YlGn", aspect="auto", vmin=0, vmax=max(55, np.max(mat) * 1.05))
        ax.grid(False)
        ax.set_xticks(np.arange(len(weights)))
        ax.set_xticklabels([f"β = {w}" for w in weights], fontweight="bold")
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels([m[1] for m in methods], fontweight="bold")
        ax.set_title(title, pad=12)
        for r in range(len(methods)):
            for c in range(len(weights)):
                val = mat[r, c]
                text_color = "white" if val > 35 else "#111111"
                ax.text(c, r, annot[r][c], ha="center", va="center", color=text_color, fontsize=9.5, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% Reduction (Higher = Better)")

    fig.suptitle(
        "Relative Improvement Over Equal-Budget Fixed-Period Scheduling Across All (Method, β) Configurations",
        y=1.03,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"04_heatmap_relative_gains.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_intra_episode_dynamics(full_records):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.8), sharex=True)

    series = [
        ("experimentDeepOfflineGNN", 0.95, "EdgeGAT GNN + Vec-MORL (β = 0.95)", "#d62728", "-"),
        ("experimentDeepOfflineContinuous", 0.95, "Continuous MLP RL (β = 0.95)", "#ff7f0e", ":"),
        ("experimentDeepOffline", 0.95, "Discrete BCQ (β = 0.95)", "#2ca02c", "-."),
        ("experimentTable", 0.95, "Tabular Q-Learning (β = 0.95)", "#1f77b4", "--"),
    ]

    t_grid = np.linspace(0.0, 200.0, 101)

    for prefix, w, label, color, ls in series:
        matching_all = [
            r for r in full_records
            if r["prefix"] == prefix and abs(r["coords"].get("weight", -1) - w) < 1e-6
        ]
        if not matching_all:
            continue
        wakeup_curves = []
        for rec in matching_all:
            t = get_col(rec, "time")
            wakeup = get_col(rec, "nextWakeUp[mean]")
            mask = (t >= 3800.0) & (t <= 4000.5)
            t_ep = t[mask] - 3800.0
            wakeup_curves.append(np.interp(t_grid, t_ep, wakeup[mask]))
        w_mat = np.vstack(wakeup_curves)
        w_mean = np.mean(w_mat, axis=0)
        w_sem = np.std(w_mat, axis=0) / np.sqrt(w_mat.shape[0])
        axes[0].plot(t_grid, w_mean, linestyle=ls, color=color, label=label, linewidth=2.2)
        axes[0].fill_between(t_grid, w_mean - w_sem, w_mean + w_sem, color=color, alpha=0.15)

        seed2 = [r for r in matching_all if abs(r["coords"].get("random", -1) - 2.0) < 1e-6]
        rec = seed2[0] if seed2 else matching_all[0]
        t = get_col(rec, "time")
        grad_sum = get_col(rec, "localComputation[sum]")
        mask = (t >= 3800.0) & (t <= 4000.5)
        t_ep = t[mask] - 3800.0
        axes[1].plot(t_ep, grad_sum[mask] / 100.0, linestyle=ls, color=color, label=label, linewidth=2.2)

        if prefix == "experimentDeepOfflineContinuous" and abs(w - 0.95) < 1e-6:
            gt_sum = get_col(rec, "groundTruth[sum]")
            axes[1].plot(
                t_ep,
                gt_sum[mask] / 100.0,
                linestyle=":",
                color="#111111",
                linewidth=2.0,
                label="Ground Truth g*(t) (Dijkstra)",
            )

    for ax in axes:
        ax.axvline(100.0, color="#333333", linestyle="--", linewidth=1.6, label="Source Switch (t = 100s)")
        ax.set_xlim(0, 200)

    axes[0].set_title("Adaptive Scheduling Period (nextWakeUp[mean] ± SEM Over 4 Seeds) Within Final Episode")
    axes[0].set_ylabel("Mean Next Wake-Up Δt (s)")
    axes[0].set_ylim(0.5, 5.3)
    axes[0].legend(loc="lower right", ncol=2, frameon=True)

    axes[1].set_title("Network Mean Gradient Output vs Exact Shortest-Path Ground Truth g*(t) (Seed = 2)")
    axes[1].set_ylabel("Mean Gradient Value (m)")
    axes[1].set_xlabel("Time Within Episode (s)")
    axes[1].legend(loc="lower right", ncol=2, frameon=True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"intra_episode_dynamics.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_summary_bars(episode_records):
    target_w = 0.95
    configs = [
        ("experimentBaseline", 5.0, "Baseline\nΔt = 5.0s", "#8c564b"),
        ("experimentBaseline", 3.0, "Baseline\nΔt = 3.0s", "#9467bd"),
        ("experimentBaseline", 2.0, "Baseline\nΔt = 2.0s", "#7f7f7f"),
        ("experimentTable", None, "Tabular\nQ-Learning", "#1f77b4"),
        ("experimentDeepOffline", None, "Discrete BCQ\n(Offline RL)", "#2ca02c"),
        ("experimentDeepOfflineContinuous", None, "Continuous\nMLP RL", "#ff7f0e"),
        ("experimentDeepOfflineGNN", None, "EdgeGAT GNN\n+ Vec-MORL", "#d62728"),
    ]

    metrics = [
        ("ticks[sum]", "Mean Ticks per Episode (↓ Less Energy)", "Ticks"),
        ("accumulatedUnstableTime[mean]", "Mean Unstable Time [s] (↓ Faster)", "Seconds"),
        ("accumulatedError[mean]", "Integrated Error AUEC (↓ More Accurate)", "AUEC"),
        ("accumulatedReward[mean]", "Mean Accumulated Reward (↑ Better)", "Reward"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.6))

    for ax, (col_name, title, ylabel) in zip(axes, metrics):
        means, sems, labels, colors = [], [], [], []
        for prefix, act_val, lbl, col in configs:
            if act_val is not None:
                matching = [
                    r for r in episode_records
                    if r["prefix"] == prefix
                    and abs(r["coords"].get("weight", -1) - target_w) < 1e-6
                    and abs(r["coords"].get("actions", -1) - act_val) < 1e-6
                ]
            else:
                matching = [
                    r for r in episode_records
                    if r["prefix"] == prefix
                    and abs(r["coords"].get("weight", -1) - target_w) < 1e-6
                ]
            vals = []
            for rec in matching:
                t = get_col(rec, "time")
                mask = t >= 2200.0
                vals.append(np.mean(get_col(rec, col_name)[mask]))
            n = max(1, len(vals))
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(n))
            labels.append(lbl)
            colors.append(col)

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=sems, color=colors, alpha=0.88, capsize=4, edgecolor="#222222", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5, rotation=20, ha="right")
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(ylabel)
        ax.margins(y=0.18)
        for bar, m, s in zip(bars, means, sems):
            va = "bottom" if m >= 0 else "top"
            y_pos = (m + s) if m >= 0 else (m - s)
            offset = 3 if m >= 0 else -4
            ax.annotate(
                f"{m:.1f}" if abs(m) < 100 else f"{int(round(m))}",
                (bar.get_x() + bar.get_width() / 2.0, y_pos),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
            )

    fig.suptitle(
        "Exploitation Performance Comparison at β = 0.95 (Mean ± SEM Over 4 Random Seeds, Episodes 11–20)",
        y=1.03,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"summary_comparison_beta08.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_iso_period_comparison(episode_records, full_records):
    """Compares Convergence Time and Error wrt Ground-Truth Optimum at EQUAL Average Wake-Up Period Δt_avg (s)."""
    weights = [0.5, 0.8, 0.95, 0.99, 1.0]
    a_vals, bl_ticks, bl_ticks_s, bl_unst, bl_unst_s, bl_err, bl_err_s = collect_baseline_stats(episode_records)
    # Effective average wake-up period Δt_avg = (200s * 100 nodes) / ticks
    bl_dt = 20000.0 / bl_ticks

    methods = [
        ("experimentDeepOfflineGNN", "EdgeGAT GNN + Vec-MORL", "#d62728", "-*"),
        ("experimentDeepOfflineContinuous", "Continuous MLP RL (TD3+BC)", "#ff7f0e", "-^"),
        ("experimentDeepOffline", "Discrete BCQ (Offline RL)", "#2ca02c", "-s"),
        ("experimentTable", "Tabular Q-Learning", "#1f77b4", "-o"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    method_curves = {}
    for prefix, label, color, style in methods:
        w_vals, tick_m, tick_s, unst_m, unst_s, err_m, err_s, _, _ = collect_method_stats(
            episode_records, prefix, weights
        )
        dt_m = 20000.0 / tick_m
        dt_s = (20000.0 / (tick_m ** 2)) * tick_s
        order = np.argsort(dt_m)
        dt_sorted = dt_m[order]
        u_sorted, u_s_sorted = unst_m[order], unst_s[order]
        e_sorted, e_s_sorted = err_m[order], err_s[order]
        w_sorted = w_vals[order]
        method_curves[prefix] = (dt_sorted, u_sorted, u_s_sorted, e_sorted, e_s_sorted, label, color)

        axes[0, 0].errorbar(
            dt_sorted, u_sorted, xerr=dt_s[order], yerr=u_s_sorted,
            fmt=style, color=color, label=label, markersize=6.5, capsize=3, zorder=4
        )
        axes[0, 1].errorbar(
            dt_sorted, e_sorted, xerr=dt_s[order], yerr=e_s_sorted,
            fmt=style, color=color, label=label, markersize=6.5, capsize=3, zorder=4
        )

    if len(bl_dt) > 0:
        order_bl = np.argsort(bl_dt)
        bl_dt_s = bl_dt[order_bl]
        bl_u_s = bl_unst[order_bl]
        bl_e_s = bl_err[order_bl]
        axes[0, 0].errorbar(
            bl_dt_s, bl_u_s, yerr=bl_unst_s[order_bl],
            fmt="--D", color="#444444", label="Fixed-Period Baseline (Constant Δt)", markersize=6, capsize=3, zorder=3
        )
        axes[0, 1].errorbar(
            bl_dt_s, bl_e_s, yerr=bl_err_s[order_bl],
            fmt="--D", color="#444444", label="Fixed-Period Baseline (Constant Δt)", markersize=6, capsize=3, zorder=3
        )

        # Shade improvement zone between Baseline and Continuous Offline RL
        c_dt, c_u, _, c_e, _, _, _ = method_curves["experimentDeepOfflineContinuous"]
        grid_dt = np.linspace(max(bl_dt_s.min(), c_dt.min()), min(bl_dt_s.max(), c_dt.max()), 100)
        axes[0, 0].fill_between(
            grid_dt,
            np.interp(grid_dt, c_dt, c_u),
            np.interp(grid_dt, bl_dt_s, bl_u_s),
            color="#d62728", alpha=0.10, label="RL Gain Over Equal-Period Baseline"
        )
        axes[0, 1].fill_between(
            grid_dt,
            np.interp(grid_dt, c_dt, c_e),
            np.interp(grid_dt, bl_dt_s, bl_e_s),
            color="#d62728", alpha=0.10, label="RL Gain Over Equal-Period Baseline"
        )

    axes[0, 0].set_title("Convergence Time vs Effective Mean Wake-Up Period Δt_avg")
    axes[0, 0].set_xlabel("Effective Mean Wake-Up Period Δt_avg [s] (Same X = Same Energy)")
    axes[0, 0].set_ylabel("Mean Convergence / Unstable Time [s] (↓ Faster)")

    axes[0, 1].set_title("Error wrt Ground-Truth Optimum g*(t) vs Mean Wake-Up Period Δt_avg")
    axes[0, 1].set_xlabel("Effective Mean Wake-Up Period Δt_avg [s] (Same X = Same Energy)")
    axes[0, 1].set_ylabel("Integrated Error wrt Optimum AUEC (↓ Lower Error)")

    # Bottom row: Grouped Bar Charts at exact Equal Mean Wake-Up Periods Δt_avg in {1.5s, 2.0s, 2.5s, 3.0s, 3.5s}
    target_periods = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    bar_series = [
        ("Baseline (Fixed Δt)", bl_dt[np.argsort(bl_dt)], bl_unst[np.argsort(bl_dt)], bl_err[np.argsort(bl_dt)], "#666666"),
        ("Tabular Q-Learning", method_curves["experimentTable"][0], method_curves["experimentTable"][1], method_curves["experimentTable"][3], "#1f77b4"),
        ("Discrete BCQ (Offline RL)", method_curves["experimentDeepOffline"][0], method_curves["experimentDeepOffline"][1], method_curves["experimentDeepOffline"][3], "#2ca02c"),
        ("Continuous MLP RL", method_curves["experimentDeepOfflineContinuous"][0], method_curves["experimentDeepOfflineContinuous"][1], method_curves["experimentDeepOfflineContinuous"][3], "#ff7f0e"),
        ("EdgeGAT GNN + Vec-MORL", method_curves["experimentDeepOfflineGNN"][0], method_curves["experimentDeepOfflineGNN"][1], method_curves["experimentDeepOfflineGNN"][3], "#d62728"),
    ]

    x = np.arange(len(target_periods))
    width = 0.155
    offsets = [-2.0 * width, -1.0 * width, 0.0, 1.0 * width, 2.0 * width]

    for (s_label, dt_x, u_y, e_y, col), off in zip(bar_series, offsets):
        u_interp = np.interp(target_periods, dt_x, u_y)
        e_interp = np.interp(target_periods, dt_x, e_y)
        b1 = axes[1, 0].bar(x + off, u_interp, width, label=s_label, color=col, alpha=0.88, edgecolor="#222222", linewidth=0.7)
        b2 = axes[1, 1].bar(x + off, e_interp, width, label=s_label, color=col, alpha=0.88, edgecolor="#222222", linewidth=0.7)
        for bar, val in zip(b1, u_interp):
            axes[1, 0].annotate(
                f"{val:.1f}s",
                (bar.get_x() + bar.get_width() / 2.0, val),
                textcoords="offset points", xytext=(0, 2), ha="center", va="bottom", fontsize=7.5, fontweight="bold"
            )
        for bar, val in zip(b2, e_interp):
            axes[1, 1].annotate(
                f"{val:.1f}",
                (bar.get_x() + bar.get_width() / 2.0, val),
                textcoords="offset points", xytext=(0, 2), ha="center", va="bottom", fontsize=7.5, fontweight="bold"
            )

    for ax_b, title_b, ylabel_b in [
        (axes[1, 0], "Direct Comparison at Equal Mean Wake-Up Period: Convergence Time", "Convergence / Unstable Time [s] (↓)"),
        (axes[1, 1], "Direct Comparison at Equal Mean Wake-Up Period: Error wrt Optimum", "Integrated Error wrt Optimum AUEC (↓)"),
    ]:
        ax_b.set_xticks(x)
        ax_b.set_xticklabels([f"Δt_avg = {p:.1f}s\n({int(round(20000/p))} ticks)" for p in target_periods], fontweight="bold")
        ax_b.set_title(title_b)
        ax_b.set_ylabel(ylabel_b)
        ax_b.margins(y=0.16)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Iso-Period (Equal Average Wake-Up Time Δt_avg) Comparison: Convergence Speed & Error wrt Optimum",
        y=0.99,
        fontsize=13.5,
        fontweight="bold",
    )
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.945))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"05_iso_wakeup_period_comparison.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_neural_embeddings_and_attribution():
    """Generates (1) 06_neural_embeddings_pca_tsne.png and (2) 07_feature_importance_and_response_surface.png."""
    emb_dir = "build/exports/embeddings"
    emb_path_95 = os.path.join(emb_dir, "embeddings_weight-0.95.csv")
    if not os.path.exists(emb_path_95):
        return

    def read_emb_csv(path):
        with open(path, "r") as f:
            header = f.readline().strip().split(",")
            rows = [line.strip().split(",") for line in f if line.strip()]
        regime_idx = header.index("regime")
        regimes = [r[regime_idx] for r in rows]
        num_cols = [i for i in range(len(header)) if i != regime_idx]
        num_names = [header[i] for i in num_cols]
        mat = np.array([[float(r[i]) for i in num_cols] for r in rows], dtype=float)
        return num_names, mat, regimes

    names, mat, regimes = read_emb_csv(emb_path_95)
    h_indices = [i for i, n in enumerate(names) if n.startswith("h_")]
    H = mat[:, h_indices]
    pred_dt = mat[:, names.index("predDt")]
    q_val = mat[:, names.index("qValue")]

    feat_keys = [
        ("deltaLatest", "Latest Trend Δg(t)"),
        ("deltaPrev", "Prev Trend Δg(t−1)"),
        ("selfDeltaMag", "Self Change |Δg_i|"),
        ("maxNeighDeltaMag", "Max Neighbor |Δg_j|"),
        ("neighChangedRatio", "Neighbor Change Ratio"),
        ("bfResidual", "Bellman-Ford Residual"),
        ("childrenRatio", "Backbone Children Ratio"),
        ("minNeighAction", "Min Neighbor Clock a_j"),
    ]
    X_feat = np.column_stack([mat[:, names.index(k)] for k, _ in feat_keys])

    # PCA on 64D hidden activations H
    H_centered = H - np.mean(H, axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
    Z_pca = U[:, :2] * S[:2]
    var_exp = (S ** 2) / np.sum(S ** 2) * 100.0

    # Kernel PCA (RBF) / Spectral Manifold Embedding on 64D hidden activations H
    norms = np.sum(H_centered ** 2, axis=1, keepdims=True)
    sq_dists = np.maximum(0.0, norms + norms.T - 2.0 * (H_centered @ H_centered.T))
    sigma2 = np.median(sq_dists[sq_dists > 0]) if np.any(sq_dists > 0) else 1.0
    K = np.exp(-sq_dists / (2.0 * sigma2))
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K_c = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    eigvals, eigvecs = np.linalg.eigh(K_c)
    Z_kpca = eigvecs[:, -1:-3:-1] * np.sqrt(np.maximum(eigvals[-1:-3:-1], 1e-9))

    # --- CHART 06: 2D Neural Network Latent Space Embeddings ---
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    regime_palette = {
        "Steady_State": ("Steady State (Converged)", "#1f77b4", "o"),
        "Neighbor_EarlyWarning": ("1-Hop Neighbor Early-Warning", "#ff7f0e", "s"),
        "Leaf_Wave": ("Active Wave (Leaf Node)", "#2ca02c", "^"),
        "Backbone_Wave": ("Active Wave (Backbone Node)", "#d62728", "P"),
    }

    for reg_key, (reg_label, col, marker) in regime_palette.items():
        idx = [i for i, r in enumerate(regimes) if r == reg_key]
        if not idx:
            continue
        axes[0].scatter(
            Z_pca[idx, 0], Z_pca[idx, 1],
            c=col, marker=marker, s=34, alpha=0.8, label=reg_label, edgecolors="white", linewidths=0.3
        )

    # Overlay Feature Biplot Loading Arrows on PCA embedding
    z_std = np.std(Z_pca, axis=0)
    arrow_scale = 2.2
    top_biplot_feats = [
        ("selfDeltaMag", "|Δg_i| (Self)"),
        ("maxNeighDeltaMag", "max |Δg_j| (Neigh)"),
        ("bfResidual", "BF Residual"),
        ("childrenRatio", "Children Ratio"),
        ("minNeighAction", "Min Neigh Clock"),
    ]
    for f_key, f_short in top_biplot_feats:
        f_vec = mat[:, names.index(f_key)]
        if np.std(f_vec) < 1e-6:
            continue
        corr_x = np.corrcoef(f_vec, Z_pca[:, 0])[0, 1] * z_std[0] * arrow_scale
        corr_y = np.corrcoef(f_vec, Z_pca[:, 1])[0, 1] * z_std[1] * arrow_scale
        axes[0].annotate(
            "", xy=(corr_x, corr_y), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#111111", lw=1.6, alpha=0.85)
        )
        axes[0].text(
            corr_x * 1.08, corr_y * 1.08, f_short,
            fontsize=8, fontweight="bold", color="#111111",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75)
        )

    axes[0].set_title(f"64D Hidden Layer PCA Biplot by Physical Regime\n(PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}% Variance)")
    axes[0].set_xlabel("Latent Principal Component 1 (PC1)")
    axes[0].set_ylabel("Latent Principal Component 2 (PC2)")
    axes[0].legend(loc="lower left", fontsize=8.5, frameon=True)

    sc1 = axes[1].scatter(
        Z_pca[:, 0], Z_pca[:, 1],
        c=pred_dt, cmap="plasma", s=34, alpha=0.88, edgecolors="white", linewidths=0.3
    )
    axes[1].set_title("PCA Latent Manifold Colored by Policy Output\nPredicted Wake-Up Period π(s) = Δt [s]")
    axes[1].set_xlabel("Latent Principal Component 1 (PC1)")
    axes[1].set_ylabel("Latent Principal Component 2 (PC2)")
    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04, label="Predicted Δt (s)")

    sc2 = axes[2].scatter(
        Z_kpca[:, 0], Z_kpca[:, 1],
        c=q_val, cmap="viridis", s=34, alpha=0.88, edgecolors="white", linewidths=0.3
    )
    axes[2].set_title("Nonlinear RBF-Kernel Embedding Colored by Critic\nLearned State-Action Value Q(s, π(s))")
    axes[2].set_xlabel("Nonlinear Manifold Dimension 1")
    axes[2].set_ylabel("Nonlinear Manifold Dimension 2")
    fig.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04, label="Critic Value Q(s, π(s))")

    fig.suptitle(
        "Learned Neural Network Hidden Representations (64D Penultimate Layer Embeddings, β = 0.95)",
        y=1.03,
        fontsize=13.5,
        fontweight="bold",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"06_neural_embeddings_pca_tsne.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # --- CHART 07: Input Feature Attribution & 2D Policy Decision Surfaces ---
    fig = plt.figure(figsize=(15.5, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.25)
    ax_imp = fig.add_subplot(gs[0, 0])
    ax_pdp = fig.add_subplot(gs[0, 1])
    ax_surf_leaf = fig.add_subplot(gs[1, 0])
    ax_surf_back = fig.add_subplot(gs[1, 1])

    # Panel A: Multi-Beta Input Feature Attribution (Normalized Jacobian Sensitivity |∂π(s)/∂x_k|)
    beta_files = [
        (0.8, "β = 0.80 (Balanced)", "#2ca02c"),
        (0.95, "β = 0.95 (High Reactivity)", "#d62728"),
        (0.99, "β = 0.99 (Max Speed)", "#9467bd"),
    ]
    y_pos = np.arange(len(feat_keys))
    bar_h = 0.25
    offsets = [-bar_h, 0.0, bar_h]

    for (b_val, b_lbl, b_col), off in zip(beta_files, offsets):
        p = os.path.join(emb_dir, f"embeddings_weight-{b_val}.csv")
        if not os.path.exists(p):
            continue
        b_names, b_mat, _ = read_emb_csv(p)
        imp_vals = []
        for f_key, _ in feat_keys:
            s_col = b_mat[:, b_names.index(f"sens_{f_key}")]
            f_std = max(0.15, np.std(b_mat[:, b_names.index(f_key)]))
            imp_vals.append(np.mean(s_col) * f_std)
        imp_vals = np.array(imp_vals)
        imp_norm = imp_vals / (np.sum(imp_vals) + 1e-9) * 100.0
        ax_imp.barh(y_pos + off, imp_norm, height=bar_h, color=b_col, alpha=0.88, label=b_lbl, edgecolor="#222222", linewidth=0.6)

    ax_imp.set_yticks(y_pos)
    ax_imp.set_yticklabels([lbl for _, lbl in feat_keys], fontweight="bold", fontsize=9.5)
    ax_imp.set_xlabel("Relative Impact on Policy Output Δt (%)")
    ax_imp.set_title("Input Feature Attribution (Gradient Sensitivity |∂π(s)/∂x_i|)")
    ax_imp.legend(loc="lower right", frameon=True)

    # Panel B: 1D Partial Dependence Curves (How each key feature drives Δt)
    surf_path_95 = os.path.join(emb_dir, "surface_weight-0.95.csv")
    if os.path.exists(surf_path_95):
        with open(surf_path_95, "r") as f:
            s_hdr = f.readline().strip().split(",")
            s_mat = np.array([[float(x) for x in line.strip().split(",")] for line in f if line.strip()])
        grid_u = np.unique(s_mat[:, 0])
        leaf_mask = np.abs(s_mat[:, 2] - 0.0) < 1e-4
        back_mask = np.abs(s_mat[:, 2] - 0.5) < 1e-4

        # Slice 1: Self Change Magnitude (with Neighbor Change = 0)
        s_leaf = s_mat[leaf_mask]
        s_back = s_mat[back_mask]
        n0_leaf = s_leaf[np.abs(s_leaf[:, 1] - 0.0) < 1e-4]
        n0_back = s_back[np.abs(s_back[:, 1] - 0.0) < 1e-4]
        s0_leaf = s_leaf[np.abs(s_leaf[:, 0] - 0.0) < 1e-4]
        s0_back = s_back[np.abs(s_back[:, 0] - 0.0) < 1e-4]

        ax_pdp.plot(n0_back[:, 0], n0_back[:, 4], "-o", color="#d62728", markersize=4, label="Self Change |Δg_i| (Backbone Node)")
        ax_pdp.plot(n0_leaf[:, 0], n0_leaf[:, 4], "--s", color="#ff7f0e", markersize=4, label="Self Change |Δg_i| (Leaf Node)")
        ax_pdp.plot(s0_back[:, 1], s0_back[:, 4], "-^", color="#2ca02c", markersize=4, label="Neighbor Change max|Δg_j| (Backbone)")
        ax_pdp.plot(s0_leaf[:, 1], s0_leaf[:, 4], ":d", color="#1f77b4", markersize=4, label="Neighbor Change max|Δg_j| (Leaf)")
        ax_pdp.set_title("Partial Dependence: Policy Response Δt vs Input Magnitude (β = 0.95)")
        ax_pdp.set_xlabel("Normalized Feature Intensity tanh(|Δg| / 50) ∈ [0, 1]")
        ax_pdp.set_ylabel("Chosen Wake-Up Period Δt (s)")
        ax_pdp.legend(loc="upper right", fontsize=8.5, frameon=True)

        # Panels C & D: 2D Policy Decision Landscapes (Leaf vs Backbone Node)
        G = len(grid_u)
        Z_leaf = s_leaf[:, 4].reshape(G, G) # rows = selfDeltaMag, cols = maxNeighDeltaMag
        Z_back = s_back[:, 4].reshape(G, G)
        vmin = min(Z_leaf.min(), Z_back.min())
        vmax = max(Z_leaf.max(), Z_back.max())

        for ax_s, Z_grid, role_title in [
            (ax_surf_leaf, Z_leaf, "Decision Surface Δt(s) — Leaf Node (childrenRatio = 0.0)"),
            (ax_surf_back, Z_back, "Decision Surface Δt(s) — Backbone Node (childrenRatio = 0.5)"),
        ]:
            im_s = ax_s.contourf(grid_u, grid_u, Z_grid.T, levels=18, cmap="plasma", vmin=vmin, vmax=vmax)
            cs = ax_s.contour(grid_u, grid_u, Z_grid.T, levels=6, colors="white", linewidths=0.8, alpha=0.7)
            ax_s.clabel(cs, inline=True, fontsize=8, fmt="%.2fs")
            ax_s.set_title(role_title, fontsize=11)
            ax_s.set_xlabel("Self Change Intensity tanh(|Δg_i| / 50)")
            ax_s.set_ylabel("Neighbor Change Intensity tanh(max|Δg_j| / 50)")
            fig.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04, label="Chosen Wake-Up Δt (s)")

    fig.suptitle(
        "Explainable AI (XAI) Study: Input Feature Attribution, Partial Dependence & 2D Decision Surfaces",
        y=0.99,
        fontsize=13.5,
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"07_feature_importance_and_response_surface.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_spatial_heatmaps_and_gif():
    """Generates:
       - charts/08_spatial_wave_filmstrip.{png,pdf}
       - charts/09_spatial_computation_heatmap.{png,pdf}
       - charts/spatial_wave_animation.gif
    """
    import matplotlib.tri as mtri
    from PIL import Image
    import io

    snap_path = "build/exports/spatial/spatial_snapshots_weight-0.8.csv"
    edge_path = "build/exports/spatial/spatial_edges_weight-0.8.csv"
    if not os.path.exists(snap_path):
        snap_path = "build/exports/spatial/spatial_snapshots_weight-0.95.csv"
        edge_path = "build/exports/spatial/spatial_edges_weight-0.95.csv"
    if not os.path.exists(snap_path):
        return

    with open(snap_path, "r") as f:
        hdr = f.readline().strip().split(",")
        raw = np.array([[float(x) for x in line.strip().split(",")] for line in f if line.strip()])

    edges = []
    if os.path.exists(edge_path):
        with open(edge_path, "r") as f:
            f.readline()
            for line in f:
                if line.strip():
                    parts = [float(x) for x in line.strip().split(",")]
                    edges.append((parts[2], parts[3], parts[4], parts[5]))

    times = np.unique(raw[:, 0])
    # Deduplicate per (tEp, nodeId) in case of multiple writes
    def get_frame(target_t):
        idx_t = np.argmin(np.abs(times - target_t))
        t_val = times[idx_t]
        sub = raw[np.abs(raw[:, 0] - t_val) < 1e-3]
        # Keep unique nodeIds (first 100)
        _, uniq_idx = np.unique(sub[:, 1], return_index=True)
        sub = sub[uniq_idx]
        return t_val, sub

    _, f0 = get_frame(times[0])
    triang = mtri.Triangulation(f0[:, 2], f0[:, 3])

    # --- CHART 08: 5-Column Spatial Filmstrip Across the Source Switch ---
    film_times = [95.0, 103.0, 108.0, 116.0, 138.0]
    fig, axes = plt.subplots(3, 5, figsize=(18.5, 10.5), sharex=True, sharey=True)

    for col_idx, target_t in enumerate(film_times):
        t_val, sub = get_frame(target_t)
        x, y = sub[:, 2], sub[:, 3]
        is_src = sub[:, 4] > 0.5
        grad = np.minimum(sub[:, 5], 550.0)
        wakeup = sub[:, 7]
        freq = 1.0 / np.maximum(0.5, wakeup)
        err = np.minimum(sub[:, 8], 250.0)

        # Row 0: Gradient Field g_i(t)
        ax0 = axes[0, col_idx]
        im0 = ax0.tricontourf(triang, grad, levels=np.linspace(0, 520, 15), cmap="viridis", alpha=0.85)
        for xU, yU, xV, yV in edges:
            ax0.plot([xU, xV], [yU, yV], color="white", lw=0.35, alpha=0.35)
        ax0.scatter(x[~is_src], y[~is_src], c=grad[~is_src], cmap="viridis", vmin=0, vmax=520, s=22, edgecolors="black", linewidths=0.4, zorder=3)
        if np.any(is_src):
            ax0.scatter(x[is_src], y[is_src], c="#ff1111", marker="*", s=220, edgecolors="white", linewidths=1.0, zorder=5, label="Source")
        ax0.set_title(f"t = {int(round(t_val))}s\n" + ("(Pre-Switch Steady)" if t_val < 100 else "(Post-Switch Wave)" if t_val < 125 else "(Re-Converged)"), fontsize=10.5)
        if col_idx == 0:
            ax0.set_ylabel("Gradient Field g(x,y)\nY Position (m)", fontweight="bold")

        # Row 1: Wake-Up Frequency Heatmap 1 / Δt_i(t) [Hz]
        ax1 = axes[1, col_idx]
        im1 = ax1.tricontourf(triang, freq, levels=np.linspace(0.2, 1.0, 15), cmap="inferno", alpha=0.88, extend="both")
        for xU, yU, xV, yV in edges:
            ax1.plot([xU, xV], [yU, yV], color="white", lw=0.35, alpha=0.3)
        ax1.scatter(x, y, c=freq, cmap="inferno", vmin=0.2, vmax=1.0, s=26, edgecolors="white", linewidths=0.4, zorder=3)
        if np.any(is_src):
            ax1.scatter(x[is_src], y[is_src], c="#00ffff", marker="*", s=200, edgecolors="black", linewidths=0.8, zorder=5)
        if col_idx == 0:
            ax1.set_ylabel("Computation Rate 1/Δt [Hz]\nY Position (m)", fontweight="bold")

        # Row 2: Instantaneous Local Error |g_i(t) - g*_i(t)| [m]
        ax2 = axes[2, col_idx]
        im2 = ax2.tricontourf(triang, err, levels=np.linspace(0, 180, 15), cmap="magma", alpha=0.88, extend="max")
        for xU, yU, xV, yV in edges:
            ax2.plot([xU, xV], [yU, yV], color="white", lw=0.35, alpha=0.3)
        ax2.scatter(x, y, c=err, cmap="magma", vmin=0, vmax=180, s=24, edgecolors="white", linewidths=0.4, zorder=3)
        if np.any(is_src):
            ax2.scatter(x[is_src], y[is_src], c="#00ff88", marker="*", s=200, edgecolors="black", linewidths=0.8, zorder=5)
        ax2.set_xlabel("X Position (m)")
        if col_idx == 0:
            ax2.set_ylabel("Gradient Error |g − g*| [m]\nY Position (m)", fontweight="bold")

    fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.015, pad=0.015, label="Gradient g(x,y) [m]")
    fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.015, pad=0.015, label="Wake-Up Rate 1/Δt [Hz] (Hot = Active)")
    fig.colorbar(im2, ax=axes[2, :].tolist(), fraction=0.015, pad=0.015, label="Error |g − g*| [m]")

    fig.suptitle(
        "2D Spatial-Temporal Evolution of Aggregate Gradient, Computation Rate (1/Δt), and Error Across Source Switch (t = 100s)",
        y=0.98,
        fontsize=14,
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"08_spatial_wave_filmstrip.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # --- CHART 09: Spatial Cumulative Computation Heatmap (Total Node Executions Over Episode) ---
    _, f_start = get_frame(times[0])
    _, f_end = get_frame(times[-1])
    _, f_pre = get_frame(95.0)
    _, f_post = get_frame(150.0)

    x, y = f_end[:, 2], f_end[:, 3]
    total_ticks_node = f_end[:, 9] - f_start[:, 9]
    src_pre = f_pre[:, 4] > 0.5
    src_post = f_post[:, 4] > 0.5

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    im_c = axes[0].tricontourf(triang, total_ticks_node, levels=16, cmap="YlOrRd", alpha=0.88)
    for xU, yU, xV, yV in edges:
        axes[0].plot([xU, xV], [yU, yV], color="#333333", lw=0.5, alpha=0.35)
    sc_c = axes[0].scatter(
        x, y, c=total_ticks_node, cmap="YlOrRd", s=65, edgecolors="black", linewidths=0.7, zorder=4
    )
    if np.any(src_pre):
        axes[0].scatter(x[src_pre], y[src_pre], c="#1f77b4", marker="*", s=280, edgecolors="white", linewidths=1.2, zorder=6, label="Source 1 (t ∈ [0, 100s])")
    if np.any(src_post):
        axes[0].scatter(x[src_post], y[src_post], c="#2ca02c", marker="*", s=280, edgecolors="white", linewidths=1.2, zorder=6, label="Source 2 (t ∈ [100, 200s])")

    axes[0].set_title("2D Spatial Heatmap of Node Computations (Ticks per Episode)")
    axes[0].set_xlabel("X Coordinate (m)")
    axes[0].set_ylabel("Y Coordinate (m)")
    axes[0].legend(loc="upper right", frameon=True)
    fig.colorbar(sc_c, ax=axes[0], fraction=0.046, pad=0.04, label="Executions (Ticks) per Node")

    # Right panel: Node Executions vs Node Degree (Topological Connectivity) & Distance to Source
    degrees = np.zeros(len(x))
    node_ids = f_end[:, 1].astype(int)
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    with open(edge_path, "r") as f:
        f.readline()
        for line in f:
            if line.strip():
                parts = line.strip().split(",")
                u, v = int(float(parts[0])), int(float(parts[1]))
                if u in id_to_idx:
                    degrees[id_to_idx[u]] += 1
                if v in id_to_idx:
                    degrees[id_to_idx[v]] += 1

    dist_to_sources = np.minimum(f_pre[:, 6], f_post[:, 6])
    sc_d = axes[1].scatter(
        dist_to_sources, total_ticks_node, c=degrees, cmap="viridis", s=70, alpha=0.9, edgecolors="black", linewidths=0.6
    )
    z_fit = np.polyfit(dist_to_sources, total_ticks_node, 1)
    d_line = np.linspace(dist_to_sources.min(), dist_to_sources.max(), 50)
    axes[1].plot(d_line, np.polyval(z_fit, d_line), "--", color="#d62728", lw=2.0, label="Linear Trend (Proximity to Sources)")
    axes[1].set_title("Node Computation Allocation vs Distance to Sources & Degree")
    axes[1].set_xlabel("Minimum Distance to Active Sources (m)")
    axes[1].set_ylabel("Total Executions (Ticks) in Episode")
    axes[1].legend(loc="upper right", frameon=True)
    fig.colorbar(sc_d, ax=axes[1], fraction=0.046, pad=0.04, label="Node Neighborhood Degree |N_i|")

    fig.suptitle(
        "Spatial Distribution of Computation Effort Across the 2D Network Topology",
        y=1.02,
        fontsize=13.5,
        fontweight="bold",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"09_spatial_computation_heatmap.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # --- ANIMATED GIF: charts/spatial_wave_animation.gif ---
    anim_times = [t for t in times if 88.0 <= t <= 148.0][::2]
    frames_pil = []
    for t_val in anim_times:
        _, sub = get_frame(t_val)
        x, y = sub[:, 2], sub[:, 3]
        is_src = sub[:, 4] > 0.5
        grad = np.minimum(sub[:, 5], 520.0)
        wakeup = sub[:, 7]
        freq = 1.0 / np.maximum(0.5, wakeup)
        err = np.minimum(sub[:, 8], 180.0)

        fig_g, axes_g = plt.subplots(1, 3, figsize=(14.5, 4.6))
        axes_g[0].tricontourf(triang, grad, levels=np.linspace(0, 520, 14), cmap="viridis", alpha=0.85)
        axes_g[0].scatter(x, y, c=grad, cmap="viridis", vmin=0, vmax=520, s=28, edgecolors="black", linewidths=0.4)
        if np.any(is_src):
            axes_g[0].scatter(x[is_src], y[is_src], c="#ff1111", marker="*", s=220, edgecolors="white", linewidths=1.0)
        axes_g[0].set_title("Gradient Field g(x,y) [m]")

        axes_g[1].tricontourf(triang, freq, levels=np.linspace(0.2, 1.0, 14), cmap="inferno", alpha=0.88, extend="both")
        axes_g[1].scatter(x, y, c=freq, cmap="inferno", vmin=0.2, vmax=1.0, s=30, edgecolors="white", linewidths=0.4)
        if np.any(is_src):
            axes_g[1].scatter(x[is_src], y[is_src], c="#00ffff", marker="*", s=220, edgecolors="black", linewidths=0.8)
        axes_g[1].set_title(f"Wake-Up Rate 1/Δt [Hz] (Mean Δt = {np.mean(wakeup):.2f}s)")

        axes_g[2].tricontourf(triang, err, levels=np.linspace(0, 180, 14), cmap="magma", alpha=0.88, extend="max")
        axes_g[2].scatter(x, y, c=err, cmap="magma", vmin=0, vmax=180, s=28, edgecolors="white", linewidths=0.4)
        if np.any(is_src):
            axes_g[2].scatter(x[is_src], y[is_src], c="#00ff88", marker="*", s=220, edgecolors="black", linewidths=0.8)
        axes_g[2].set_title(f"Error |g − g*| [m] (Mean = {np.mean(err):.1f}m)")

        fig_g.suptitle(f"Aggregate Computing Spatial Wave Dynamics — Episode Time t = {t_val:.0f}s", fontsize=13, fontweight="bold")
        fig_g.tight_layout()
        buf = io.BytesIO()
        fig_g.savefig(buf, format="png", dpi=110)
        plt.close(fig_g)
        buf.seek(0)
        frames_pil.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    if frames_pil:
        gif_path = os.path.join(OUTPUT_DIR, "spatial_wave_animation.gif")
        frames_pil[0].save(
            gif_path, save_all=True, append_images=frames_pil[1:], duration=220, loop=0
        )


def plot_generalization_across_blocks():
    """Generates:
       - charts/10_zero_shot_generalization_blocks.{png,pdf} (Bar comparison with Adaptive Heuristic)
       - charts/11_generalization_ticks_vs_error_pareto.{png,pdf} (Ticks vs Error & Unstable Time against Δt=5,4,3,2,1s + Adaptive Heuristic)
    """
    gen_dir = "build/exports/generalization/episode"
    if not os.path.exists(gen_dir):
        return
    gen_records = load_all_exports(gen_dir)
    if not gen_records:
        return

    programs = [
        (0, "BlockG: Euclidean Gradient\n(classicGradient)"),
        (1, "BlockG: Hop-Count Distance\n(hopDistance — Zero-Shot)"),
        (2, "BlockG: Value Broadcast\n(G-cast — Zero-Shot)"),
        (3, "BlockC + BlockG: Subtree Collect\n(C-cast — Zero-Shot)"),
    ]
    bar_algos = [
        (8, "Fixed\nΔt = 5.0s", "#8c564b"),
        (6, "Fixed\nΔt = 3.0s", "#9467bd"),
        (3, "Adaptive\nHeuristic (AIMD)", "#e377c2"),
        (2, "Tabular\nQ-Learning", "#1f77b4"),
        (1, "Discrete BCQ\n(Offline RL)", "#2ca02c"),
        (0, "Continuous\nMLP RL", "#ff7f0e"),
        (9, "EdgeGAT GNN\n+ Vec-MORL", "#d62728"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 10.8), sharex="col")

    for col_idx, (prog_id, prog_title) in enumerate(programs):
        unst_vals, err_vals, tick_vals, rew_vals = [], [], [], []
        labels, colors = [], []

        ref_err_5s = 1.0
        for r in gen_records:
            if int(round(r["coords"].get("progId", -1))) == prog_id and int(round(r["coords"].get("algoId", -1))) == 8:
                t = get_col(r, "time")
                ref_err_5s = max(1e-6, np.mean(get_col(r, "accumulatedError[mean]")[t >= 2200.0]))

        for algo_id, a_lbl, a_col in bar_algos:
            matching = [
                r for r in gen_records
                if int(round(r["coords"].get("progId", -1))) == prog_id
                and int(round(r["coords"].get("algoId", -1))) == algo_id
            ]
            if not matching:
                continue
            rec = matching[0]
            t = get_col(rec, "time")
            # Compare all adaptive/RL methods at equal ~7,500 tick budget:
            # For RL (algoId in 0,1) that is the β_in = 0.95 slice (t in [2950, 3250]);
            # For AIMD Heuristic (algoId == 3) that is the β_in = 0.80 slice (t in [2550, 2850], ~7,300 ticks)
            if algo_id in (0, 1, 9):
                mask = (t >= 2950.0) & (t <= 3250.0)
            elif algo_id == 3:
                mask = (t >= 2550.0) & (t <= 2850.0)
            else:
                mask = t >= 2200.0
            unst_vals.append(np.mean(get_col(rec, "accumulatedUnstableTime[mean]")[mask]))
            err_vals.append(np.mean(get_col(rec, "accumulatedError[mean]")[mask]) / ref_err_5s * 100.0)
            tick_vals.append(np.mean(get_col(rec, "ticks[sum]")[mask]))
            rew_vals.append(np.mean(get_col(rec, "accumulatedReward[mean]")[mask]))
            labels.append(a_lbl)
            colors.append(a_col)

        x = np.arange(len(labels))

        ax0 = axes[0, col_idx]
        b0 = ax0.bar(x, unst_vals, color=colors, alpha=0.88, edgecolor="#222222", linewidth=0.8)
        ax0.set_title(prog_title, fontsize=11, fontweight="bold")
        ax0.margins(y=0.18)
        if col_idx == 0:
            ax0.set_ylabel("Mean Unstable Time [s]\n(↓ Faster Convergence)", fontweight="bold")
        for bar, val in zip(b0, unst_vals):
            ax0.annotate(
                f"{val:.1f}s",
                (bar.get_x() + bar.get_width() / 2.0, val),
                textcoords="offset points", xytext=(0, 3), ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

        ax1 = axes[1, col_idx]
        b1 = ax1.bar(x, err_vals, color=colors, alpha=0.88, edgecolor="#222222", linewidth=0.8)
        ax1.margins(y=0.18)
        if col_idx == 0:
            ax1.set_ylabel("Normalized Error AUEC [%]\n(Δt=5s = 100%, ↓ Better)", fontweight="bold")
        for bar, val in zip(b1, err_vals):
            ax1.annotate(
                f"{val:.1f}%",
                (bar.get_x() + bar.get_width() / 2.0, val),
                textcoords="offset points", xytext=(0, 3), ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

        ax2 = axes[2, col_idx]
        b2 = ax2.bar(x, tick_vals, color=colors, alpha=0.88, edgecolor="#222222", linewidth=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=8, rotation=25, ha="right", fontweight="bold")
        ax2.margins(y=0.22)
        if col_idx == 0:
            ax2.set_ylabel("Mean Ticks per Episode\n(↓ Lower Energy)", fontweight="bold")
        for bar, val, rw in zip(b2, tick_vals, rew_vals):
            ax2.annotate(
                f"{int(round(val))}\n(R={rw:.1f})",
                (bar.get_x() + bar.get_width() / 2.0, val),
                textcoords="offset points", xytext=(0, 2), ha="center", va="bottom", fontsize=7, fontweight="bold"
            )

    fig.suptitle(
        "Zero-Shot Generalization Across Self-Stabilizing Building Blocks (Trained ONLY on Euclidean Gradient, Tested Zero-Shot vs Fixed & Adaptive Baselines)",
        y=0.99,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"10_zero_shot_generalization_blocks.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # --- CHART 11: Full β-Conditioned Single-Network Pareto Curves Across All 4 Self-Stabilizing Building Blocks ---
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 9.0), sharex="col")

    fixed_ids = [
        (4, "1s"),
        (5, "2s"),
        (6, "3s"),
        (7, "4s"),
        (8, "5s"),
    ]
    beta_slices = [
        (0.50, 2150.0, 2450.0),
        (0.80, 2550.0, 2850.0),
        (0.95, 2950.0, 3250.0),
        (0.99, 3350.0, 3650.0),
        (1.00, 3750.0, 4050.0),
    ]
    curve_algos = [
        (3, "Adaptive Heuristic (AIMD, β-swept)", "#e377c2", "-.P", 6.0),
        (1, "Discrete BCQ (1 Net, β-input Zero-Shot)", "#2ca02c", "--s", 6.0),
        (0, "Continuous MLP RL (1 Net, β-input Zero-Shot)", "#ff7f0e", "-^", 6.5),
        (9, "EdgeGAT GNN + Vec-MORL (1 Net, Zero-Shot)", "#d62728", "-*", 8.5),
    ]

    for col_idx, (prog_id, prog_title) in enumerate(programs):
        ax_u = axes[0, col_idx]
        ax_e = axes[1, col_idx]

        f_ticks, f_unst, f_err, f_lbls = [], [], [], []
        for aid, dt_lbl in fixed_ids:
            matching = [
                r for r in gen_records
                if int(round(r["coords"].get("progId", -1))) == prog_id
                and int(round(r["coords"].get("algoId", -1))) == aid
            ]
            if not matching:
                continue
            rec = matching[0]
            t = get_col(rec, "time")
            mask = t >= 2200.0
            f_ticks.append(np.mean(get_col(rec, "ticks[sum]")[mask]))
            f_unst.append(np.mean(get_col(rec, "accumulatedUnstableTime[mean]")[mask]))
            f_err.append(np.mean(get_col(rec, "accumulatedError[mean]")[mask]))
            f_lbls.append(dt_lbl)

        f_ticks = np.array(f_ticks)
        f_unst = np.array(f_unst)
        f_err = np.array(f_err)
        order = np.argsort(f_ticks)

        ax_u.plot(f_ticks[order], f_unst[order], "--D", color="#555555", markersize=5.5, label="Fixed-Period Curve (Δt=1..5s)", zorder=2)
        ax_e.plot(f_ticks[order], f_err[order], "--D", color="#555555", markersize=5.5, label="Fixed-Period Curve (Δt=1..5s)", zorder=2)

        for tx, uy, ey, dl in zip(f_ticks, f_unst, f_err, f_lbls):
            ax_u.annotate(f"Δt={dl}", (tx, uy), textcoords="offset points", xytext=(5, 2), fontsize=8, fontweight="bold", color="#333333")
            ax_e.annotate(f"Δt={dl}", (tx, ey), textcoords="offset points", xytext=(5, 2), fontsize=8, fontweight="bold", color="#333333")

        # Plot Tabular Q-Learning point
        q_match = [
            r for r in gen_records
            if int(round(r["coords"].get("progId", -1))) == prog_id
            and int(round(r["coords"].get("algoId", -1))) == 2
        ]
        if q_match:
            rec_q = q_match[0]
            tq = get_col(rec_q, "time")
            mq = tq >= 2200.0
            ax_u.scatter([np.mean(get_col(rec_q, "ticks[sum]")[mq])], [np.mean(get_col(rec_q, "accumulatedUnstableTime[mean]")[mq])], c="#1f77b4", marker="o", s=90, edgecolors="black", zorder=5, label="Tabular Q-Learning (Zero-Shot)")
            ax_e.scatter([np.mean(get_col(rec_q, "ticks[sum]")[mq])], [np.mean(get_col(rec_q, "accumulatedError[mean]")[mq])], c="#1f77b4", marker="o", s=90, edgecolors="black", zorder=5, label="Tabular Q-Learning (Zero-Shot)")

        # Plot β-conditioned Pareto curves for Heuristic, Discrete BCQ, and Continuous Offline RL
        for aid, lbl, col, style, msz in curve_algos:
            matching = [
                r for r in gen_records
                if int(round(r["coords"].get("progId", -1))) == prog_id
                and int(round(r["coords"].get("algoId", -1))) == aid
            ]
            if not matching:
                continue
            rec = matching[0]
            t = get_col(rec, "time")
            c_tx, c_uy, c_ey, c_betas = [], [], [], []
            for b_val, t_lo, t_hi in beta_slices:
                m_slice = (t >= t_lo) & (t <= t_hi)
                if np.any(m_slice):
                    c_tx.append(np.mean(get_col(rec, "ticks[sum]")[m_slice]))
                    c_uy.append(np.mean(get_col(rec, "accumulatedUnstableTime[mean]")[m_slice]))
                    c_ey.append(np.mean(get_col(rec, "accumulatedError[mean]")[m_slice]))
                    c_betas.append(b_val)
            c_tx, c_uy, c_ey = np.array(c_tx), np.array(c_uy), np.array(c_ey)
            s_ord = np.argsort(c_tx)
            ax_u.plot(c_tx[s_ord], c_uy[s_ord], style, color=col, markersize=msz, label=lbl, zorder=4)
            ax_e.plot(c_tx[s_ord], c_ey[s_ord], style, color=col, markersize=msz, label=lbl, zorder=4)

            if aid == 9 and len(f_ticks) >= 2:
                # Shade Pareto region between Fixed-Period curve and EdgeGAT GNN curve
                g_min = max(f_ticks.min(), c_tx.min())
                g_max = min(f_ticks.max(), c_tx.max())
                if g_max > g_min:
                    g_x = np.linspace(g_min, g_max, 60)
                    ax_u.fill_between(
                        g_x,
                        np.interp(g_x, c_tx[s_ord], c_uy[s_ord]),
                        np.interp(g_x, f_ticks[order], f_unst[order]),
                        color="#d62728", alpha=0.10
                    )
                    ax_e.fill_between(
                        g_x,
                        np.interp(g_x, c_tx[s_ord], c_ey[s_ord]),
                        np.interp(g_x, f_ticks[order], f_err[order]),
                        color="#d62728", alpha=0.10
                    )

        ax_u.set_title(prog_title, fontsize=11, fontweight="bold")
        ax_e.set_xlabel("Mean Ticks per Episode (↓ Less Energy)")
        if col_idx == 0:
            ax_u.set_ylabel("Mean Unstable Time [s]\n(↓ Faster Convergence)", fontweight="bold")
            ax_e.set_ylabel("Integrated Error AUEC\n(↓ More Accurate vs Optimum)", fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Single β-Conditioned Neural Network Tested Zero-Shot Across All 4 Self-Stabilizing Building Blocks (β_in ∈ [0.5, 1.0])",
        y=0.99,
        fontsize=13.5,
        fontweight="bold",
    )
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"11_generalization_ticks_vs_error_pareto.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_gnn_spatiotemporal_embeddings():
    """Generates charts/12_gnn_spatiotemporal_embeddings_and_attention.{png,pdf}."""
    import matplotlib.tri as mtri

    gnn_path = "build/exports/spatial/gnn_spatiotemporal_embeddings.csv"
    if not os.path.exists(gnn_path):
        return

    with open(gnn_path, "r") as f:
        hdr = f.readline().strip().split(",")
        raw = np.array([[float(x) for x in line.strip().split(",")] for line in f if line.strip()])

    h_cols = [i for i, c in enumerate(hdr) if c.startswith("h_")]
    H = raw[:, h_cols]
    H_mean = np.mean(H, axis=0, keepdims=True)
    H_c = H - H_mean
    U, S, Vt = np.linalg.svd(H_c, full_matrices=False)
    PC = H_c @ Vt[:2].T
    # Orient PC1 so that active wave states (high error / low wakeup) have positive PC1
    if np.corrcoef(PC[:, 0], -raw[:, 7])[0, 1] < 0:
        PC[:, 0] = -PC[:, 0]

    times = np.unique(raw[:, 0])

    def get_gnn_frame(target_t):
        idx_t = np.argmin(np.abs(times - target_t))
        t_val = times[idx_t]
        mask = np.abs(raw[:, 0] - t_val) < 1e-3
        sub = raw[mask]
        pc_sub = PC[mask]
        _, uniq_idx = np.unique(sub[:, 1], return_index=True)
        return t_val, sub[uniq_idx], pc_sub[uniq_idx]

    _, f0, _ = get_gnn_frame(times[0])
    triang = mtri.Triangulation(f0[:, 2], f0[:, 3])

    fig = plt.figure(figsize=(19.0, 10.8))
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[1.08, 1.0], hspace=0.34)
    top_gs = outer_gs[0].subgridspec(1, 5, wspace=0.14)
    bot_gs = outer_gs[1].subgridspec(1, 3, wspace=0.34)

    film_times = [95.0, 103.0, 108.0, 115.0, 138.0]
    pc1_min, pc1_max = np.percentile(PC[:, 0], 2), np.percentile(PC[:, 0], 98)

    im_top = None
    for idx, target_t in enumerate(film_times):
        ax = fig.add_subplot(top_gs[0, idx])
        t_val, sub, pc_sub = get_gnn_frame(target_t)
        x, y = sub[:, 2], sub[:, 3]
        is_src = sub[:, 4] > 0.5
        best_nx, best_ny = sub[:, 13], sub[:, 14]
        pc1 = np.clip(pc_sub[:, 0], pc1_min, pc1_max)

        im_top = ax.tricontourf(triang, pc1, levels=np.linspace(pc1_min, pc1_max, 16), cmap="coolwarm", alpha=0.88, extend="both")
        # Draw GNN Top-1 Attention Tree Edges (i -> argmax_j alpha_ij)
        dx = (best_nx - x) * 0.72
        dy = (best_ny - y) * 0.72
        ax.quiver(
            x, y, dx, dy,
            angles="xy", scale_units="xy", scale=1.0,
            color="#111111", alpha=0.65, width=0.0055, headwidth=3.8, headlength=4.5, zorder=3
        )
        ax.scatter(x, y, c=pc1, cmap="coolwarm", vmin=pc1_min, vmax=pc1_max, s=28, edgecolors="black", linewidths=0.4, zorder=4)
        if np.any(is_src):
            ax.scatter(x[is_src], y[is_src], c="#00ff66", marker="*", s=230, edgecolors="black", linewidths=1.0, zorder=6, label="Active Source")

        phase = "Pre-Switch Steady" if t_val < 100 else "Wave Ignition" if t_val < 105 else "Wave Front Expansion" if t_val < 120 else "Post-Wave Converged"
        ax.set_title(f"t = {int(round(t_val))}s ({phase})\nGNN Latent PC1 & Attention Tree", fontsize=10)
        ax.set_xlabel("X Position (m)")
        if idx == 0:
            ax.set_ylabel("Y Position (m)", fontweight="bold")
        else:
            ax.set_yticklabels([])

    # Bottom Row: 3 Analytical Panels
    ax_hov = fig.add_subplot(bot_gs[0, 0])
    ax_zones = fig.add_subplot(bot_gs[0, 1])
    ax_orbit = fig.add_subplot(bot_gs[0, 2])

    # Panel B1: Space-Time Hovmöller Diagram (Distance from New Source vs Time, colored by GNN PC1)
    _, f_post, _ = get_gnn_frame(140.0)
    src2_mask = f_post[:, 4] > 0.5
    src2_x = f_post[src2_mask, 2][0] if np.any(src2_mask) else 360.0
    src2_y = f_post[src2_mask, 3][0] if np.any(src2_mask) else 55.0

    _, f_pre, _ = get_gnn_frame(95.0)
    src1_mask = f_pre[:, 4] > 0.5
    src1_x = f_pre[src1_mask, 2][0] if np.any(src1_mask) else 110.0
    src1_y = f_pre[src1_mask, 3][0] if np.any(src1_mask) else -8.0

    dist_bins = np.linspace(0, 480, 25)
    t_list = [t for t in times if 88.0 <= t <= 150.0]
    hov_mat = np.zeros((len(dist_bins) - 1, len(t_list)))

    zone_old, zone_mid, zone_new = [], [], []
    orbit_pc1, orbit_pc2, orbit_dt = [], [], []

    for c_i, t_val in enumerate(t_list):
        _, sub, pc_sub = get_gnn_frame(t_val)
        d_new = np.sqrt((sub[:, 2] - src2_x) ** 2 + (sub[:, 3] - src2_y) ** 2)
        d_old = np.sqrt((sub[:, 2] - src1_x) ** 2 + (sub[:, 3] - src1_y) ** 2)
        for b_i in range(len(dist_bins) - 1):
            b_mask = (d_new >= dist_bins[b_i]) & (d_new < dist_bins[b_i + 1])
            if np.any(b_mask):
                hov_mat[b_i, c_i] = np.mean(pc_sub[b_mask, 0])

        zone_old.append(np.mean(pc_sub[d_old < 140.0, 0]))
        zone_new.append(np.mean(pc_sub[d_new < 140.0, 0]))
        zone_mid.append(np.mean(pc_sub[(d_old >= 140.0) & (d_new >= 140.0), 0]))

        orbit_pc1.append(np.mean(pc_sub[:, 0]))
        orbit_pc2.append(np.mean(pc_sub[:, 1]))
        orbit_dt.append(np.mean(sub[:, 7]))

    im_h = ax_hov.imshow(
        hov_mat, aspect="auto", origin="lower",
        extent=[t_list[0], t_list[-1], dist_bins[0], dist_bins[-1]],
        cmap="coolwarm", vmin=pc1_min, vmax=pc1_max
    )
    ax_hov.axvline(100.0, color="black", ls="--", lw=1.5)
    ax_hov.plot([101.0, 121.0], [20.0, 440.0], ":", color="#ffff00", lw=2.2, label="Wave Front Velocity (~21 m/s)")
    ax_hov.set_title("Space-Time Hovmöller Diagram of GNN PC1(x, t)\n(Diagonal Band = Traveling Wave Pattern)")
    ax_hov.set_xlabel("Episode Time t (s)")
    ax_hov.set_ylabel("Distance from New Source (m)")
    ax_hov.legend(loc="upper right", fontsize=8.5, frameon=True)
    fig.colorbar(im_h, ax=ax_hov, fraction=0.046, pad=0.04, label="GNN Latent PC1")

    # Panel B2: Phase-Shifted GNN Embedding Pulses Across Spatial Zones
    ax_zones.plot(t_list, zone_new, "-", color="#d62728", lw=2.2, label="Zone 1: Near New Source (<140m)")
    ax_zones.plot(t_list, zone_old, "--", color="#1f77b4", lw=2.2, label="Zone 2: Near Old Source (<140m)")
    ax_zones.plot(t_list, zone_mid, "-.", color="#2ca02c", lw=2.2, label="Zone 3: Far / Perimeter Nodes (>140m)")
    ax_zones.axvline(100.0, color="#222222", ls=":", lw=1.5, label="Source Switch (t = 100s)")
    ax_zones.set_title("Temporal Evolution of GNN Embedding PC1(t)\nAcross 3 Spatial Network Zones")
    ax_zones.set_xlabel("Episode Time t (s)")
    ax_zones.set_ylabel("Mean GNN Latent Activation PC1")
    ax_zones.legend(loc="upper right", fontsize=8.5, frameon=True)

    # Panel B3: Collective cyclic trajectory in 2D GNN Latent Space (PC1(t), PC2(t))
    orbit_pc1 = np.array(orbit_pc1)
    orbit_pc2 = np.array(orbit_pc2)
    orbit_dt = np.array(orbit_dt)
    ax_orbit.plot(orbit_pc1, orbit_pc2, "-", color="#555555", lw=1.5, alpha=0.6, zorder=2)
    sc_orb = ax_orbit.scatter(
        orbit_pc1, orbit_pc2, c=orbit_dt, cmap="plasma", s=55, edgecolors="black", linewidths=0.5, zorder=4
    )
    for mark_t in [95.0, 102.0, 106.0, 112.0, 120.0, 140.0]:
        idx_m = np.argmin(np.abs(np.array(t_list) - mark_t))
        ax_orbit.annotate(
            f"t={int(round(t_list[idx_m]))}s",
            (orbit_pc1[idx_m], orbit_pc2[idx_m]),
            textcoords="offset points", xytext=(6, 4), fontsize=8.5, fontweight="bold"
        )
    ax_orbit.set_title("Collective Dynamical Cycle in 2D GNN Latent Space\n(Attractor → Transient Wave Orbit → Attractor)")
    ax_orbit.set_xlabel("Mean Network GNN Latent PC1(t)")
    ax_orbit.set_ylabel("Mean Network GNN Latent PC2(t)")
    fig.colorbar(sc_orb, ax=ax_orbit, fraction=0.046, pad=0.04, label="Mean Δt (s)")

    fig.suptitle(
        "Spatio-Temporal Analysis of EdgeGAT Neural Embeddings & Learned Graph Attention Tree Over Time",
        y=0.98,
        fontsize=14,
        fontweight="bold",
    )
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"12_gnn_spatiotemporal_embeddings_and_attention.{ext}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ep_records = load_all_exports(EPISODE_DIR)
    full_records = []
    for d in FULL_DIRS:
        if os.path.exists(d):
            full_records.extend(load_all_exports(d))

    print(f"Loaded {len(ep_records)} episode files and {len(full_records)} full trajectory files.")
    plot_learning_curves(ep_records)
    plot_ticks_and_unstable_curves(ep_records)
    plot_tradeoff_and_pareto(ep_records)
    plot_dual_pareto_frontiers(ep_records)
    plot_instantaneous_error_zoom(full_records)
    plot_multi_beta_intra_episode(full_records)
    plot_heatmap_gains(episode_records := ep_records)
    plot_intra_episode_dynamics(full_records)
    plot_summary_bars(ep_records)
    plot_iso_period_comparison(ep_records, full_records)
    plot_neural_embeddings_and_attribution()
    plot_spatial_heatmaps_and_gif()
    plot_generalization_across_blocks()
    plot_gnn_spatiotemporal_embeddings()
    print(f"Saved all charts to '{OUTPUT_DIR}/'.")
