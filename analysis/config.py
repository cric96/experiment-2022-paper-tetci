"""Experiment constants shared by the whole analysis.

They mirror the Alchemist YAML files in ``src/main/yaml`` and the
``CentralAgent`` reaction; change them only together with the simulations.
"""

from __future__ import annotations

from pathlib import Path

# --- simulation protocol ---------------------------------------------------
EPISODE_LENGTH_S = 200.0  # every episode lasts 200 s of simulated time
N_EPISODES = 20  # 20 episodes per run (4000 s)
TRAINING_EPISODES = 10  # episodes 1..10: exploration + learning; then frozen
EVAL_EPISODES = tuple(range(TRAINING_EPISODES + 1, N_EPISODES + 1))  # 11..20
FINAL_EPISODE_START_S = (N_EPISODES - 1) * EPISODE_LENGTH_S  # 3800 s
SOURCE_SWITCH_S = 100.0  # the gradient source moves halfway through an episode
N_NODES = 100  # 10 x 10 perturbed grid
FULL_EXPORT_INTERVAL_S = 2.0  # sampling period of the "full" exports
EPISODE_RESET_DELAY_S = 1.0  # nodes of a new episode are created 1 s after the boundary

# --- experimental factors ---------------------------------------------------
WEIGHTS = (0.5, 0.8, 0.95, 0.99, 1.0)  # beta: convergence vs. energy trade-off
FOCUS_WEIGHT = 0.95  # beta used by single-beta figures
FIXED_PERIODS_S = (1.0, 2.0, 3.0, 4.0, 5.0)  # fixed-period baselines (= action space)
SPATIAL_SEED = 2  # seed of the spatial / embedding exports

# In the generalization experiment the beta-conditioned policies are queried,
# after training, with beta_in = WEIGHTS[(episode - 11) // 2]: two episodes per beta.
EPISODES_PER_GENERALIZATION_BETA = 2

# Self-stabilising building blocks of the generalization experiment (progId).
PROGRAMS = {0: "gradient", 1: "hop", 2: "broadcast", 3: "collect", 4: "leaderElection"}
PROGRAM_LABELS = {
    "gradient": "Gradient (G, trained)",
    "hop": "Hop-count gradient (G)",
    "broadcast": "Broadcast (G-cast)",
    "collect": "Collect (C)",
    "leaderElection": r"Election + broadcast ($S\to G$)",
}
PROGRAM_SHORT_LABELS = {"gradient": "Gradient", "hop": "Hop count", "broadcast": "Broadcast", "collect": "Collect", "leaderElection": "Election + G"}

# Grain of sparse leader election, matching LeaderElection.Grain in the simulator.
LEADER_ELECTION_GRAIN_M = 400.0

# --- statistics -------------------------------------------------------------
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 10_000
RANDOM_SEED = 20221  # for bootstrap resampling, so outputs are reproducible

# --- default locations (relative to the repository root) ---------------------
DEFAULT_EXPORTS_DIR = Path("build/exports")
DEFAULT_OUTPUT_DIR = Path("charts")


def ticks_to_mean_period(ticks: float) -> float:
    """Effective mean wake-up period (s) of a node given the rounds per episode."""
    return N_NODES * EPISODE_LENGTH_S / ticks
