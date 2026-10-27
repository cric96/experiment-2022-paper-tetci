"""Tests for the tidy tables built from raw Alchemist runs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import config
from analysis.data import (
    episode_table,
    evaluation_summary,
    final_episode_trace,
    generalization_beta,
    generalization_table,
    resample,
)
from analysis.io import AlchemistRun

EPISODE_COLUMNS = [
    "time",
    "ticks[sum]",
    "accumulatedReward[mean]",
    "accumulatedUnstableTime[mean]",
    "accumulatedError[mean]",
]


def _episode_run(experiment: str, params: dict[str, float], values: list[float]) -> AlchemistRun:
    """A run whose episode k has every metric equal to ``values[k - 1]``."""
    rows = [[0.0, 0.0, np.nan, np.nan, np.nan], [0.0049, 1.0, 0.0, 0.0, 0.0]]
    for k, v in enumerate(values, start=1):
        rows.append([k * config.EPISODE_LENGTH_S + 0.004, v, v, v, v])
    data = pd.DataFrame(rows, columns=EPISODE_COLUMNS)
    return AlchemistRun(path=Path(f"{experiment}_x.csv"), experiment=experiment, params=params, data=data)


def test_episode_table_maps_experiments_to_methods_and_parameters() -> None:
    runs = [
        _episode_run("experimentTable", {"random": 1.0, "weight": 0.95}, [1.0, 2.0]),
        _episode_run("experimentBaseline", {"random": 3.0, "weight": 0.8, "actions": 2.0}, [5.0, 6.0]),
    ]
    table = episode_table(runs)
    q = table[table.method == "qlearning"]
    assert q.weight.unique().tolist() == [0.95]
    assert q.seed.unique().tolist() == [1]
    assert q.period.isna().all()
    fixed = table[table.method == "fixed"]
    assert fixed.period.unique().tolist() == [2.0]
    assert fixed.seed.unique().tolist() == [3]


def test_episode_table_indexes_episodes_from_time_and_drops_start_rows() -> None:
    table = episode_table([_episode_run("experimentTable", {"random": 1.0, "weight": 0.5}, [10.0, 20.0, 30.0])])
    assert table.episode.tolist() == [1, 2, 3]
    assert table.ticks.tolist() == [10.0, 20.0, 30.0]
    assert set(["reward", "unstable", "error"]).issubset(table.columns)


def test_evaluation_summary_averages_only_evaluation_episodes() -> None:
    values = [100.0] * config.TRAINING_EPISODES + [1.0, 3.0] * ((config.N_EPISODES - config.TRAINING_EPISODES) // 2)
    table = episode_table([_episode_run("experimentDeepOfflineGNN", {"random": 2.0, "weight": 1.0}, values)])
    summary = evaluation_summary(table)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row.method == "gnn"
    assert row.ticks == pytest.approx(2.0)
    assert row.error == pytest.approx(2.0)


def test_generalization_beta_follows_evaluation_schedule() -> None:
    assert np.isnan(generalization_beta(config.TRAINING_EPISODES))
    assert generalization_beta(11) == 0.5
    assert generalization_beta(12) == 0.5
    assert generalization_beta(15) == 0.95
    assert generalization_beta(20) == 1.0


def test_generalization_table_decodes_algorithm_and_program_ids() -> None:
    runs = [
        _episode_run("experimentGeneralization", {"random": 2.0, "progId": 1.0, "algoId": 6.0}, [1.0] * 12),
        _episode_run("experimentGeneralization", {"random": 2.0, "progId": 3.0, "algoId": 9.0}, [1.0] * 12),
    ]
    table = generalization_table(runs)
    fixed = table[table.method == "fixed"]
    assert fixed.period.unique().tolist() == [3.0]
    assert fixed.program.unique().tolist() == ["hop"]
    gnn = table[table.method == "gnn"]
    assert gnn.program.unique().tolist() == ["collect"]
    assert gnn[gnn.episode == 11].beta_in.tolist() == [0.5]


def _full_run() -> AlchemistRun:
    start = config.FINAL_EPISODE_START_S
    times = np.array([start - 2.0, start + 2.0, start + 100.0, start + 102.0])
    data = pd.DataFrame(
        {
            "time": times,
            "ticks[sum]": [9.0, 100.0, 5000.0, 5100.0],
            "nextWakeUp[mean]": [3.0, 1.0, 2.0, 1.5],
            "localComputation[sum]": [0.0, 100.0, 300.0, 200.0],
            "groundTruth[sum]": [0.0, 100.0, 100.0, 200.0],
        }
    )
    return AlchemistRun(path=Path("x.csv"), experiment="experimentTable", params={"random": 1.0}, data=data)


def test_final_episode_trace_uses_episode_relative_time() -> None:
    trace = final_episode_trace(_full_run())
    assert trace.t.tolist() == [2.0, 100.0, 102.0]
    assert trace.wakeup.tolist() == [1.0, 2.0, 1.5]


def test_final_episode_trace_computes_mean_gradient_deviation() -> None:
    trace = final_episode_trace(_full_run())
    # |sum(g) - sum(g*)| / N_NODES
    assert trace.deviation.tolist() == pytest.approx([0.0, 200.0 / config.N_NODES, 0.0])
    assert trace.gradient_mean.tolist() == pytest.approx([1.0, 3.0, 2.0])


def test_resample_interpolates_each_trace_on_a_common_grid() -> None:
    a = pd.DataFrame({"t": [0.0, 10.0], "v": [0.0, 10.0]})
    b = pd.DataFrame({"t": [0.0, 10.0], "v": [10.0, 30.0]})
    grid = np.array([0.0, 5.0, 10.0])
    matrix = resample([a, b], "v", grid)
    assert matrix.shape == (2, 3)
    assert matrix[0].tolist() == [0.0, 5.0, 10.0]
    assert matrix[1].tolist() == [10.0, 20.0, 30.0]


def test_generalization_decodes_leader_election_transfer():
    run = _episode_run("experimentGeneralization", {"random": 2.0, "progId": 4.0, "algoId": 9.0}, [1.0] * 20)
    table = generalization_table([run])
    assert table.program.unique().tolist() == ["leaderElection"]
    assert table[table.episode == 16].beta_in.tolist() == [0.95]
