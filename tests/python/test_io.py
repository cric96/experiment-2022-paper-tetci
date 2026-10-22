"""Tests for the Alchemist CSV parser."""
from pathlib import Path

import numpy as np
import pytest

from analysis.io import experiment_name, load_runs, parse_alchemist_csv

FIXTURES = Path(__file__).parent / "fixtures"
TABLE_RUN = FIXTURES / "experimentTable_random-1.0_weight-0.95.csv"


def test_parses_variables_from_header() -> None:
    run = parse_alchemist_csv(TABLE_RUN)
    assert run.params == {"random": 1.0, "weight": 0.95}


def test_ignores_numbers_in_start_and_end_banners() -> None:
    run = parse_alchemist_csv(TABLE_RUN)
    assert "started" not in " ".join(run.params)
    assert set(run.params) == {"random", "weight"}


def test_uses_last_comment_line_as_column_names() -> None:
    run = parse_alchemist_csv(TABLE_RUN)
    assert list(run.data.columns) == [
        "time",
        "ticks[sum]",
        "accumulatedReward[mean]",
        "accumulatedUnstableTime[mean]",
        "accumulatedError[mean]",
        "epsilon",
    ]


def test_reads_every_data_row_including_nan() -> None:
    run = parse_alchemist_csv(TABLE_RUN)
    assert len(run.data) == 5
    assert np.isnan(run.data["accumulatedReward[mean]"].iloc[0])
    assert run.data["ticks[sum]"].iloc[-1] == pytest.approx(7062.0)


def test_experiment_name_is_file_prefix() -> None:
    assert experiment_name(TABLE_RUN) == "experimentTable"
    assert experiment_name(Path("experimentDeepOfflineGNN_random-2.0_weight-0.5.csv")) == "experimentDeepOfflineGNN"


def test_run_exposes_experiment_and_path() -> None:
    run = parse_alchemist_csv(TABLE_RUN)
    assert run.experiment == "experimentTable"
    assert run.path == TABLE_RUN


def test_load_runs_reads_all_csv_in_directory() -> None:
    runs = load_runs(FIXTURES)
    assert [r.experiment for r in runs] == ["experimentTable"]


def test_load_runs_on_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert load_runs(tmp_path / "does-not-exist") == []
