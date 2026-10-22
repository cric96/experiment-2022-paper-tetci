"""Tests for the LaTeX table helpers."""
import numpy as np

from analysis.tables import booktabs, format_interval, format_mean_ci, format_p


def test_format_mean_ci_uses_half_width() -> None:
    assert format_mean_ci(12.345, 11.0, 13.69, digits=1) == "$12.3 \\pm 1.3$"


def test_format_mean_ci_without_interval_shows_mean_only() -> None:
    assert format_mean_ci(3.0, np.nan, np.nan, digits=2) == "$3.00$"


def test_format_mean_ci_of_deterministic_value_shows_mean_only() -> None:
    assert format_mean_ci(19900.0, 19900.0, 19900.0, digits=0) == "$19900$"


def test_format_mean_ci_uses_unicode_free_minus() -> None:
    assert format_mean_ci(-2.5, -2.6, -2.4, digits=2) == "$-2.50 \\pm 0.10$"


def test_format_interval_as_percent_with_brackets() -> None:
    assert format_interval(0.456, 0.43, 0.48, percent=True) == "$46$ [$43$, $48$]"


def test_format_interval_of_undefined_value_is_dash() -> None:
    assert format_interval(np.nan, np.nan, np.nan, percent=True) == "--"


def test_format_p_switches_to_threshold_for_tiny_values() -> None:
    assert format_p(0.0312) == "$0.031$"
    assert format_p(0.00004) == "$<0.001$"
    assert format_p(1.0) == "$1.000$"


def test_booktabs_wraps_rows_in_rules() -> None:
    latex = booktabs(["A", "B"], [["1", "2"], None, ["3", "4"]], align="lr")
    lines = latex.splitlines()
    assert lines[0] == "\\begin{tabular}{lr}"
    assert "\\toprule" in lines and "\\bottomrule" in lines
    assert lines[lines.index("\\toprule") + 1] == "A & B \\\\"
    assert "\\midrule" in lines[lines.index("1 & 2 \\\\"):]
    assert lines[-1] == "\\end{tabular}"
