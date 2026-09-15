"""Minimal unit tests for the shared engine (run with: pytest)."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from engine import stats, insights  # noqa: E402


def test_describe_basic():
    s = pd.Series([1, 2, 3, 4, 5])
    d = stats.describe(s)
    assert d["count"] == 5
    assert d["mean"] == 3.0


def test_describe_empty():
    assert stats.describe(pd.Series([], dtype=float)) == {"count": 0}


def test_pct_change():
    assert math.isclose(stats.pct_change(110, 100), 0.10)
    assert stats.pct_change(10, 0) is None


def test_detect_anomalies_zscore_flags_outlier():
    s = pd.Series([10, 11, 9, 10, 12, 100, 11, 10])
    anomalies = stats.detect_anomalies_zscore(s, threshold=2.0)
    assert len(anomalies) == 1
    assert anomalies[0].value == 100


def test_detect_anomalies_zscore_constant_series_no_crash():
    s = pd.Series([5, 5, 5, 5])
    assert stats.detect_anomalies_zscore(s) == []


def test_rank_with_percentile():
    s = pd.Series({"A": 100, "B": 50, "C": 200})
    ranked = stats.rank_with_percentile(s, ascending=False)
    assert ranked.loc["C", "rank"] == 1
    assert ranked.loc["B", "rank"] == 3


def test_period_over_period_shape():
    dates = pd.date_range("2026-01-01", periods=90, freq="D")
    df = pd.DataFrame({"date": dates, "sales": np.arange(90)})
    pop = stats.period_over_period(df, "date", "sales", freq="MS")
    assert list(pop.columns) == ["value", "prev_value", "pct_change"]
    assert len(pop) == 3


def test_drop_incomplete_last_month_removes_partial_trailing_bucket():
    dates = pd.date_range("2026-06-01", periods=87, freq="D")  # runs through Aug 26 (partial Aug)
    df = pd.DataFrame({"date": dates, "sales": np.ones(len(dates))})
    pop = stats.period_over_period(df, "date", "sales", freq="MS")
    trimmed = stats.drop_incomplete_last_month(pop, last_actual_date=dates.max())
    assert len(trimmed) == len(pop) - 1


def test_drop_incomplete_last_month_keeps_complete_trailing_bucket():
    dates = pd.date_range("2026-06-01", "2026-08-31", freq="D")  # full August
    df = pd.DataFrame({"date": dates, "sales": np.ones(len(dates))})
    pop = stats.period_over_period(df, "date", "sales", freq="MS")
    trimmed = stats.drop_incomplete_last_month(pop, last_actual_date=dates.max())
    assert len(trimmed) == len(pop)


def test_finding_sales_vs_margin_gap_flags_high_sales_low_margin_store():
    store_summary = pd.DataFrame(
        {
            "sales": [1000, 500, 400, 300, 200],
            "gross_margin": [0.10, 0.40, 0.38, 0.35, 0.42],
        },
        index=["B店", "A店", "C店", "D店", "E店"],
    )
    findings = insights.finding_sales_vs_margin_gap(store_summary)
    titles = [f.title for f in findings]
    assert any("B店" in t for t in titles)


def test_to_markdown_handles_empty():
    assert "検出" in insights.to_markdown([])


def test_moving_average_smooths_series():
    s = pd.Series([10, 100, 10, 100, 10, 100, 10])
    ma = stats.moving_average(s, window=3)
    # the smoothed series should have far less variance than the raw noisy one
    assert ma.dropna().std() < s.std()


def test_quadrant_classify_median_split():
    df = pd.DataFrame(
        {
            "sales": [1000, 900, 100, 50],
            "margin": [0.10, 0.45, 0.40, 0.05],
        },
        index=["B店", "A店", "C店", "D店"],
    )
    labels = stats.quadrant_classify(df, "sales", "margin")
    assert labels["B店"] == "要注意"     # high sales, low margin
    assert labels["A店"] == "優良"       # high sales, high margin
    assert labels["C店"] == "堅実"       # low sales, high margin
    assert labels["D店"] == "要改善"     # low sales, low margin


def test_quadrant_classify_custom_labels():
    df = pd.DataFrame({"x": [10, 1], "y": [10, 1]}, index=["hi", "lo"])
    labels = stats.quadrant_classify(df, "x", "y", labels={"good": "GOOD", "needs_work": "BAD"})
    assert labels["hi"] == "GOOD"
    assert labels["lo"] == "BAD"
