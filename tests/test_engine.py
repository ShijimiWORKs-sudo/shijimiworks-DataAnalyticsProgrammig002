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


def test_finding_high_cancellation_rate_flags_spike_month():
    dates = pd.date_range("2026-01-01", "2026-04-30", freq="D")
    df = pd.DataFrame({"date": dates, "rooms_sold": 50, "cancellations": 4})
    # March: a sustained cancellation spike
    march_mask = dates.month == 3
    df.loc[march_mask, "cancellations"] = 40
    findings = insights.finding_high_cancellation_rate(df, "date", "cancellations", "rooms_sold")
    assert len(findings) == 1
    assert "2026-03" in findings[0].title


def test_finding_high_cancellation_rate_no_flag_when_stable():
    dates = pd.date_range("2026-01-01", "2026-04-30", freq="D")
    df = pd.DataFrame({"date": dates, "rooms_sold": 50, "cancellations": 4})
    assert insights.finding_high_cancellation_rate(df, "date", "cancellations", "rooms_sold") == []


def test_finding_weekday_occupancy_gap_flags_large_gap():
    dates = pd.date_range("2026-01-01", periods=28, freq="D")
    occ = [0.9 if d.day_name() in ("Friday", "Saturday") else 0.4 for d in dates]
    df = pd.DataFrame({"date": dates, "occupancy_rate": occ})
    findings = insights.finding_weekday_occupancy_gap(df, "date", "occupancy_rate")
    assert len(findings) == 1


def test_finding_weekday_occupancy_gap_no_flag_when_even():
    dates = pd.date_range("2026-01-01", periods=28, freq="D")
    df = pd.DataFrame({"date": dates, "occupancy_rate": 0.6})
    assert insights.finding_weekday_occupancy_gap(df, "date", "occupancy_rate") == []


def test_finding_checkout_slump_flags_collapsed_month():
    dates = pd.date_range("2026-01-01", "2026-04-30", freq="D")
    df = pd.DataFrame({"date": dates, "checkout_hits": 3, "checkout_attempts": 8})
    march_mask = dates.month == 3
    df.loc[march_mask, ["checkout_hits", "checkout_attempts"]] = [1, 10]
    findings = insights.finding_checkout_slump(df, "date", "checkout_hits", "checkout_attempts")
    assert len(findings) == 1
    assert "2026-03" in findings[0].title


def test_finding_checkout_slump_no_flag_when_stable():
    dates = pd.date_range("2026-01-01", "2026-04-30", freq="D")
    df = pd.DataFrame({"date": dates, "checkout_hits": 3, "checkout_attempts": 8})
    assert insights.finding_checkout_slump(df, "date", "checkout_hits", "checkout_attempts") == []


def test_finding_toughest_rival_flags_frequent_losing_matchup():
    summary = pd.DataFrame(
        {"matches": [20, 3, 10], "win_rate": [0.20, 0.10, 0.60]},
        index=["nemesis", "rare_opponent", "easy_opponent"],
    )
    findings = insights.finding_toughest_rival(summary)
    assert len(findings) == 1
    assert "nemesis" in findings[0].title


def test_finding_toughest_rival_ignores_infrequent_opponent():
    summary = pd.DataFrame({"matches": [3], "win_rate": [0.0]}, index=["rare_opponent"])
    assert insights.finding_toughest_rival(summary) == []
