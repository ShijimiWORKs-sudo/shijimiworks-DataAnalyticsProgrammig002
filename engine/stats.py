"""
engine.stats
============

Reusable statistical primitives: descriptive stats, period-over-period
change, and two classic outlier-detection methods (z-score, IQR).

Every function here takes/returns plain pandas/NumPy objects so it can be
reused by any app (sales, hotel, darts, ...) without modification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Anomaly:
    index: object          # the DataFrame/Series index label of the anomalous row
    value: float
    z_score: float
    method: str             # "zscore" | "iqr"


def describe(series: pd.Series) -> dict:
    """Basic descriptive statistics for a numeric series."""
    s = series.dropna().astype(float)
    if s.empty:
        return {"count": 0}
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
    }


def pct_change(current: float, previous: float) -> float | None:
    """Percent change from previous -> current. None if previous is 0/NaN."""
    if previous in (0, None) or pd.isna(previous):
        return None
    return (current - previous) / previous


def period_over_period(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "MS",
) -> pd.DataFrame:
    """
    Resample a daily time series to `freq` (default: month start) and
    compute period-over-period (MoM by default) change.

    Returns a DataFrame indexed by period with columns:
    [value, prev_value, pct_change]
    """
    ts = (
        df[[date_col, value_col]]
        .dropna()
        .set_index(date_col)[value_col]
        .resample(freq)
        .sum()
    )
    out = pd.DataFrame({"value": ts})
    out["prev_value"] = out["value"].shift(1)
    out["pct_change"] = (out["value"] - out["prev_value"]) / out["prev_value"]
    return out


def drop_incomplete_last_month(pop_df: pd.DataFrame, last_actual_date: pd.Timestamp) -> pd.DataFrame:
    """
    Drop the trailing row of a period_over_period(freq="MS") table if the
    underlying data doesn't actually reach the end of that calendar month
    (e.g. the log's last order was Aug 26 -- comparing a partial "August"
    against a full July would read as a fake decline). Safe to call even
    when the last bucket IS complete: it's then a no-op.
    """
    if pop_df.empty:
        return pop_df
    last_bucket_start = pop_df.index[-1]
    month_end = (last_bucket_start + pd.offsets.MonthEnd(0)).normalize()
    if pd.Timestamp(last_actual_date).normalize() < month_end:
        return pop_df.iloc[:-1]
    return pop_df


def year_over_year(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Monthly totals with a same-month-last-year comparison column."""
    ts = (
        df[[date_col, value_col]]
        .dropna()
        .set_index(date_col)[value_col]
        .resample("MS")
        .sum()
    )
    out = pd.DataFrame({"value": ts})
    out["prev_year_value"] = out["value"].shift(12)
    out["yoy_pct_change"] = (out["value"] - out["prev_year_value"]) / out["prev_year_value"]
    return out


def detect_anomalies_zscore(series: pd.Series, threshold: float = 2.5) -> list[Anomaly]:
    """Flag points whose |z-score| exceeds `threshold`."""
    s = series.dropna().astype(float)
    if s.std(ddof=0) == 0 or len(s) < 3:
        return []
    z = (s - s.mean()) / s.std(ddof=0)
    hits = z[z.abs() >= threshold]
    return [
        Anomaly(index=idx, value=float(s.loc[idx]), z_score=float(z.loc[idx]), method="zscore")
        for idx in hits.index
    ]


def detect_anomalies_iqr(series: pd.Series, k: float = 1.5) -> list[Anomaly]:
    """Flag points outside [Q1 - k*IQR, Q3 + k*IQR] (Tukey's fences)."""
    s = series.dropna().astype(float)
    if len(s) < 4:
        return []
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return []
    lo, hi = q1 - k * iqr, q3 + k * iqr
    hits = s[(s < lo) | (s > hi)]
    mean, std = s.mean(), s.std(ddof=0) or 1.0
    return [
        Anomaly(
            index=idx,
            value=float(s.loc[idx]),
            z_score=float((s.loc[idx] - mean) / std),
            method="iqr",
        )
        for idx in hits.index
    ]


def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Pearson correlation matrix for the given numeric columns."""
    return df[columns].corr(numeric_only=True)


def rank_with_percentile(series: pd.Series, ascending: bool = False) -> pd.DataFrame:
    """
    Rank entries (e.g. stores by sales) and attach each one's percentile,
    used to phrase findings like '売上ランキング2位/全店舗中8位' (rank X of Y).
    """
    s = series.dropna().astype(float)
    out = pd.DataFrame({"value": s})
    out["rank"] = s.rank(ascending=ascending, method="min").astype(int)
    out["percentile"] = s.rank(pct=True, ascending=ascending)
    out = out.sort_values("rank")
    return out


def moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    """Simple trailing moving average, used to separate signal from day-to-day noise."""
    return series.rolling(window=window, min_periods=max(1, window // 2)).mean()


def quadrant_classify(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    labels: dict[str, str] | None = None,
    x_split: float | None = None,
    y_split: float | None = None,
) -> pd.Series:
    """
    Generic 2x2 quadrant classifier (median split by default), reusable for
    any "volume x quality" comparison: sales x margin (stores/categories),
    occupancy x ADR (hotel), volume x accuracy (darts), etc.

    Returns a Series of labels aligned to df.index:
        high x / high y -> labels["good"]        (default: "優良")
        high x / low  y -> labels["warning"]      (default: "要注意": 量は多いが質が低い)
        low  x / high y -> labels["efficient"]    (default: "堅実": 量は少ないが質は高い)
        low  x / low  y -> labels["needs_work"]   (default: "要改善")
    """
    default_labels = {
        "good": "優良",
        "warning": "要注意",
        "efficient": "堅実",
        "needs_work": "要改善",
    }
    labels = {**default_labels, **(labels or {})}

    x_split = df[x_col].median() if x_split is None else x_split
    y_split = df[y_col].median() if y_split is None else y_split

    def _label(row) -> str:
        high_x = row[x_col] >= x_split
        high_y = row[y_col] >= y_split
        if high_x and high_y:
            return labels["good"]
        if high_x and not high_y:
            return labels["warning"]
        if not high_x and high_y:
            return labels["efficient"]
        return labels["needs_work"]

    return df.apply(_label, axis=1)
