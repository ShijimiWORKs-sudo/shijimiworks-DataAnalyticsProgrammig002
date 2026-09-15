"""Unit tests for engine.forecast (run with: pytest)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from engine import forecast  # noqa: E402


def _synthetic_series(n_days: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    weekday_boost = np.where(dates.weekday.isin([4, 5]), 0.15, 0.0)
    trend = np.linspace(0, 0.05, n_days)
    noise = rng.normal(0, 0.02, size=n_days)
    value = np.clip(0.6 + weekday_boost + trend + noise, 0, 1)
    return pd.DataFrame({"date": dates, "occupancy_rate": value})


def test_seasonal_regression_forecast_returns_expected_shape():
    df = _synthetic_series()
    result = forecast.seasonal_regression_forecast(df, "date", "occupancy_rate", horizon_days=30)
    assert len(result.forecast) == 30
    assert list(result.forecast.columns) == ["predicted", "lower", "upper"]
    assert (result.forecast["lower"] <= result.forecast["predicted"]).all()
    assert (result.forecast["predicted"] <= result.forecast["upper"]).all()


def test_seasonal_regression_forecast_continues_dates_from_history():
    df = _synthetic_series()
    result = forecast.seasonal_regression_forecast(df, "date", "occupancy_rate", horizon_days=10)
    assert result.forecast.index[0] == df["date"].max() + pd.Timedelta(days=1)


def test_seasonal_regression_forecast_captures_weekday_pattern():
    df = _synthetic_series()
    result = forecast.seasonal_regression_forecast(df, "date", "occupancy_rate", horizon_days=14)
    fri_sat = result.forecast[result.forecast.index.day_name().isin(["Friday", "Saturday"])]
    other = result.forecast[~result.forecast.index.day_name().isin(["Friday", "Saturday"])]
    assert fri_sat["predicted"].mean() > other["predicted"].mean()


def test_seasonal_regression_forecast_positive_trend_detected():
    df = _synthetic_series()
    result = forecast.seasonal_regression_forecast(df, "date", "occupancy_rate", horizon_days=10)
    assert result.trend_per_day > 0
    assert 0 <= result.r2 <= 1


def test_seasonal_regression_forecast_raises_on_too_little_history():
    df = _synthetic_series(n_days=5)
    with pytest.raises(ValueError):
        forecast.seasonal_regression_forecast(df, "date", "occupancy_rate", horizon_days=10)
