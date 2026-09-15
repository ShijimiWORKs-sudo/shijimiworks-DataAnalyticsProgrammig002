"""
engine.forecast
================

Lightweight, fully-explainable forecasting for a daily time series that
has weekly + monthly seasonality (occupancy rate, daily sales, ...).

Uses `sklearn.linear_model.LinearRegression` on a day-index trend term
plus weekday/month dummy variables -- no black-box model needed for a
portfolio-scale "next month" forecast, and every coefficient can be
explained in plain language if asked. Takes/returns plain pandas objects
so it's reusable by any app with a daily date + numeric value column
(hotel occupancy today; sales/darts-score trend tomorrow).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

_ALL_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_ALL_MONTHS = list(range(1, 13))
_Z_FOR_INTERVAL = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}


@dataclass
class ForecastResult:
    forecast: pd.DataFrame  # indexed by date, columns: [predicted, lower, upper]
    r2: float               # in-sample fit quality (not a guarantee of future accuracy)
    trend_per_day: float    # regression coefficient on the day-index term


def seasonal_regression_forecast(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    horizon_days: int = 30,
    interval: float = 0.80,
) -> ForecastResult:
    """
    Fit day-index + weekday-dummy + month-dummy linear regression on a
    daily series and forecast the next `horizon_days` days.

    Returns a point forecast plus an `interval`-level prediction band
    (default 80%, derived from in-sample residual std -- a simple normal
    approximation, not a formal prediction interval) and the in-sample
    R^2 as a rough fit-quality signal to show alongside the forecast.
    """
    hist = df[[date_col, value_col]].dropna().sort_values(date_col)
    if len(hist) < 14:
        raise ValueError("forecast には最低14日分の実績データが必要です。")

    dates = pd.DatetimeIndex(hist[date_col])
    day0 = dates.min()
    future_dates = pd.date_range(dates.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    # Build weekday/month dummies for train + future together against a fixed
    # category set, so the two design matrices always line up column-for-column.
    combined = dates.append(future_dates)
    day_index = (combined - day0).days.to_numpy(dtype=float).reshape(-1, 1)
    weekday_dummies = pd.get_dummies(
        pd.Categorical(combined.day_name(), categories=_ALL_WEEKDAYS), prefix="wd", drop_first=True
    )
    month_dummies = pd.get_dummies(
        pd.Categorical(combined.month, categories=_ALL_MONTHS), prefix="m", drop_first=True
    )
    X_all = np.hstack([day_index, weekday_dummies.to_numpy(dtype=float), month_dummies.to_numpy(dtype=float)])

    n_train = len(dates)
    X_train, X_future = X_all[:n_train], X_all[n_train:]
    y_train = hist[value_col].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = float(model.score(X_train, y_train))

    resid = y_train - model.predict(X_train)
    resid_std = float(resid.std(ddof=1)) if len(resid) > 2 else 0.0
    z = _Z_FOR_INTERVAL.get(round(interval, 2), 1.2816)

    predicted = model.predict(X_future)
    forecast = pd.DataFrame(
        {
            "predicted": predicted,
            "lower": predicted - z * resid_std,
            "upper": predicted + z * resid_std,
        },
        index=future_dates,
    )
    return ForecastResult(forecast=forecast, r2=r2, trend_per_day=float(model.coef_[0]))
