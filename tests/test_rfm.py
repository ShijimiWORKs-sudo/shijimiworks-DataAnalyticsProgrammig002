"""Unit tests for engine.rfm (run with: pytest)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from engine import rfm  # noqa: E402


def _sample_orders() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # A: recent, frequent, high spend -> should score high on all three
            {"customer_id": "A", "order_date": "2026-08-01", "amount": 10000},
            {"customer_id": "A", "order_date": "2026-08-15", "amount": 12000},
            {"customer_id": "A", "order_date": "2026-08-25", "amount": 11000},
            {"customer_id": "A", "order_date": "2026-08-28", "amount": 9000},
            # D: single order long ago, low spend -> should score low on all three
            {"customer_id": "D", "order_date": "2025-04-01", "amount": 500},
        ]
    )


def _orders_df() -> pd.DataFrame:
    df = _sample_orders()
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def test_compute_rfm_shape_and_values():
    df = _orders_df()
    r = rfm.compute_rfm(df, "customer_id", "order_date", "amount", snapshot_date=pd.Timestamp("2026-08-31"))
    assert set(r.index) == {"A", "D"}
    assert r.loc["A", "frequency"] == 4
    assert r.loc["A", "monetary"] == 42000
    assert r.loc["D", "frequency"] == 1
    assert r.loc["A", "recency"] < r.loc["D", "recency"]


def test_score_rfm_orders_best_and_worst_customer_correctly():
    df = _orders_df()
    r = rfm.compute_rfm(df, "customer_id", "order_date", "amount", snapshot_date=pd.Timestamp("2026-08-31"))
    scored = rfm.score_rfm(r)
    assert scored.loc["A", "r_score"] >= scored.loc["D", "r_score"]
    assert scored.loc["A", "f_score"] >= scored.loc["D", "f_score"]
    assert scored.loc["A", "m_score"] >= scored.loc["D", "m_score"]


def test_segment_label_covers_all_3x3_combinations():
    for r_score in range(1, 6):
        for f_score in range(1, 6):
            label = rfm.segment_label(r_score, f_score)
            assert isinstance(label, str) and label


def test_segment_label_vip_and_dormant():
    assert rfm.segment_label(5, 5) == "優良顧客 (VIP)"
    assert rfm.segment_label(1, 1) == "休眠顧客"
    assert rfm.segment_label(5, 1) == "新規優良顧客"
    assert rfm.segment_label(1, 5) == "離脱リスク（優良だが最近来ていない）"


def test_kmeans_segments_runs_and_separates_extremes():
    # Build a slightly bigger synthetic RFM table so KMeans has enough points.
    import numpy as np

    rng = np.random.default_rng(0)
    n = 60
    vip = pd.DataFrame(
        {
            "recency": rng.integers(1, 10, size=n // 3),
            "frequency": rng.integers(15, 30, size=n // 3),
            "monetary": rng.integers(80000, 150000, size=n // 3),
        }
    )
    dormant = pd.DataFrame(
        {
            "recency": rng.integers(300, 500, size=n // 3),
            "frequency": rng.integers(1, 3, size=n // 3),
            "monetary": rng.integers(500, 3000, size=n // 3),
        }
    )
    mid = pd.DataFrame(
        {
            "recency": rng.integers(50, 150, size=n - 2 * (n // 3)),
            "frequency": rng.integers(4, 10, size=n - 2 * (n // 3)),
            "monetary": rng.integers(10000, 30000, size=n - 2 * (n // 3)),
        }
    )
    full = pd.concat([vip, dormant, mid], ignore_index=True)
    full.index = [f"C{i}" for i in range(len(full))]

    clustered = rfm.kmeans_segments(full, k=3)
    assert "cluster" in clustered.columns
    assert "cluster_profile" in clustered.columns
    assert clustered["cluster"].nunique() == 3

    vip_ids = full.index[: n // 3]
    dormant_ids = full.index[n // 3 : 2 * (n // 3)]
    assert clustered.loc[vip_ids, "cluster_profile"].mode()[0] != clustered.loc[dormant_ids, "cluster_profile"].mode()[0]
