"""
engine.rfm
==========

RFM (Recency / Frequency / Monetary) analysis, plus a KMeans-based
alternative segmentation -- the two "judgement" methods Customer
Analytics is built around (see docs in the app for the worked example).

Two ways to turn R/F/M into a segment are provided:

  score_rfm()      -- classic quintile scoring (1-5 per dimension) +
                       a transparent, rule-based 3x3 label (fast, fully
                       explainable, no ML).
  kmeans_segments() -- scikit-learn KMeans on standardized R/F/M, with
                       auto-generated (not hardcoded) cluster profiles --
                       this is the "machine learning" half of the engine.

Both take the output of compute_rfm() and never need raw order-level
data again, keeping them reusable for any transaction log with a
customer id, an order date, and an amount.
"""

from __future__ import annotations

import pandas as pd


def compute_rfm(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    amount_col: str,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Collapse an order-level transaction log into one row per customer:
    recency (days since last order, as of snapshot_date), frequency
    (order count), monetary (total spend).
    """
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_col).agg(
        recency=(date_col, lambda s: (snapshot_date - s.max()).days),
        frequency=(date_col, "count"),
        monetary=(amount_col, "sum"),
    )
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]
    return rfm


def _quintile_score(series: pd.Series, ascending: bool) -> pd.Series:
    """
    1-5 score via quantile bins. ascending=False means "higher value is
    better" (frequency, monetary); ascending=True means "lower value is
    better" (recency -- fewer days since last order is better).
    Falls back gracefully when there aren't enough distinct values for 5
    clean bins (small/synthetic datasets sometimes tie a lot).
    """
    try:
        ranks = pd.qcut(series.rank(method="first"), 5, labels=False, duplicates="drop")
    except ValueError:
        ranks = pd.qcut(series.rank(method="first"), 2, labels=False, duplicates="drop")
    ranks = ranks.astype(float)
    max_rank = ranks.max()
    scaled = 1 + (ranks / max_rank) * 4 if max_rank > 0 else pd.Series(3, index=series.index)
    scaled = scaled.round().astype(int)
    return (6 - scaled) if ascending else scaled


def segment_label(r_score: int, f_score: int) -> str:
    """
    A transparent, rule-based 3x3 segment label from R/F scores (1-5
    each; M is reported alongside but not required to name the segment,
    keeping the rule simple and explainable to a non-technical reader).
    """
    r_bucket = "high" if r_score >= 4 else ("mid" if r_score == 3 else "low")
    f_bucket = "high" if f_score >= 4 else ("mid" if f_score == 3 else "low")

    labels = {
        ("high", "high"): "優良顧客 (VIP)",
        ("high", "mid"): "成長中の顧客",
        ("high", "low"): "新規優良顧客",
        ("mid", "high"): "安定顧客",
        ("mid", "mid"): "標準顧客",
        ("mid", "low"): "様子見顧客",
        ("low", "high"): "離脱リスク（優良だが最近来ていない）",
        ("low", "mid"): "離脱予備軍",
        ("low", "low"): "休眠顧客",
    }
    return labels[(r_bucket, f_bucket)]


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Attach r_score/f_score/m_score (1-5) and a rule-based segment label."""
    out = rfm.copy()
    out["r_score"] = _quintile_score(out["recency"], ascending=True)
    out["f_score"] = _quintile_score(out["frequency"], ascending=False)
    out["m_score"] = _quintile_score(out["monetary"], ascending=False)
    out["rfm_score"] = out["r_score"].astype(str) + out["f_score"].astype(str) + out["m_score"].astype(str)
    out["segment"] = [segment_label(r, f) for r, f in zip(out["r_score"], out["f_score"])]
    return out


def kmeans_segments(rfm: pd.DataFrame, k: int = 4, random_state: int = 42) -> pd.DataFrame:
    """
    Alternative, ML-based segmentation: standardize [recency (inverted),
    frequency, log1p(monetary)] and run KMeans. Cluster labels are
    auto-generated from each cluster's centroid ranking (highest combined
    "value" -> most positive label) rather than hardcoded, so this stays
    reusable for any RFM table, not just this dataset.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    features = pd.DataFrame(
        {
            "recency_inv": -rfm["recency"],
            "frequency": rfm["frequency"],
            "monetary_log": np.log1p(rfm["monetary"]),
        },
        index=rfm.index,
    )
    X = StandardScaler().fit_transform(features)

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_id = model.fit_predict(X)

    out = rfm.copy()
    out["cluster"] = cluster_id

    # Rank clusters by a simple composite "value" score (recency_inv + frequency + monetary_log,
    # each z-scored via the same StandardScaler output) so labels reflect relative standing,
    # not absolute thresholds -- this is what makes the labeling reusable across datasets.
    composite = X.sum(axis=1)
    cluster_value = pd.Series(composite, index=rfm.index).groupby(cluster_id).mean().sort_values(ascending=False)
    rank_of_cluster = {cid: rank for rank, cid in enumerate(cluster_value.index)}

    n = len(cluster_value)
    def _profile_label(rank: int) -> str:
        if n <= 1:
            return "全顧客"
        if rank == 0:
            return "上位顧客クラスタ（高頻度・高単価・直近）"
        if rank == n - 1:
            return "休眠クラスタ（低頻度・低単価・長期未購入）"
        frac = rank / (n - 1)
        return "中位顧客クラスタ（やや優良）" if frac < 0.5 else "中位顧客クラスタ（要フォロー）"

    out["cluster_profile"] = out["cluster"].map(lambda c: _profile_label(rank_of_cluster[c]))
    return out
