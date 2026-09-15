"""
Synthetic EC (e-commerce) transaction log generator for RFM analysis.

Produces ~18 months of order history for a few thousand customers, built
from six behavioural personas (VIP / Loyal / New / At-risk-churned /
Dormant / Occasional) so that RFM scoring and clustering have real,
recognisable segments to find -- mirroring the worked example in the
portfolio plan:

    顧客 | R | F | M | 判定
    A    | 5 | 5 | 5 | VIP
    B    | 5 | 2 | 4 | 新規優良
    C    | 1 | 5 | 5 | 離脱危険
    D    | 1 | 1 | 1 | 休眠

The persona used to generate each customer is NOT written to the output
CSV (a real transaction log wouldn't have it either) -- it's only used
internally, and printed as a validation summary, to confirm the
generated data actually produces the intended RFM spread.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
END_DATE = pd.Timestamp("2026-08-31")
START_DATE = END_DATE - pd.DateOffset(months=18) + pd.Timedelta(days=1)
N_CUSTOMERS = 3000

# gap_mean: average days between orders while "active".
# start_frac / end_frac: where in the observation window (0=START_DATE,
# 1=END_DATE) this persona's activity begins/ends -- this is what drives
# Recency (a persona whose end_frac is low looks "gone" by the snapshot date).
PERSONAS = {
    "VIP":        dict(share=0.05, gap_mean=18, amount_mean=8000, amount_sigma=0.35,
                        start_frac=(0.00, 0.20), end_frac=(0.95, 1.00), max_orders=None),
    "Loyal":      dict(share=0.15, gap_mean=40, amount_mean=5000, amount_sigma=0.40,
                        start_frac=(0.00, 0.30), end_frac=(0.85, 1.00), max_orders=None),
    "New":        dict(share=0.15, gap_mean=25, amount_mean=4000, amount_sigma=0.40,
                        start_frac=(0.85, 0.95), end_frac=(0.95, 1.00), max_orders=2),
    "AtRisk":     dict(share=0.10, gap_mean=22, amount_mean=7500, amount_sigma=0.35,
                        start_frac=(0.00, 0.25), end_frac=(0.45, 0.65), max_orders=None),
    "Dormant":    dict(share=0.25, gap_mean=35, amount_mean=3000, amount_sigma=0.40,
                        start_frac=(0.00, 0.30), end_frac=(0.35, 0.45), max_orders=3),
    "Occasional": dict(share=0.30, gap_mean=75, amount_mean=3500, amount_sigma=0.45,
                        start_frac=(0.00, 0.40), end_frac=(0.60, 1.00), max_orders=None),
}

CHANNELS = ["Web", "App", "Store"]
CHANNEL_WEIGHTS = [0.55, 0.35, 0.10]


def _gen_customer_orders(rng: np.random.Generator, persona: str, cfg: dict, window_days: int):
    active_start = START_DATE + pd.Timedelta(
        days=int(rng.uniform(*cfg["start_frac"]) * window_days)
    )
    active_end = START_DATE + pd.Timedelta(
        days=int(rng.uniform(*cfg["end_frac"]) * window_days)
    )
    active_end = max(active_end, active_start + pd.Timedelta(days=1))

    dates = [active_start]
    while True:
        if cfg["max_orders"] and len(dates) >= cfg["max_orders"]:
            break
        gap_days = max(1, int(round(rng.gamma(shape=2.0, scale=cfg["gap_mean"] / 2.0))))
        next_date = dates[-1] + pd.Timedelta(days=gap_days)
        if next_date > active_end or next_date > END_DATE:
            break
        dates.append(next_date)

    mu = np.log(cfg["amount_mean"]) - 0.5 * cfg["amount_sigma"] ** 2
    amounts = rng.lognormal(mean=mu, sigma=cfg["amount_sigma"], size=len(dates))
    return dates, amounts


def generate(seed: int = SEED, n_customers: int = N_CUSTOMERS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    window_days = (END_DATE - START_DATE).days

    persona_names = list(PERSONAS.keys())
    persona_probs = [PERSONAS[p]["share"] for p in persona_names]
    assigned = rng.choice(persona_names, size=n_customers, p=persona_probs)

    rows = []
    persona_of = {}
    for i, persona in enumerate(assigned):
        customer_id = f"C{i + 1:05d}"
        persona_of[customer_id] = persona
        cfg = PERSONAS[persona]
        dates, amounts = _gen_customer_orders(rng, persona, cfg, window_days)
        channels = rng.choice(CHANNELS, size=len(dates), p=CHANNEL_WEIGHTS)
        for d, amt, ch in zip(dates, amounts, channels):
            rows.append({"customer_id": customer_id, "order_date": d, "amount": round(float(amt), 0), "channel": ch})

    df = pd.DataFrame(rows).sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    return df, persona_of


def _validate(df: pd.DataFrame, persona_of: dict) -> None:
    """Print a quick sanity summary of recency/frequency/monetary by persona (not saved to CSV)."""
    snapshot = df["order_date"].max() + pd.Timedelta(days=1)
    g = df.groupby("customer_id").agg(
        recency=("order_date", lambda s: (snapshot - s.max()).days),
        frequency=("order_date", "count"),
        monetary=("amount", "sum"),
    )
    g["persona"] = g.index.map(persona_of)
    print(g.groupby("persona")[["recency", "frequency", "monetary"]].mean().round(0))


def main() -> None:
    df, persona_of = generate()
    out_path = Path(__file__).parent / "data" / "sample_transactions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {len(df):,} orders across {df['customer_id'].nunique():,} customers -> {out_path}")
    print(f"date range: {df['order_date'].min().date()} .. {df['order_date'].max().date()}")
    _validate(df, persona_of)


if __name__ == "__main__":
    main()
