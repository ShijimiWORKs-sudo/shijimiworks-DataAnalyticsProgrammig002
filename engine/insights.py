"""
engine.insights
===============

Turns statistical output (rankings, anomalies, period-over-period change)
into the "so what?" layer the plan calls for:

    分析結果 (what happened) -> 改善提案 (what to do about it)

Every function returns plain `Finding` objects (dataclasses) so any app
can render them as cards, or hand them to engine.llm.narrate() for a
prose write-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import stats


@dataclass
class Finding:
    title: str
    detail: str
    recommendation: str
    severity: str = "info"  # "info" | "warning" | "critical"
    tags: list[str] = field(default_factory=list)


def finding_sales_vs_margin_gap(
    store_summary: pd.DataFrame,
    sales_col: str = "sales",
    margin_col: str = "gross_margin",
    gap_threshold_ranks: int = 3,
) -> list[Finding]:
    """
    Flags stores that rank well on sales but poorly on gross margin
    (or vice versa) -- the "A店は売上ランキング2位だが粗利益率は8位" pattern.
    """
    sales_rank = stats.rank_with_percentile(store_summary[sales_col], ascending=False)
    margin_rank = stats.rank_with_percentile(store_summary[margin_col], ascending=False)
    n = len(store_summary)

    findings: list[Finding] = []
    for store in store_summary.index:
        s_rank = int(sales_rank.loc[store, "rank"])
        m_rank = int(margin_rank.loc[store, "rank"])
        gap = m_rank - s_rank
        if s_rank <= max(2, n // 3) and gap >= gap_threshold_ranks:
            sales_val = store_summary.loc[store, sales_col]
            margin_val = store_summary.loc[store, margin_col]
            findings.append(
                Finding(
                    title=f"{store}: 売上は好調だが粗利益率が低い",
                    detail=(
                        f"{store}は売上ランキング{s_rank}位/全{n}店舗中だが、"
                        f"粗利益率は{m_rank}位（売上 {sales_val:,.0f}円、粗利益率 {margin_val:.1%}）。"
                    ),
                    recommendation=(
                        f"{store}の商品構成を確認し、原価率の高い商品比率を下げるか、"
                        "セット販売・価格設計を見直して利益率の高い商品への誘導を検討。"
                    ),
                    severity="warning",
                    tags=["store", "margin"],
                )
            )
    return findings


def finding_weekday_concentration(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    concentration_threshold: float = 0.20,
) -> list[Finding]:
    """Flags a single weekday (e.g. Friday) that carries an outsized share of total value."""
    tmp = df[[date_col, value_col]].dropna().copy()
    tmp["weekday"] = tmp[date_col].dt.day_name()
    by_weekday = tmp.groupby("weekday")[value_col].sum()
    total = by_weekday.sum()
    if total == 0:
        return []
    share = (by_weekday / total).sort_values(ascending=False)
    top_day, top_share = share.index[0], share.iloc[0]

    findings = []
    if top_share >= concentration_threshold:
        findings.append(
            Finding(
                title=f"{top_day}に売上が集中",
                detail=f"{top_day}が全体の{top_share:.1%}を占め、曜日の中で最も高い。",
                recommendation=(
                    f"{top_day}向けの在庫・人員配置を厚くする一方、"
                    "他の曜日への送客施策（クーポン・限定メニュー等）を検討。"
                ),
                severity="info",
                tags=["weekday", "seasonality"],
            )
        )
    return findings


def finding_category_margin(
    category_summary: pd.DataFrame,
    sales_col: str = "sales",
    margin_col: str = "gross_margin",
    low_margin_threshold: float = None,
) -> list[Finding]:
    """
    Flags categories that sell well but carry low gross margin
    -- the "B商品の売上は高いが粗利益率が低い" pattern.
    """
    if low_margin_threshold is None:
        low_margin_threshold = category_summary[margin_col].median()

    sales_median = category_summary[sales_col].median()
    findings = []
    for cat, row in category_summary.iterrows():
        if row[sales_col] >= sales_median and row[margin_col] < low_margin_threshold:
            findings.append(
                Finding(
                    title=f"{cat}: 売上は高いが粗利益率が低い",
                    detail=(
                        f"{cat}の売上は{row[sales_col]:,.0f}円（中央値以上）だが、"
                        f"粗利益率は{row[margin_col]:.1%}で全カテゴリ中央値を下回る。"
                    ),
                    recommendation=f"{cat}の仕入れ条件見直し、または高利益率商品とのセット化を検討。",
                    severity="warning",
                    tags=["category", "margin"],
                )
            )
    return findings


def finding_period_change(
    pop_df: pd.DataFrame,
    label: str = "月次",
    pct_col: str = "pct_change",
    notable_threshold: float = 0.15,
) -> list[Finding]:
    """Flags the most recent period-over-period change if it's notably large (up or down)."""
    valid = pop_df.dropna(subset=[pct_col])
    if valid.empty:
        return []
    last_period = valid.index[-1]
    change = valid.loc[last_period, pct_col]
    if abs(change) < notable_threshold:
        return []
    direction = "増加" if change > 0 else "減少"
    severity = "info" if change > 0 else "critical"
    return [
        Finding(
            title=f"直近{label}で{abs(change):.1%}の{direction}",
            detail=f"{last_period}の値は前{label}比{change:+.1%}。",
            recommendation=(
                "この変化の要因（特定の店舗・商品・曜日への偏り等）を分解して確認することを推奨。"
                if change < 0
                else "好調要因を特定し、他店舗・他カテゴリへ横展開できないか検討。"
            ),
            severity=severity,
            tags=["trend", label],
        )
    ]


def finding_anomalies(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    threshold: float = 2.5,
) -> list[Finding]:
    """Turns statistical anomalies (z-score) in a daily series into findings."""
    series = df.set_index(date_col)[value_col]
    anomalies = stats.detect_anomalies_zscore(series, threshold=threshold)
    findings = []
    for a in anomalies:
        direction = "急増" if a.z_score > 0 else "急減"
        findings.append(
            Finding(
                title=f"{a.index:%Y-%m-%d} に{value_col}が{direction}",
                detail=f"値: {a.value:,.0f}（z-score: {a.z_score:+.2f}）。",
                recommendation="該当日のキャンペーン・天候・イベント等の外部要因を確認。",
                severity="warning",
                tags=["anomaly"],
            )
        )
    return findings


def finding_segment_distribution(
    rfm_scored: pd.DataFrame,
    segment_col: str = "segment",
    dormant_labels: tuple[str, ...] = ("休眠顧客", "離脱予備軍"),
    at_risk_labels: tuple[str, ...] = ("離脱リスク（優良だが最近来ていない）",),
    dormant_share_threshold: float = 0.30,
) -> list[Finding]:
    """
    Flags when a large share of the customer base has gone quiet, and
    separately calls out customers who used to be high-value but have
    stopped buying -- the two patterns a retention program needs to know
    about first.
    """
    total = len(rfm_scored)
    if total == 0:
        return []
    counts = rfm_scored[segment_col].value_counts()
    findings = []

    dormant_n = counts.reindex(dormant_labels, fill_value=0).sum()
    dormant_share = dormant_n / total
    if dormant_share >= dormant_share_threshold:
        findings.append(
            Finding(
                title="休眠・離脱予備軍が顧客全体の大きな割合を占める",
                detail=f"「休眠顧客」「離脱予備軍」が合計{dormant_n:,}人（全体の{dormant_share:.1%}）。",
                recommendation="休眠顧客向けの掘り起こしキャンペーン（限定クーポン・再入荷通知等）を優先的に検討。",
                severity="warning",
                tags=["rfm", "dormant"],
            )
        )

    at_risk_n = counts.reindex(at_risk_labels, fill_value=0).sum()
    if at_risk_n > 0:
        at_risk_monetary = rfm_scored.loc[rfm_scored[segment_col].isin(at_risk_labels), "monetary"].sum()
        findings.append(
            Finding(
                title=f"離脱リスクの高い優良顧客が{at_risk_n:,}人",
                detail=f"過去の購入額合計は{at_risk_monetary:,.0f}円に上るが、最近の購入が途絶えている。",
                recommendation="離脱リスク顧客には個別クーポンやパーソナライズしたリマインドメールで再訪を促す。",
                severity="critical",
                tags=["rfm", "at_risk"],
            )
        )
    return findings


def finding_new_customer_repeat(
    rfm: pd.DataFrame,
    frequency_col: str = "frequency",
    one_time_threshold: int = 1,
) -> list[Finding]:
    """Flags a high share of one-time buyers -- a repeat-purchase / onboarding opportunity."""
    total = len(rfm)
    if total == 0:
        return []
    one_time = int((rfm[frequency_col] <= one_time_threshold).sum())
    share = one_time / total
    if share < 0.25:
        return []
    return [
        Finding(
            title="一度しか購入していない顧客が多い",
            detail=f"全顧客{total:,}人中{one_time:,}人（{share:.1%}）が購入1回のみ。",
            recommendation="初回購入後のフォローメール・2回目購入クーポンなど、リピート促進施策を検討。",
            severity="info",
            tags=["rfm", "repeat_purchase"],
        )
    ]


def to_markdown(findings: list[Finding]) -> str:
    """Render findings as a Markdown report (used by the Streamlit apps and for exports)."""
    if not findings:
        return "特筆すべき異常・傾向は検出されませんでした。"
    lines = []
    icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🔴"}
    for f in findings:
        lines.append(f"### {icon.get(f.severity, '')} {f.title}")
        lines.append(f"**分析結果:** {f.detail}")
        lines.append(f"**改善提案:** {f.recommendation}")
        lines.append("")
    return "\n".join(lines)
