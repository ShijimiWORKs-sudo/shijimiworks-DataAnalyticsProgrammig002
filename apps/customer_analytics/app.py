"""
Customer Analytics -- ShijimiWORKs Data Analytics Portfolio, App 2/5

EC顧客分析ダッシュボード。RFM分析（Recency/Frequency/Monetary）とKMeans
クラスタリングの両方でセグメント化し、セグメントごとの推奨アクションと
自動検出された注目ポイントを表示する（docs/phase_plan.md参照）。

Run locally:
    pip install -r requirements.txt
    streamlit run apps/customer_analytics/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

from engine import stats, insights, llm, rfm  # noqa: E402

DATA_PATH = Path(__file__).parent / "data" / "sample_transactions.csv"
SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

SEGMENT_ACTIONS = {
    "優良顧客 (VIP)": "特別キャンペーン・先行案内で特別感を演出",
    "成長中の顧客": "アップセル・クロスセル提案でVIP化を後押し",
    "新規優良顧客": "2回目購入クーポンでリピート化を促進",
    "安定顧客": "満足度調査・ロイヤルティプログラムで関係を強化",
    "標準顧客": "定期的なメルマガでの関係維持・re-engagement",
    "様子見顧客": "軽いインセンティブで再訪を後押し",
    "離脱リスク（優良だが最近来ていない）": "個別クーポン・パーソナライズメールで引き戻し",
    "離脱予備軍": "リマインドメール・限定オファーで再訪を促進",
    "休眠顧客": "掘り起こしキャンペーン（大幅割引・再入荷通知等）",
}

st.set_page_config(page_title="Customer Analytics | ShijimiWORKs", page_icon="🎯", layout="wide")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date"])
    return df


def kpi_row(orders: pd.DataFrame, rfm_table: pd.DataFrame) -> None:
    total_customers = rfm_table.shape[0]
    total_revenue = orders["amount"].sum()
    total_orders = len(orders)
    avg_order_value = total_revenue / total_orders if total_orders else 0
    repeat_rate = (rfm_table["frequency"] > 1).mean() if total_customers else 0
    avg_orders_per_customer = rfm_table["frequency"].mean() if total_customers else 0

    cols = st.columns(6)
    cols[0].metric("顧客数", f"{total_customers:,}")
    cols[1].metric("総売上", f"¥{total_revenue:,.0f}")
    cols[2].metric("総注文数", f"{total_orders:,}")
    cols[3].metric("平均注文単価", f"¥{avg_order_value:,.0f}")
    cols[4].metric("リピート率", f"{repeat_rate:.1%}")
    cols[5].metric("平均購入回数/人", f"{avg_orders_per_customer:.1f}")


def main() -> None:
    st.title("🎯 Customer Analytics")
    st.caption("ShijimiWORKs Data Analytics Portfolio — App 2/5 · Shijimi AI Data Engine")

    df = load_data(DATA_PATH)

    with st.sidebar:
        st.header("フィルター")
        min_d, max_d = df["order_date"].min().date(), df["order_date"].max().date()
        date_range = st.date_input("期間", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        channels = st.multiselect("チャネル", sorted(df["channel"].unique()), default=sorted(df["channel"].unique()))

        st.header("クラスタリング")
        k = st.slider("クラスタ数 (KMeans)", min_value=2, max_value=6, value=4)

        st.header("AIレポート")
        ai_mode = st.selectbox(
            "生成モード",
            options=["none", "cloud", "local"],
            format_func=lambda m: {
                "none": "テンプレート（APIキー不要・既定）",
                "cloud": "Cloud AI (OpenAI API)",
                "local": "Local AI (自前サーバー)",
            }[m],
        )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start, end = pd.Timestamp(min_d), pd.Timestamp(max_d)

    mask = df["order_date"].between(start, end) & df["channel"].isin(channels)
    fdf = df.loc[mask].copy()

    if fdf.empty or fdf["customer_id"].nunique() < 10:
        st.warning("フィルター条件に一致する顧客数が少なすぎます（10人未満）。条件を広げてください。")
        return

    rfm_table = rfm.compute_rfm(fdf, "customer_id", "order_date", "amount", snapshot_date=end + pd.Timedelta(days=1))
    scored = rfm.score_rfm(rfm_table)

    kpi_row(fdf, rfm_table)
    st.divider()

    tab_overview, tab_rfm, tab_cluster, tab_insights = st.tabs(
        ["📊 概要", "🎯 RFMセグメント", "🤖 クラスタリング", "🔍 自動分析"]
    )

    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            monthly = fdf.set_index("order_date")["amount"].resample("MS").sum().reset_index()
            fig = px.line(monthly, x="order_date", y="amount", title="月別売上推移", markers=True)
            st.plotly_chart(fig, width="stretch")
        with c2:
            ch = fdf.groupby("channel")["amount"].sum().reset_index()
            fig = px.pie(ch, names="channel", values="amount", title="チャネル別売上構成", hole=0.4)
            st.plotly_chart(fig, width="stretch")

        freq_hist = px.histogram(
            rfm_table.reset_index(), x="frequency", nbins=30, title="購入回数の分布（顧客ベース）"
        )
        st.plotly_chart(freq_hist, width="stretch")

    with tab_rfm:
        st.subheader("セグメント分布")
        seg_counts = scored["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "customers"]
        seg_counts["推奨アクション"] = seg_counts["segment"].map(SEGMENT_ACTIONS)

        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.bar(
                seg_counts.sort_values("customers", ascending=True),
                x="customers", y="segment", orientation="h",
                title="セグメント別 顧客数",
            )
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.scatter(
                scored.reset_index(),
                x="frequency", y="monetary", color="segment", size="recency",
                hover_name="customer_id",
                title="購入頻度 x 累計購入額（バブルサイズ=最終購入からの経過日数）",
                labels={"frequency": "購入回数", "monetary": "累計購入額"},
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("セグメント別 サマリー & 推奨アクション")
        st.dataframe(seg_counts, width="stretch", hide_index=True)

        st.subheader("顧客別RFMスコア（サンプル20件）")
        sample_cols = ["recency", "frequency", "monetary", "r_score", "f_score", "m_score", "rfm_score", "segment"]
        st.dataframe(
            scored[sample_cols].sample(min(20, len(scored)), random_state=1).sort_values("m_score", ascending=False),
            width="stretch",
        )

    with tab_cluster:
        clustered = rfm.kmeans_segments(rfm_table, k=k)
        st.caption("R・F・M（標準化+対数変換）に対するKMeansクラスタリング。ラベルはクラスタの相対的な特徴から自動生成されます。")

        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.scatter(
                clustered.reset_index(),
                x="recency", y="monetary", color="cluster_profile",
                size="frequency", hover_name="customer_id",
                title="Recency x Monetary（色=クラスタ、サイズ=購入回数）",
            )
            fig.update_xaxes(autorange="reversed")  # small recency (recent) on the right
            st.plotly_chart(fig, width="stretch")
        with c2:
            cluster_summary = clustered.groupby("cluster_profile").agg(
                customers=("cluster", "count"),
                avg_recency=("recency", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"),
            ).sort_values("avg_monetary", ascending=False)
            st.dataframe(
                cluster_summary.style.format(
                    {"avg_recency": "{:.0f}日", "avg_frequency": "{:.1f}回", "avg_monetary": "¥{:,.0f}"}
                ),
                width="stretch",
            )

    with tab_insights:
        findings = []
        findings += insights.finding_segment_distribution(scored)
        findings += insights.finding_new_customer_repeat(rfm_table)

        daily_revenue = fdf.groupby("order_date", as_index=False)["amount"].sum()
        pop_monthly = stats.period_over_period(daily_revenue, "order_date", "amount", freq="MS")
        pop_monthly_complete = stats.drop_incomplete_last_month(pop_monthly, last_actual_date=fdf["order_date"].max())
        findings += insights.finding_period_change(pop_monthly_complete, label="月次")
        findings += insights.finding_anomalies(daily_revenue, "order_date", "amount", threshold=2.5)
        findings.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 9))

        st.subheader(f"検出された注目ポイント（{len(findings)}件）")
        findings_md = insights.to_markdown(findings)
        st.markdown(findings_md)

        summary = {
            "period": f"{start.date()} 〜 {end.date()}",
            "total_customers": int(rfm_table.shape[0]),
            "total_revenue": float(fdf["amount"].sum()),
            "repeat_rate": float((rfm_table["frequency"] > 1).mean()),
            "num_findings": len(findings),
        }
        report_md = (
            f"# Customer Analytics レポート\n\n"
            f"対象期間: {summary['period']}\n\n"
            f"- 顧客数: {summary['total_customers']:,}\n"
            f"- 売上: ¥{summary['total_revenue']:,.0f}\n"
            f"- リピート率: {summary['repeat_rate']:.1%}\n\n"
            f"## 検出された注目ポイント\n\n{findings_md}\n"
        )
        st.download_button(
            "📥 レポートをダウンロード (Markdown)",
            data=report_md,
            file_name=f"customer_report_{start.date()}_{end.date()}.md",
            mime="text/markdown",
        )

        st.divider()
        st.subheader("🤖 AI分析レポート")
        st.caption("上記の要約データのみをAIに渡します（生データは送信しません）。")
        if st.button("レポートを生成"):
            with st.spinner("生成中..."):
                report = llm.narrate(summary, findings_md, mode=ai_mode)
            st.markdown(report)


if __name__ == "__main__":
    main()
