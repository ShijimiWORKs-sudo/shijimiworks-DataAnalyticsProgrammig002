# ShijimiWORKs Data Analytics Portfolio — Customer Analytics

ShijimiWORKs Data Analyticsポートフォリオの2本目のアプリ。EC顧客の購買データからRFM分析と
機械学習（KMeansクラスタリング）の両方で顧客をセグメント化し、セグメントごとの推奨アクションと
自動検出された注目ポイントを表示する。

ポートフォリオ全体（1本目のSales Analytics、共通の分析基盤の設計方針など）は
[shijimiworks-DataAnalyticsProgrammig001](https://github.com/ShijimiWORKs-sudo/shijimiworks-DataAnalyticsProgrammig001)
を参照。このリポジトリはStreamlit Community Cloudでの単独デプロイのため、
同じ`engine/`（Shijimi AI Data Engine）を自己完結する形で同梱している。

## Customer Analytics (`apps/customer_analytics`)

- **RFM分析**: 直近購入日（Recency）・購入頻度（Frequency）・累計購入額（Monetary）を
  5段階でスコアリングし、「優良顧客(VIP)」「新規優良顧客」「離脱リスク」「休眠顧客」等の
  9セグメントに分類。セグメントごとの推奨アクション（VIP→特別キャンペーン、離脱リスク→
  個別クーポン、新規→2回目購入促進 等）を併記する。
- **KMeansクラスタリング**: scikit-learnのKMeansで標準化・対数変換したR/F/Mをクラスタ化する
  もう一つの切り口。クラスタのラベルは相対的な特徴（他クラスタと比べて高頻度・高単価か等）から
  自動生成されるため、他のデータセットにもそのまま再利用できる汎用実装（`engine/rfm.py`）。
- **自動分析**: 休眠顧客の割合、離脱リスクの高い優良顧客数、月次トレンドの急変、異常値を
  自動検出し、改善提案とともにMarkdownレポートとしてダウンロード可能。

```bash
pip install -r requirements.txt

# サンプルデータを再生成する場合（既に data/sample_transactions.csv を同梱）
python apps/customer_analytics/data_gen.py

# ダッシュボードを起動
streamlit run apps/customer_analytics/app.py
```

サンプルデータは6種類の顧客ペルソナ（VIP・ロイヤル・新規・離脱リスク・休眠・不定期）を
もとに18ヶ月分の購買履歴として合成生成しており（`data_gen.py`、seed固定・再現可能）、
RFM分析・クラスタリングの両方が意図した通りにセグメントを分離できることをテストで確認済み。

サイドバーの「AIレポート」で、要約データのみをもとにした自然言語レポートを生成できる
（既定はAPIキー不要のテンプレートモード。`OPENAI_API_KEY` を設定すればCloud AIモードも利用可能）。

```bash
pytest tests/
```

## Shijimi AI Data Engine

- `engine/stats.py` — 記述統計、前月比/前年比、z-score・IQR異常検知、ランキング、移動平均、
  2x2クアドラント分類、不完全な最終月を除外するヘルパー
- `engine/rfm.py` — RFM集計・スコアリング・セグメントラベリング、KMeansクラスタリング
- `engine/insights.py` — 統計結果から「発見 + 改善提案」を生成
- `engine/llm.py` — 発見を人間向けの文章にする（`none`=テンプレート / `cloud`=OpenAI API / `local`=自前LLMサーバー）
