"""
engine.llm
==========

The "narration" layer. Judgement happens upstream in engine.stats /
engine.insights -- this module only turns an already-computed, small
JSON-like summary into human-readable Japanese prose.

Three modes, selectable at call time (never required):

  mode="none"  (default) - template-based narrative, zero cost, zero
               dependencies, always available. This is what the apps use
               out of the box so the portfolio demo never needs a key.
  mode="cloud" - calls the OpenAI API (needs OPENAI_API_KEY). Only the
               small `summary` dict is sent, never raw rows -- see
               docs/phase_plan.md "LLMに送るのは要約だけ" principle.
  mode="local" - calls a local OpenAI-compatible endpoint (e.g. llama.cpp
               server / Ollama) via OPENAI_BASE_URL, for the "ローカルAI"
               track. No external network call, no API cost.

This module never sends full datasets to any API -- only the compact
summary dict the caller builds (a few numbers + a handful of finding
strings), which keeps cloud-mode cost negligible even on a free-tier key.
"""

from __future__ import annotations

import os


def _template_narrative(summary: dict, findings_md: str) -> str:
    """Zero-dependency fallback narrative, built from the summary dict."""
    parts = []
    period = summary.get("period", "対象期間")
    parts.append(f"【{period}のサマリー】")

    for key, val in summary.items():
        if key == "period":
            continue
        if isinstance(val, float):
            if abs(val) < 1:
                parts.append(f"- {key}: {val:+.1%}")
            else:
                parts.append(f"- {key}: {val:,.1f}")
        else:
            parts.append(f"- {key}: {val}")

    parts.append("")
    parts.append("【検出された注目ポイント】")
    parts.append(findings_md)
    return "\n".join(parts)


def _cloud_narrative(summary: dict, findings_md: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "openai パッケージが未インストールです。`pip install openai` を実行してください。"
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が設定されていません。")

    client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL") or None)
    prompt = (
        "あなたはデータアナリストです。以下のJSON形式の分析サマリーと検出済みの注目ポイントをもとに、"
        "経営者向けの分かりやすい日本語レポートを300字程度で書いてください。"
        "数値を捏造せず、与えられた情報のみを根拠にしてください。\n\n"
        f"サマリー: {summary}\n\n検出済みポイント:\n{findings_md}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


def _local_narrative(summary: dict, findings_md: str, model: str) -> str:
    """
    Same OpenAI-compatible chat-completions call, pointed at a local
    server (llama.cpp `server`, Ollama's OpenAI-compat endpoint, etc.)
    via OPENAI_BASE_URL, e.g. http://localhost:8080/v1
    """
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "ローカルAIモードには OPENAI_BASE_URL (例: http://localhost:8080/v1) の設定が必要です。"
        )
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "openai パッケージが未インストールです。`pip install openai` を実行してください。"
        ) from e

    client = OpenAI(api_key="local", base_url=base_url)
    prompt = (
        "以下の分析サマリーと注目ポイントをもとに、日本語で簡潔なレポートを書いてください。\n\n"
        f"サマリー: {summary}\n\n検出済みポイント:\n{findings_md}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


def narrate(
    summary: dict,
    findings_md: str,
    mode: str = "none",
    model: str = "gpt-4o-mini",
) -> str:
    """
    Turn a compact analysis summary into prose.

    Parameters
    ----------
    summary : dict
        Small dict of already-computed numbers, e.g.
        {"period": "2026-08", "sales_change": -0.22, "customer_change": -0.19}
    findings_md : str
        Markdown produced by engine.insights.to_markdown(...)
    mode : "none" | "cloud" | "local"
    model : model name to use for "cloud"/"local" modes.
    """
    if mode == "cloud":
        try:
            return _cloud_narrative(summary, findings_md, model)
        except Exception as e:
            return f"(クラウドAI呼び出しに失敗したため、テンプレートで代替表示しています: {e})\n\n" + _template_narrative(
                summary, findings_md
            )
    if mode == "local":
        try:
            return _local_narrative(summary, findings_md, model)
        except Exception as e:
            return f"(ローカルAI呼び出しに失敗したため、テンプレートで代替表示しています: {e})\n\n" + _template_narrative(
                summary, findings_md
            )
    return _template_narrative(summary, findings_md)
