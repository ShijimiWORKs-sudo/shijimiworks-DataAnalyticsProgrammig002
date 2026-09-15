"""
Shijimi AI Data Engine
======================

A small, reusable analysis core shared by every app in this repository
(Sales Analytics, Customer Analytics, Hotel Analytics, Darts Analytics,
AI Data Analyst, ...).

The design principle (see docs/phase_plan.md):

    judgement (statistics / ML)  -> built ourselves, in this package
    narration (human-readable report) -> optionally handed to an LLM

Nothing in this package requires an API key. `engine.llm.narrate()` only
calls out to a language model if you explicitly ask for it and provide a
key; otherwise it falls back to a template-based Japanese narrative.
"""

from . import stats, insights, llm, rfm, forecast  # noqa: F401

__all__ = ["stats", "insights", "llm", "rfm", "forecast"]
