from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd


def _df_snippet(df: pd.DataFrame, *, max_rows: int = 12) -> str:
    if df is None or df.empty:
        return "(empty)"
    return df.head(max_rows).to_markdown(index=False)


def build_grounding_context(insights: Any) -> str:
    """
    Build a compact, data-backed context blob for the LLM.
    We keep this intentionally small-ish to control token usage while still being useful.
    """
    s = insights.summary

    parts: list[str] = []
    parts.append(
        f"Period: {s.get('date_min')} to {s.get('date_max')}. "
        f"{s.get('leads'):,} leads → {s.get('bound_leads'):,} bound ({s.get('bind_rate'):.1%}). "
        f"${s.get('total_spend_usd'):,.0f} spend → ${s.get('total_premium_annual_usd'):,.0f} premium. "
        f"EGP24: ${s.get('total_egp24_usd'):,.0f}. "
        f"Automation: {s.get('automation_rate', 0):.1%}, LTV:CAC: {s.get('cac_ltv_ratio', 0):.1f}:1, "
        f"Net commission: ${s.get('net_commission_revenue_usd', 0):,.0f}."
    )

    parts.append("\nFunnel table (with conversion lift):\n" + _df_snippet(insights.funnel, max_rows=10))
    parts.append("\nSTQ by vertical (carrier friction):\n" + _df_snippet(insights.stq_by_vertical, max_rows=12))
    parts.append("\nLead score intensity (intent vs risk):\n" + _df_snippet(insights.lead_score_intensity, max_rows=12))
    parts.append("\nVertical opportunity score (find+flood):\n" + _df_snippet(insights.vertical_opportunity_score, max_rows=12))
    parts.append("\nLead quality by channel (risk+ROI):\n" + _df_snippet(insights.lead_quality_by_channel, max_rows=15))
    parts.append("\nAI vs Human performance:\n" + _df_snippet(insights.ai_vs_human_perf, max_rows=10))
    parts.append("\nGolden window leakage:\n" + _df_snippet(insights.golden_window_leakage, max_rows=5))
    parts.append("\nCarrier hit ratio:\n" + _df_snippet(insights.carrier_hit_ratio, max_rows=15))
    parts.append("\nHuman leverage ratio:\n" + _df_snippet(insights.human_leverage_ratio.head(20), max_rows=20))
    parts.append("\nChurn risk analysis:\n" + _df_snippet(insights.churn_risk_analysis, max_rows=10))
    parts.append("\nPaid efficiency by channel:\n" + _df_snippet(insights.paid_efficiency_by_channel, max_rows=20))
    parts.append("\nSpeed-to-lead buckets:\n" + _df_snippet(insights.speed_to_lead, max_rows=12))
    parts.append("\nVertical performance:\n" + _df_snippet(insights.vertical_perf, max_rows=12))
    parts.append("\nAttribution (first-touch vs last-touch):\n" + _df_snippet(insights.attribution, max_rows=30))
    parts.append("\nDaily trends (last 14 rows):\n" + _df_snippet(insights.trends_daily.tail(14), max_rows=14))

    if getattr(insights, "recommendations", None):
        parts.append("\nPrecomputed recommendations:")
        for r in insights.recommendations[:8]:
            parts.append(f"- {r.get('title')}: {r.get('why')} | Action: {r.get('what_to_do')}")

    return "\n".join(parts)


def build_system_prompt(insights: Any) -> str:
    ctx = build_grounding_context(insights)
    return (
        "You are a senior GTM Data Scientist at Harper, an AI-native commercial insurance brokerage. "
        "You have real-time access to all operational metrics, funnel data, and performance tables.\n\n"
        "Communication style:\n"
        "- Write naturally, like you're having a conversation with a colleague\n"
        "- For simple questions or specific topics, respond conversationally—no bullets needed\n"
        "- Use bullets or numbered lists ONLY when breaking down multiple details, comparisons, or action items\n"
        "- Be concise and direct—get to the point quickly\n"
        "- Use specific numbers from the data tables naturally in your sentences\n"
        "- Skip fluff and disclaimers—just answer the question\n"
        "- If you don't have the data, say so briefly and suggest what would help\n\n"
        "When analyzing:\n"
        "- Lead with the insight in a natural sentence, then support with data\n"
        "- Compare metrics conversationally (e.g., 'Bind rate is sitting at 14%, which is below our 18% target')\n"
        "- Call out anomalies or opportunities naturally in the flow\n"
        "- For complex topics with multiple points, THEN use bullets—otherwise keep it conversational\n\n"
        "Examples:\n"
        "- Simple question: 'What's our bind rate?' → 'Bind rate is 14% across 102K leads. That's below our 18% target, so we're leaving money on the table.'\n"
        "- Complex question: 'How should we rebalance our paid channels?' → Use bullets to break down channel comparisons and recommendations\n\n"
        "Current data context:\n"
        f"{ctx}"
    )

