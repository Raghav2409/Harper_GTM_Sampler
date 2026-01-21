from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else float("nan")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.date


def add_derived_columns(leads: pd.DataFrame) -> pd.DataFrame:
    out = leads.copy()

    if "lead_created_at" in out.columns:
        out["lead_date"] = _to_date(out["lead_created_at"])

    # Some datasets include this already; keep if present.
    if "ttc_bucket" not in out.columns and "time_to_first_contact_min" in out.columns:
        ttc = out["time_to_first_contact_min"]
        bins = [-1, 5, 15, 60, 4 * 60, 24 * 60, 7 * 24 * 60]
        labels = ["<=5m", "5-15m", "15-60m", "1-4h", "4-24h", "1-7d"]
        out["ttc_bucket"] = pd.cut(ttc, bins=bins, labels=labels).astype(str)

    return out


@dataclass(frozen=True)
class InsightsPack:
    summary: Dict[str, Any]
    funnel: pd.DataFrame
    paid_efficiency_by_channel: pd.DataFrame
    paid_efficiency_by_campaign: pd.DataFrame
    speed_to_lead: pd.DataFrame
    vertical_perf: pd.DataFrame
    state_perf: pd.DataFrame
    agent_perf: pd.DataFrame
    trends_daily: pd.DataFrame
    attribution: pd.DataFrame
    recommendations: list[dict[str, Any]]
    # New operational metrics
    stq_by_vertical: pd.DataFrame
    lead_score_intensity: pd.DataFrame
    vertical_opportunity_score: pd.DataFrame
    lead_quality_by_channel: pd.DataFrame
    ai_vs_human_perf: pd.DataFrame
    golden_window_leakage: pd.DataFrame
    carrier_hit_ratio: pd.DataFrame
    human_leverage_ratio: pd.DataFrame
    churn_risk_analysis: pd.DataFrame


def compute_funnel(leads: pd.DataFrame) -> pd.DataFrame:
    n = len(leads)
    contacted = int(leads["is_contacted"].sum())
    quoted = int(leads["is_quoted"].sum())
    bound = int(leads["is_bound"].sum())
    premium = float(leads["bound_premium_annual_usd"].sum())
    egp24 = float(leads["expected_gross_profit_24m_usd"].sum())

    df = pd.DataFrame(
        [
            ("Leads", n, 1.0),
            ("Contacted", contacted, _safe_div(contacted, n)),
            ("Quoted", quoted, _safe_div(quoted, n)),
            ("Bound", bound, _safe_div(bound, n)),
        ],
        columns=["stage", "count", "rate_vs_leads"],
    )
    df["premium_annual_usd_total"] = [0, 0, 0, premium]
    df["egp_24m_total_usd"] = [0, 0, 0, egp24]
    
    # Add conversion lift (operational focus)
    df["conversion_lift"] = [0.0, df.loc[0, "rate_vs_leads"], 
                             _safe_div(df.loc[2, "rate_vs_leads"], df.loc[1, "rate_vs_leads"]) if df.loc[1, "rate_vs_leads"] > 0 else 0.0,
                             _safe_div(df.loc[3, "rate_vs_leads"], df.loc[2, "rate_vs_leads"]) if df.loc[2, "rate_vs_leads"] > 0 else 0.0]
    return df


def compute_paid_efficiency(
    leads: pd.DataFrame, ad_spend_daily: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paid_leads = leads[leads["source_campaign_id"].notna()].copy()
    spend_total = (
        ad_spend_daily.groupby(["channel"], as_index=False)
        .agg(spend_usd=("spend_usd", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum"))
        .copy()
    )
    leads_ch = (
        paid_leads.groupby(["source_channel"], as_index=False)
        .agg(
            leads=("lead_id", "count"),
            bound=("is_bound", "sum"),
            quoted=("is_quoted", "sum"),
            premium=("bound_premium_annual_usd", "sum"),
            egp24=("expected_gross_profit_24m_usd", "sum"),
        )
        .rename(columns={"source_channel": "channel"})
    )
    ch = spend_total.merge(leads_ch, on="channel", how="left").fillna(0)
    ch["cpl"] = ch.apply(lambda r: _safe_div(r["spend_usd"], r["leads"]), axis=1)
    ch["cpb"] = ch.apply(lambda r: _safe_div(r["spend_usd"], r["bound"]), axis=1)
    ch["cost_per_1k_premium"] = ch.apply(lambda r: _safe_div(r["spend_usd"], r["premium"] / 1000.0), axis=1)
    ch["roas_premium"] = ch.apply(lambda r: _safe_div(r["premium"], r["spend_usd"]), axis=1)
    ch["roi_egp24"] = ch.apply(lambda r: _safe_div(r["egp24"], r["spend_usd"]), axis=1)
    ch = ch.sort_values("spend_usd", ascending=False)

    spend_camp = (
        ad_spend_daily.groupby(["channel", "campaign_id"], as_index=False)
        .agg(spend_usd=("spend_usd", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum"))
        .copy()
    )
    leads_camp = (
        paid_leads.groupby(["source_channel", "source_campaign_id"], as_index=False)
        .agg(
            leads=("lead_id", "count"),
            bound=("is_bound", "sum"),
            quoted=("is_quoted", "sum"),
            premium=("bound_premium_annual_usd", "sum"),
            egp24=("expected_gross_profit_24m_usd", "sum"),
        )
        .rename(columns={"source_channel": "channel", "source_campaign_id": "campaign_id"})
    )
    camp = spend_camp.merge(leads_camp, on=["channel", "campaign_id"], how="left").fillna(0)
    camp["cpl"] = camp.apply(lambda r: _safe_div(r["spend_usd"], r["leads"]), axis=1)
    camp["cpb"] = camp.apply(lambda r: _safe_div(r["spend_usd"], r["bound"]), axis=1)
    camp["cost_per_1k_premium"] = camp.apply(lambda r: _safe_div(r["spend_usd"], r["premium"] / 1000.0), axis=1)
    camp["roi_egp24"] = camp.apply(lambda r: _safe_div(r["egp24"], r["spend_usd"]), axis=1)
    camp = camp.sort_values("spend_usd", ascending=False)

    return ch, camp


def compute_speed_to_lead(leads: pd.DataFrame) -> pd.DataFrame:
    df = leads.copy()
    df = df[df["source_channel"].notna()]

    g = (
        df.groupby("ttc_bucket", as_index=False)
        .agg(
            leads=("lead_id", "count"),
            contacted_rate=("is_contacted", "mean"),
            quoted_rate=("is_quoted", "mean"),
            bound_rate=("is_bound", "mean"),
            avg_p_bind=("p_bind_synthetic", "mean"),
            avg_egp24=("expected_gross_profit_24m_usd", "mean"),
        )
        .sort_values("ttc_bucket")
    )
    return g


def compute_dim_perf(leads: pd.DataFrame, dim: str, min_n: int = 200) -> pd.DataFrame:
    df = leads.copy()
    g = (
        df.groupby(dim, as_index=False)
        .agg(
            leads=("lead_id", "count"),
            bound_rate=("is_bound", "mean"),
            quote_rate=("is_quoted", "mean"),
            avg_ttc_min=("time_to_first_contact_min", "mean"),
            premium_per_lead=("bound_premium_annual_usd", "mean"),
            egp24_per_lead=("expected_gross_profit_24m_usd", "mean"),
        )
        .sort_values("leads", ascending=False)
    )
    g = g[g["leads"] >= min_n]
    return g


def compute_agent_perf(leads: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    g = (
        leads.groupby("assigned_agent_id", as_index=False)
        .agg(
            leads=("lead_id", "count"),
            contacted_rate=("is_contacted", "mean"),
            quote_rate=("is_quoted", "mean"),
            bound_rate=("is_bound", "mean"),
            avg_ttc_min=("time_to_first_contact_min", "mean"),
            egp24_total=("expected_gross_profit_24m_usd", "sum"),
        )
        .rename(columns={"assigned_agent_id": "agent_id"})
        .merge(agents, on="agent_id", how="left")
    )
    g["egp24_per_lead"] = g.apply(lambda r: _safe_div(r["egp24_total"], r["leads"]), axis=1)
    return g.sort_values("egp24_total", ascending=False)


def compute_trends_daily(leads: pd.DataFrame, ad_spend_daily: pd.DataFrame) -> pd.DataFrame:
    lf = (
        leads.groupby("lead_date", as_index=False)
        .agg(
            leads=("lead_id", "count"),
            contacted=("is_contacted", "sum"),
            quoted=("is_quoted", "sum"),
            bound=("is_bound", "sum"),
            premium=("bound_premium_annual_usd", "sum"),
            egp24=("expected_gross_profit_24m_usd", "sum"),
        )
        .rename(columns={"lead_date": "date"})
    )
    lf["date"] = pd.to_datetime(lf["date"])

    spend = ad_spend_daily.groupby("date", as_index=False).agg(spend_usd=("spend_usd", "sum"))
    out = lf.merge(spend, on="date", how="left").fillna({"spend_usd": 0.0})
    out["cpl"] = out.apply(lambda r: _safe_div(r["spend_usd"], r["leads"]), axis=1)
    out["cpb"] = out.apply(lambda r: _safe_div(r["spend_usd"], r["bound"]), axis=1)
    out["bind_rate"] = out.apply(lambda r: _safe_div(r["bound"], r["leads"]), axis=1)
    return out.sort_values("date")


def compute_attribution(leads: pd.DataFrame) -> pd.DataFrame:
    # Simple view: origin (TOF) vs last-touch (source_channel). This is intentionally lightweight.
    df = leads.copy()
    a = (
        df.groupby(["origin_channel"], as_index=False)
        .agg(leads=("lead_id", "count"), bound=("is_bound", "sum"), premium=("bound_premium_annual_usd", "sum"))
        .rename(columns={"origin_channel": "channel"})
    )
    a["view"] = "First-touch (origin_channel)"

    b = (
        df.groupby(["source_channel"], as_index=False)
        .agg(leads=("lead_id", "count"), bound=("is_bound", "sum"), premium=("bound_premium_annual_usd", "sum"))
        .rename(columns={"source_channel": "channel"})
    )
    b["view"] = "Last-touch (source_channel)"

    out = pd.concat([a, b], ignore_index=True)
    out["bind_rate"] = out.apply(lambda r: _safe_div(r["bound"], r["leads"]), axis=1)
    return out.sort_values(["view", "leads"], ascending=[True, False])


def compute_stq_by_vertical(leads: pd.DataFrame) -> pd.DataFrame:
    """Submission-to-Quote ratio by vertical - identifies carrier friction points"""
    df = leads[leads["is_quoted"] == 1].copy()
    if df.empty or "stq_ratio" not in df.columns:
        return pd.DataFrame(columns=["vertical", "quoted", "avg_stq_ratio", "stq_count"])
    
    g = df.groupby("vertical", as_index=False).agg(
        quoted=("lead_id", "count"),
        avg_stq_ratio=("stq_ratio", "mean"),
        stq_count=("stq_ratio", lambda x: (x > 0.5).sum()),  # Count where STQ > 50%
    )
    g["stq_success_rate"] = g.apply(lambda r: _safe_div(r["stq_count"], r["quoted"]), axis=1)
    return g.sort_values("avg_stq_ratio", ascending=False)


def compute_lead_score_intensity(leads: pd.DataFrame) -> pd.DataFrame:
    """Lead score intensity map: intent_score + risk_score distribution by vertical"""
    df = leads.copy()
    g = df.groupby("vertical", as_index=False).agg(
        leads=("lead_id", "count"),
        avg_intent_score=("intent_score", "mean"),
        avg_risk_score=("risk_score", "mean"),
        high_intent_pct=("intent_score", lambda x: (x > 0.7).mean()),
        low_risk_pct=("risk_score", lambda x: (x < 0.4).mean()),
    )
    g["opportunity_score"] = g["avg_intent_score"] * (1 - g["avg_risk_score"])
    return g.sort_values("opportunity_score", ascending=False)


def compute_vertical_opportunity_score(leads: pd.DataFrame) -> pd.DataFrame:
    """Vertical Opportunity Score: vertical + intent_score + avg_premium"""
    df = leads.copy()
    g = df.groupby("vertical", as_index=False).agg(
        leads=("lead_id", "count"),
        avg_intent_score=("intent_score", "mean"),
        avg_premium=("bound_premium_annual_usd", "mean"),
        bound_rate=("is_bound", "mean"),
        avg_egp24=("expected_gross_profit_24m_usd", "mean"),
    )
    # Opportunity score = intent * premium * bound_rate
    g["opportunity_score"] = g["avg_intent_score"] * (g["avg_premium"] / 1000.0) * g["bound_rate"]
    return g.sort_values("opportunity_score", ascending=False)


def compute_lead_quality_by_channel(leads: pd.DataFrame) -> pd.DataFrame:
    """Lead quality by channel: ROI + risk_score analysis"""
    paid_leads = leads[leads["source_campaign_id"].notna()].copy()
    if paid_leads.empty:
        return pd.DataFrame(columns=["source_channel", "leads", "avg_risk_score", "high_risk_pct", "bound_rate", "avg_egp24", "quality_score"])
    
    g = paid_leads.groupby("source_channel", as_index=False).agg(
        leads=("lead_id", "count"),
        avg_risk_score=("risk_score", "mean"),
        high_risk_pct=("risk_score", lambda x: (x > 0.6).mean() if len(x) > 0 else 0.0),
        bound_rate=("is_bound", "mean"),
        avg_egp24=("expected_gross_profit_24m_usd", "mean"),
        total_egp24=("expected_gross_profit_24m_usd", "sum"),
        total_cac=("cac_usd", "sum") if "cac_usd" in paid_leads.columns else pd.Series([0.0] * len(paid_leads.groupby("source_channel"))),
    )
    g["roi_egp24"] = g.apply(lambda r: _safe_div(r["total_egp24"], r["total_cac"]), axis=1)
    g["quality_score"] = g["bound_rate"] * (1 - g["avg_risk_score"])
    return g.sort_values("quality_score", ascending=False)


def compute_ai_vs_human_perf(leads: pd.DataFrame, touchpoints: pd.DataFrame) -> pd.DataFrame:
    """AI Voice vs Human Performance comparison"""
    # Merge touchpoints with leads to get AI contact type
    if "ai_contact_type" not in leads.columns:
        return pd.DataFrame(columns=["contact_type", "leads", "contacted_rate", "quoted_rate", "bound_rate"])
    
    df = leads.copy()
    g = df.groupby("ai_contact_type", as_index=False).agg(
        leads=("lead_id", "count"),
        contacted_rate=("is_contacted", "mean"),
        quoted_rate=("is_quoted", "mean"),
        bound_rate=("is_bound", "mean"),
        avg_egp24=("expected_gross_profit_24m_usd", "mean"),
    )
    return g.sort_values("bound_rate", ascending=False)


def compute_golden_window_leakage(leads: pd.DataFrame) -> pd.DataFrame:
    """Calculate revenue lost from leads contacted after 5 minutes"""
    df = leads.copy()
    golden_window = df[df["time_to_first_contact_min"] <= 5]
    leaked = df[df["time_to_first_contact_min"] > 5]
    
    golden_bound_rate = golden_window["is_bound"].mean() if len(golden_window) > 0 else 0.0
    leaked_bound_rate = leaked["is_bound"].mean() if len(leaked) > 0 else 0.0
    
    golden_egp24 = golden_window["expected_gross_profit_24m_usd"].sum()
    leaked_egp24 = leaked["expected_gross_profit_24m_usd"].sum()
    
    # Potential EGP24 if leaked leads had golden window treatment
    potential_egp24 = leaked["expected_gross_profit_24m_usd"].sum() * (golden_bound_rate / max(leaked_bound_rate, 0.01))
    leakage_usd = potential_egp24 - leaked_egp24
    
    return pd.DataFrame([{
        "metric": "Golden Window Leakage",
        "leads_in_golden_window": len(golden_window),
        "leads_leaked": len(leaked),
        "golden_bound_rate": golden_bound_rate,
        "leaked_bound_rate": leaked_bound_rate,
        "leakage_egp24_usd": leakage_usd,
        "potential_egp24_if_fixed": potential_egp24,
    }])


def compute_carrier_hit_ratio(policies: pd.DataFrame, leads: pd.DataFrame = None) -> pd.DataFrame:
    """Carrier hit ratio: which carrier_tier binds most frequently for which vertical"""
    if policies is None or policies.empty or "carrier_hit_ratio" not in policies.columns:
        return pd.DataFrame(columns=["carrier_tier", "policy_type", "policies", "avg_hit_ratio", "total_premium"])
    
    # Group by carrier_tier and policy_type (can merge with leads later if needed for vertical)
    g = policies.groupby(["carrier_tier", "policy_type"], as_index=False).agg(
        policies=("policy_id", "count"),
        avg_hit_ratio=("carrier_hit_ratio", "mean"),
        total_premium=("premium_annual_usd", "sum"),
    )
    return g.sort_values("avg_hit_ratio", ascending=False)


def compute_human_leverage_ratio(leads: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    """Human Leverage Ratio: premium managed per agent with AI support"""
    agent_perf = compute_agent_perf(leads, agents)
    if agent_perf.empty:
        return pd.DataFrame(columns=["agent_id", "premium_per_agent", "leads_per_agent", "leverage_ratio"])
    
    agent_perf["premium_per_agent"] = agent_perf.apply(
        lambda r: _safe_div(r["egp24_total"] / 0.12 if r.get("egp24_total", 0) > 0 else 0, 1), axis=1
    )
    agent_perf["leverage_ratio"] = agent_perf["premium_per_agent"] / 1000.0  # Normalize
    return agent_perf[["agent_id", "premium_per_agent", "leads", "leverage_ratio"]].sort_values("leverage_ratio", ascending=False)


def compute_churn_risk_analysis(leads: pd.DataFrame) -> pd.DataFrame:
    """Churn Risk Score analysis for predictive insights"""
    if "churn_risk_score" not in leads.columns:
        return pd.DataFrame(columns=["churn_risk_bucket", "customers", "avg_churn_risk", "total_premium_at_risk"])
    
    df = leads[leads["is_bound"] == 1].copy()
    df["churn_bucket"] = pd.cut(df["churn_risk_score"], bins=[0, 0.3, 0.5, 0.7, 1.0], labels=["Low", "Medium", "High", "Critical"])
    
    g = df.groupby("churn_bucket", observed=True, as_index=False).agg(
        customers=("lead_id", "count"),
        avg_churn_risk=("churn_risk_score", "mean"),
        total_premium_at_risk=("bound_premium_annual_usd", "sum"),
        avg_premium=("bound_premium_annual_usd", "mean"),
    )
    return g.sort_values("avg_churn_risk", ascending=False)


def build_recommendations(
    funnel: pd.DataFrame,
    paid_by_channel: pd.DataFrame,
    speed_to_lead: pd.DataFrame,
    vertical_perf: pd.DataFrame,
) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []

    # Speed-to-lead recommendation
    if not speed_to_lead.empty and "bound_rate" in speed_to_lead.columns:
        best = speed_to_lead.sort_values("bound_rate", ascending=False).head(1).iloc[0]
        worst = speed_to_lead.sort_values("bound_rate", ascending=True).head(1).iloc[0]
        recs.append(
            {
                "title": "Speed-to-lead is a lever",
                "why": f"Best bucket: {best['ttc_bucket']} (bind {best['bound_rate']:.1%}) vs worst: {worst['ttc_bucket']} (bind {worst['bound_rate']:.1%}).",
                "what_to_do": "Route high-intent/high-value leads to a fast-followup queue; enforce SLAs; staff evenings/weekends if needed.",
            }
        )

    # Paid channel efficiency recommendation
    if not paid_by_channel.empty and "roi_egp24" in paid_by_channel.columns:
        top = paid_by_channel.sort_values("roi_egp24", ascending=False).head(1).iloc[0]
        bot = paid_by_channel.sort_values("roi_egp24", ascending=True).head(1).iloc[0]
        recs.append(
            {
                "title": "Rebalance paid budget by value (EGP24 / spend)",
                "why": f"Top ROI: {top['channel']} (ROI {top['roi_egp24']:.2f}); lowest: {bot['channel']} (ROI {bot['roi_egp24']:.2f}).",
                "what_to_do": "Shift budget toward higher-ROI channels/campaigns and bid to expected gross profit, not just CPL.",
            }
        )

    # Vertical targeting recommendation
    if not vertical_perf.empty and "egp24_per_lead" in vertical_perf.columns:
        topv = vertical_perf.sort_values("egp24_per_lead", ascending=False).head(3)
        recs.append(
            {
                "title": "Lean into high-value verticals",
                "why": "Top 3 by expected gross profit per lead: "
                + ", ".join([f"{r['vertical']} (${r['egp24_per_lead']:.0f}/lead)" for _, r in topv.iterrows()]),
                "what_to_do": "Create verticalized landing pages + ad groups; tune routing/underwriting capacity for these segments.",
            }
        )

    # Funnel diagnostic recommendation
    if len(funnel) >= 4:
        contacted_rate = float(funnel.loc[funnel["stage"] == "Contacted", "rate_vs_leads"].iloc[0])
        quote_rate = float(funnel.loc[funnel["stage"] == "Quoted", "rate_vs_leads"].iloc[0])
        bind_rate = float(funnel.loc[funnel["stage"] == "Bound", "rate_vs_leads"].iloc[0])
        recs.append(
            {
                "title": "Funnel diagnosis",
                "why": f"Contact {contacted_rate:.1%} → Quote {quote_rate:.1%} → Bind {bind_rate:.1%} (vs leads).",
                "what_to_do": "If contact is low: fix speed/coverage; if quote is low: tighten intake & prequal; if bind is low: improve quote competitiveness + follow-up.",
            }
        )

    return recs


def compute_insights_pack(
    leads: pd.DataFrame, ad_spend_daily: pd.DataFrame, agents: pd.DataFrame, 
    touchpoints: pd.DataFrame = None, policies: pd.DataFrame = None
) -> InsightsPack:
    leads2 = add_derived_columns(leads)

    funnel = compute_funnel(leads2)
    paid_by_channel, paid_by_campaign = compute_paid_efficiency(leads2, ad_spend_daily)
    speed = compute_speed_to_lead(leads2)
    vertical_perf = compute_dim_perf(leads2, "vertical", min_n=200)
    state_perf = compute_dim_perf(leads2, "state", min_n=200)
    agent_perf = compute_agent_perf(leads2, agents)
    trends = compute_trends_daily(leads2, ad_spend_daily)
    attribution = compute_attribution(leads2)

    # New operational metrics
    stq_by_vertical = compute_stq_by_vertical(leads2)
    lead_score_intensity = compute_lead_score_intensity(leads2)
    vertical_opportunity_score = compute_vertical_opportunity_score(leads2)
    lead_quality_by_channel = compute_lead_quality_by_channel(leads2)
    ai_vs_human_perf = compute_ai_vs_human_perf(leads2, touchpoints) if touchpoints is not None else pd.DataFrame()
    golden_window_leakage = compute_golden_window_leakage(leads2)
    carrier_hit_ratio = compute_carrier_hit_ratio(policies, leads2) if policies is not None else pd.DataFrame()
    human_leverage_ratio = compute_human_leverage_ratio(leads2, agents)
    churn_risk_analysis = compute_churn_risk_analysis(leads2)

    # Calculate operational summary metrics
    automation_rate = float(leads2["automation_rate"].mean()) if "automation_rate" in leads2.columns else 0.0
    total_cac = float(leads2["cac_usd"].sum()) if "cac_usd" in leads2.columns else 0.0
    total_ltv = float(leads2["ltv_usd"].sum()) if "ltv_usd" in leads2.columns else 0.0
    cac_ltv_ratio = _safe_div(total_ltv, total_cac) if total_cac > 0 else 0.0
    net_commission_revenue = float(leads2["net_commission_revenue_usd"].sum()) if "net_commission_revenue_usd" in leads2.columns else 0.0

    summary = {
        "leads": int(len(leads2)),
        "bound_leads": int(leads2["is_bound"].sum()),
        "bind_rate": float(leads2["is_bound"].mean()),
        "total_premium_annual_usd": float(leads2["bound_premium_annual_usd"].sum()),
        "total_egp24_usd": float(leads2["expected_gross_profit_24m_usd"].sum()),
        "total_spend_usd": float(ad_spend_daily["spend_usd"].sum()),
        "date_min": str(pd.to_datetime(leads2["lead_created_at"]).min().date()),
        "date_max": str(pd.to_datetime(leads2["lead_created_at"]).max().date()),
        # New operational metrics
        "automation_rate": automation_rate,
        "cac_ltv_ratio": cac_ltv_ratio,
        "net_commission_revenue_usd": net_commission_revenue,
    }

    recs = build_recommendations(funnel, paid_by_channel, speed, vertical_perf)

    return InsightsPack(
        summary=summary,
        funnel=funnel,
        paid_efficiency_by_channel=paid_by_channel,
        paid_efficiency_by_campaign=paid_by_campaign,
        speed_to_lead=speed,
        vertical_perf=vertical_perf,
        state_perf=state_perf,
        agent_perf=agent_perf,
        trends_daily=trends,
        attribution=attribution,
        recommendations=recs,
        stq_by_vertical=stq_by_vertical,
        lead_score_intensity=lead_score_intensity,
        vertical_opportunity_score=vertical_opportunity_score,
        lead_quality_by_channel=lead_quality_by_channel,
        ai_vs_human_perf=ai_vs_human_perf,
        golden_window_leakage=golden_window_leakage,
        carrier_hit_ratio=carrier_hit_ratio,
        human_leverage_ratio=human_leverage_ratio,
        churn_risk_analysis=churn_risk_analysis,
    )

