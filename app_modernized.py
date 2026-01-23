from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

from gtm.copilot import build_system_prompt
from gtm.data import load_gtm_data
from gtm.metrics import compute_insights_pack

# ============================================================================
# CONFIG & STYLING
# ============================================================================

APP_TITLE = "Harper GTM Co‑Pilot"
APP_SUBTITLE = "Real-time optimization dashboard for growth & revenue acceleration"

# Custom CSS for modern, refined aesthetics
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Lora', serif;
        letter-spacing: -0.5px;
    }
    
    /* Root theme variables */
    :root {
        --primary: #0f1a2e;      /* Deep navy */
        --secondary: #1d3a5c;    /* Dark blue */
        --accent: #10b981;       /* Emerald */
        --accent-light: #d1fae5; /* Light mint */
        --neutral-50: #f9fafb;
        --neutral-100: #f3f4f6;
        --neutral-200: #e5e7eb;
        --neutral-300: #d1d5db;
        --neutral-600: #4b5563;
        --neutral-900: #111827;
        --danger: #ef4444;
        --warning: #f59e0b;
        --success: #10b981;
    }
    
    /* Main page background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1a2e 0%, #1d3a5c 100%);
    }
    
    /* Headers */
    h1 {
        color: var(--primary);
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -1px;
    }
    
    h2 {
        color: var(--primary);
        font-size: 1.6rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--accent);
        padding-bottom: 0.75rem;
        display: inline-block;
    }
    
    h3 {
        color: var(--secondary);
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Subheader styling */
    .stSubheader {
        margin-top: 1.5rem !important;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 1.25rem !important;
        border-left: 4px solid var(--accent);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    
    [role="grid"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Tabs */
    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.5rem;
        border-bottom: 2px solid var(--neutral-200);
        padding-bottom: 0;
    }
    
    [data-testid="stTabs"] [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 3px solid var(--accent) !important;
    }
    
    [data-testid="stTabs"] [aria-selected="false"] {
        color: var(--neutral-600) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stDateInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div > div {
        border-radius: 8px !important;
        border: 1px solid var(--neutral-200) !important;
        background-color: white !important;
        padding: 0.75rem !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] {
        padding: 1.5rem;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSubheader {
        color: white !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] .stCaption {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Expander styling */
    [data-testid="stExpander"] {
        background: white;
        border: 1px solid var(--neutral-200);
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }
    
    [data-testid="stExpander"] summary {
        color: var(--primary);
        font-weight: 600;
    }
    
    /* Caption styling */
    .stCaption {
        color: var(--neutral-600);
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    
    /* Info boxes */
    [data-testid="stInfo"], 
    [data-testid="stSuccess"], 
    [data-testid="stWarning"], 
    [data-testid="stError"] {
        border-radius: 8px;
        padding: 1rem !important;
        border-left: 4px solid;
    }
    
    [data-testid="stInfo"] {
        border-left-color: #3b82f6;
        background-color: rgba(59, 130, 246, 0.05);
    }
    
    [data-testid="stWarning"] {
        border-left-color: var(--warning);
        background-color: rgba(245, 158, 11, 0.05);
    }
    
    [data-testid="stError"] {
        border-left-color: var(--danger);
        background-color: rgba(239, 68, 68, 0.05);
    }
    
    /* Plotly charts */
    .plotly-graph-div {
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        background: white;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: var(--accent) !important;
        border-right-color: transparent !important;
    }
    
    /* Columns layout padding */
    .stColumn {
        padding: 0.5rem;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--neutral-100);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--neutral-300);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--neutral-400);
    }
    
    /* Fade-in animation for content */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    [data-testid="stMetricContainer"], 
    [data-testid="stDataFrame"], 
    .plotly-graph-div {
        animation: fadeIn 0.4s ease-out;
    }
</style>
"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False)
def _load_all(root_dir: str, cache_key: str = "default") -> dict:
    """Load raw data - cached by cache_key (use file mtime to invalidate)"""
    d = load_gtm_data(root_dir)
    return {"data": d}


def _get_data_cache_key(root_dir: str) -> str:
    """Generate cache key based on file modification times to invalidate when data changes"""
    import os
    from pathlib import Path
    root = Path(root_dir)
    key_files = ["leads.csv", "touchpoints.csv", "policies.csv", "ad_spend_daily.csv", "agents.csv"]
    mtimes = []
    for f in key_files:
        path = root / f
        if path.exists():
            mtimes.append(str(os.path.getmtime(path)))
    return "_".join(mtimes)


@st.cache_data(show_spinner="Computing insights...")
def _compute_filtered_insights(
    leads: pd.DataFrame,
    ad_spend_daily: pd.DataFrame,
    agents: pd.DataFrame,
    touchpoints: pd.DataFrame,
    policies: pd.DataFrame,
    filter_hash: str,
) -> Any:
    """Compute insights from filtered data. Cached by filter_hash."""
    return compute_insights_pack(leads, ad_spend_daily, agents, touchpoints, policies)


def _money(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"${x:,.0f}"


def _pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.1%}"


def _get_openai_client(api_key: Optional[str]) -> Optional[OpenAI]:
    key = api_key or os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not key:
        return None
    key = key.strip()
    if not key.startswith("sk-"):
        st.error(f"Invalid API key format. Key should start with 'sk-'. Got: {key[:10]}...")
        return None
    try:
        return OpenAI(api_key=key)
    except Exception as e:
        st.error(f"Failed to create OpenAI client: {str(e)}")
        return None


# ============================================================================
# UI COMPONENTS
# ============================================================================

def _render_header(insights) -> None:
    """Render enhanced header with KPI metrics"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Title section
    col_title, col_nav = st.columns([3, 1])
    with col_title:
        st.title(APP_TITLE)
        st.caption(f"📊 {APP_SUBTITLE}")
    
    # KPI metrics grid
    s = insights.summary
    st.divider()
    
    # Create responsive metric grid
    metric_cols = st.columns(7)
    
    metrics = [
        ("📈 Leads", f"{s['leads']:,}", None),
        ("🎯 Bound", f"{s['bound_leads']:,}", _pct(s["bind_rate"])),
        ("⚙️ Automation", _pct(s.get("automation_rate", 0.0)), None),
        ("💰 CAC:LTV", f"{s.get('cac_ltv_ratio', 0.0):.1f}:1", "✅" if s.get('cac_ltv_ratio', 0) >= 3.0 else "⚠️"),
        ("💵 Net Revenue", _money(s.get("net_commission_revenue_usd", 0)), None),
        ("💸 Spend", _money(s["total_spend_usd"]), None),
        ("🏆 Premium (annual)", _money(s["total_premium_annual_usd"]), None),
    ]
    
    for col, (label, value, delta) in zip(metric_cols, metrics):
        with col:
            col.metric(label, value, delta)
    
    st.divider()
    st.caption(f"📅 **Window:** {s['date_min']} → {s['date_max']} · *Fully synthetic data*")


def _render_filters(leads: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Render sidebar filters with enhanced UX"""
    st.sidebar.markdown("### 🎚️ Filters")
    st.sidebar.divider()
    
    # Date filter
    min_date = pd.to_datetime(leads["lead_created_at"]).min().date()
    max_date = pd.to_datetime(leads["lead_created_at"]).max().date()
    
    with st.sidebar.expander("📅 Date Range", expanded=True):
        date_range = st.date_input("Lead created date", value=(min_date, max_date))
        if isinstance(date_range, tuple) and len(date_range) == 2:
            d0 = pd.to_datetime(date_range[0])
            d1 = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        else:
            d0 = pd.to_datetime(min_date)
            d1 = pd.to_datetime(max_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    # Channel filter
    channels = sorted(leads["source_channel"].dropna().unique().tolist())
    with st.sidebar.expander("📱 Channel", expanded=True):
        selected_channels = st.multiselect("Source channel", options=channels, default=channels)
    
    # Vertical filter
    verticals = sorted(leads["vertical"].dropna().unique().tolist())
    with st.sidebar.expander("🏢 Vertical", expanded=False):
        selected_verticals = st.multiselect("Vertical", options=verticals, default=verticals)
    
    # State filter
    states = sorted(leads["state"].dropna().unique().tolist())
    with st.sidebar.expander("🗺️ State", expanded=False):
        selected_states = st.multiselect("State", options=states, default=states)
    
    # E&S filter
    with st.sidebar.expander("📋 E&S Eligibility", expanded=False):
        es = st.multiselect("E&S eligible", options=["All", "Yes", "No"], default=["All"])
    
    # Apply filters
    out = leads.copy()
    out = out[(out["lead_created_at"] >= d0) & (out["lead_created_at"] <= d1)]
    out = out[out["source_channel"].isin(selected_channels)]
    out = out[out["vertical"].isin(selected_verticals)]
    out = out[out["state"].isin(selected_states)]
    
    if "All" not in es:
        if "Yes" in es and "No" not in es:
            out = out[out["is_es_eligible"] == 1]
        if "No" in es and "Yes" not in es:
            out = out[out["is_es_eligible"] == 0]
    
    # Create filter hash
    filter_str = f"{d0}_{d1}_{sorted(selected_channels)}_{sorted(selected_verticals)}_{sorted(selected_states)}_{sorted(es)}"
    filter_hash = hashlib.md5(filter_str.encode()).hexdigest()
    
    return out, filter_hash


def _render_dashboard_overview(insights, f_leads) -> None:
    """Overview tab - key metrics and recommendations"""
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 📊 Conversion Funnel")
        st.caption("Track lead progression with conversion lift insights")
        funnel = insights.funnel.copy()
        st.dataframe(funnel, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 💡 Quick Wins")
        if insights.recommendations:
            for i, r in enumerate(insights.recommendations[:3]):
                with st.expander(f"#{i+1} • {r.get('title', 'Recommendation')}", expanded=(i==0)):
                    st.markdown(f"**Why:** {r.get('why')}")
                    st.markdown(f"**What to do:** {r.get('what_to_do')}")
        else:
            st.info("No recommendations available")
    
    st.divider()
    
    # STQ by Vertical
    st.markdown("### ⏱️ Submission-to-Quote (STQ) Ratio by Vertical")
    st.caption("Lower = faster carrier response | Identify bottlenecks")
    
    if not insights.stq_by_vertical.empty:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(insights.stq_by_vertical, use_container_width=True, hide_index=True)
        with col2:
            fig_stq = px.bar(
                insights.stq_by_vertical.head(10),
                x="vertical",
                y="avg_stq_ratio",
                title="STQ Ratio by Vertical",
                color="avg_stq_ratio",
                color_continuous_scale="RdYlGn_r"
            )
            fig_stq.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_stq, use_container_width=True)
    else:
        st.info("STQ data not available")
    
    st.divider()
    
    # Lead Score Intensity
    st.markdown("### 🎯 Lead Score Intensity Map")
    st.caption("Intent + Risk score distribution | Find high-value opportunities")
    
    if not insights.lead_score_intensity.empty:
        st.dataframe(insights.lead_score_intensity, use_container_width=True, hide_index=True)
        fig_intensity = px.scatter(
            insights.lead_score_intensity,
            x="avg_intent_score",
            y="avg_risk_score",
            size="leads",
            color="opportunity_score",
            hover_data=["vertical"],
            title="Intent vs Risk Score",
            color_continuous_scale="Viridis"
        )
        fig_intensity.update_layout(height=500)
        st.plotly_chart(fig_intensity, use_container_width=True)
    else:
        st.info("Lead score intensity data not available")


def _render_dashboard_efficiency(insights) -> None:
    """Paid efficiency tab - channel & campaign analysis"""
    st.markdown("### 📈 Channel Performance")
    st.caption("ROI + efficiency metrics by channel | Optimize budget allocation")
    
    ch = insights.paid_efficiency_by_channel.copy()
    st.dataframe(ch, use_container_width=True, hide_index=True)
    
    fig = px.scatter(
        ch,
        x="cpl",
        y="roi_egp24",
        size="spend_usd",
        color="channel",
        hover_data=["cpb", "cost_per_1k_premium", "leads", "bound", "premium"],
        title="Channel Tradeoffs: CPL vs ROI (EGP24/spend)",
        size_max=50
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Vertical Opportunity
    st.markdown("### 🎯 Vertical Opportunity Score")
    st.caption("Find + Flood micro-segments for maximum ROI")
    
    if not insights.vertical_opportunity_score.empty:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(insights.vertical_opportunity_score, use_container_width=True, hide_index=True)
        with col2:
            fig_opp = px.bar(
                insights.vertical_opportunity_score.head(12),
                x="vertical",
                y="opportunity_score",
                color="opportunity_score",
                color_continuous_scale="Greens",
                title="Top Opportunities"
            )
            fig_opp.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_opp, use_container_width=True)
    
    st.divider()
    
    # Lead Quality by Channel
    st.markdown("### ✨ Lead Quality by Channel")
    st.caption("ROI + Risk Score analysis | Prevent funnel clogging")
    
    if not insights.lead_quality_by_channel.empty:
        st.dataframe(insights.lead_quality_by_channel, use_container_width=True, hide_index=True)
        fig_quality = px.scatter(
            insights.lead_quality_by_channel,
            x="avg_risk_score",
            y="roi_egp24",
            size="leads",
            color="quality_score",
            hover_data=["source_channel"],
            title="Lead Quality: Risk vs ROI",
            color_continuous_scale="RdYlGn"
        )
        fig_quality.update_layout(height=500)
        st.plotly_chart(fig_quality, use_container_width=True)
    
    st.divider()
    
    # Top Campaigns
    st.markdown("### 🚀 Top Campaigns by Spend")
    camp = insights.paid_efficiency_by_campaign.copy().head(25)
    st.dataframe(camp, use_container_width=True, hide_index=True)


def _render_dashboard_speed(insights, f_leads) -> None:
    """Speed-to-lead tab - operations & golden window"""
    
    # AI vs Human
    st.markdown("### 🤖 AI Voice vs Human Performance")
    st.caption("Validates ROI of AI infrastructure")
    
    if not insights.ai_vs_human_perf.empty:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(insights.ai_vs_human_perf, use_container_width=True, hide_index=True)
        with col2:
            fig_ai = px.bar(
                insights.ai_vs_human_perf,
                x="ai_contact_type",
                y="bound_rate",
                color="contacted_rate",
                color_continuous_scale="Blues",
                title="AI vs Human: Bind Rate"
            )
            fig_ai.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_ai, use_container_width=True)
    else:
        st.info("AI vs Human data not available")
    
    st.divider()
    
    # Golden Window
    st.markdown("### ⏳ Golden Window Leakage")
    st.caption("Revenue lost from leads contacted after 5 minutes | **Burning platform for engineering**")
    
    if not insights.golden_window_leakage.empty:
        leakage = insights.golden_window_leakage.iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            col1.metric("Leakage (EGP24)", _money(leakage.get("leakage_egp24_usd", 0)))
        with col2:
            col2.metric("Potential if Fixed", _money(leakage.get("potential_egp24_if_fixed", 0)))
        with col3:
            col3.metric("Impact %", f"{(leakage.get('leakage_egp24_usd', 0) / (leakage.get('potential_egp24_if_fixed', 1) or 1) * 100):.1f}%")
        
        st.dataframe(insights.golden_window_leakage, use_container_width=True, hide_index=True)
    else:
        st.info("Golden window leakage data not available")
    
    st.divider()
    
    # Speed-to-lead lift
    st.markdown("### 📊 Speed-to-Lead Conversion Lift")
    speed = insights.speed_to_lead.copy()
    st.dataframe(speed, use_container_width=True, hide_index=True)
    
    fig = px.bar(
        speed,
        x="ttc_bucket",
        y="bound_rate",
        color="bound_rate",
        color_continuous_scale="RdYlGn",
        title="Bind Rate by Time-to-First-Contact"
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Operational drilldown
    st.markdown("### 🔧 Operational Drilldown")
    t = (
        f_leads.groupby("ttc_bucket", as_index=False)
        .agg(leads=("lead_id", "count"), bind_rate=("is_bound", "mean"), avg_ttc=("time_to_first_contact_min", "mean"))
        .sort_values("ttc_bucket")
    )
    st.dataframe(t, use_container_width=True, hide_index=True)


def _render_dashboard_segments(insights) -> None:
    """Segments tab - vertical, state, carrier analysis"""
    
    # Carrier Hit Ratio
    st.markdown("### 🎯 Carrier Hit Ratio")
    st.caption("Which carrier tier binds most for each vertical | Optimize Market-Making Engine")
    
    if not insights.carrier_hit_ratio.empty:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(insights.carrier_hit_ratio, use_container_width=True, hide_index=True)
        with col2:
            fig_carrier = px.bar(
                insights.carrier_hit_ratio.head(15),
                x="carrier_tier",
                y="avg_hit_ratio",
                color="policy_type",
                title="Carrier Hit Ratio by Tier"
            )
            fig_carrier.update_layout(height=400)
            st.plotly_chart(fig_carrier, use_container_width=True)
    else:
        st.info("Carrier hit ratio data not available")
    
    st.divider()
    
    # Human Leverage
    st.markdown("### 👥 Human Leverage Ratio")
    st.caption("Premium managed per agent with AI support | Target: 1:1,000 ratio")
    
    if not insights.human_leverage_ratio.empty:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(insights.human_leverage_ratio.head(30), use_container_width=True, hide_index=True)
        with col2:
            fig_leverage = px.bar(
                insights.human_leverage_ratio.head(20),
                x="agent_id",
                y="leverage_ratio",
                color="leverage_ratio",
                color_continuous_scale="Blues",
                title="Premium per Agent"
            )
            fig_leverage.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_leverage, use_container_width=True)
    else:
        st.info("Human leverage ratio data not available")
    
    st.divider()
    
    # Vertical Performance
    st.markdown("### 📊 Vertical Performance")
    v = insights.vertical_perf.copy()
    st.dataframe(v, use_container_width=True, hide_index=True)
    
    fig = px.bar(
        v.sort_values("egp24_per_lead", ascending=False).head(12),
        x="vertical",
        y="egp24_per_lead",
        color="egp24_per_lead",
        color_continuous_scale="Greens",
        title="Top Verticals by EGP24 per Lead"
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # State Performance
    st.markdown("### 🗺️ State Performance")
    s = insights.state_perf.copy()
    st.dataframe(s, use_container_width=True, hide_index=True)


def _render_dashboard_attribution(insights) -> None:
    """Attribution & Trends tab"""
    
    # Churn Risk
    st.markdown("### ⚠️ Churn Risk Analysis (Predictive)")
    st.caption("Flags current customers likely to defect | Enables 90-day renewal alerts")
    
    if not insights.churn_risk_analysis.empty:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(insights.churn_risk_analysis, use_container_width=True, hide_index=True)
        with col2:
            fig_churn = px.bar(
                insights.churn_risk_analysis,
                x="churn_bucket",
                y="total_premium_at_risk",
                color="churn_bucket",
                color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444", "Critical": "#991b1b"},
                title="Premium at Risk"
            )
            fig_churn.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_churn, use_container_width=True)
        
        high_risk = insights.churn_risk_analysis[insights.churn_risk_analysis["churn_bucket"].isin(["High", "Critical"])]
        if not high_risk.empty:
            total_at_risk = high_risk["total_premium_at_risk"].sum()
            st.warning(f"⚠️ **${total_at_risk:,.0f}** in premium at risk from High/Critical churn customers")
    else:
        st.info("Churn risk analysis data not available")
    
    st.divider()
    
    # Market Context
    st.markdown("### 🌍 Market Pricing Context")
    st.caption("External trends integration | Identify Harper's rate advantages")
    
    market_context = pd.DataFrame([
        {"segment": "Umbrella Coverage", "trend": "+11.5%", "harper_advantage": "Find cheaper rates"},
        {"segment": "E&S Construction", "trend": "+8.2%", "harper_advantage": "Better carrier access"},
        {"segment": "Tech SaaS", "trend": "-2.1%", "harper_advantage": "Competitive pricing"},
    ])
    st.dataframe(market_context, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Attribution
    st.markdown("### 📍 Attribution Analysis")
    st.caption("First-touch vs last-touch attribution")
    att = insights.attribution.copy()
    st.dataframe(att, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Daily Trends
    st.markdown("### 📈 Daily Trends")
    tr = insights.trends_daily.copy()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            tr,
            x="date",
            y=["leads", "bound"],
            title="Leads & Conversions",
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig2 = px.line(
            tr,
            x="date",
            y=["spend_usd", "premium"],
            title="Spend & Premium",
            markers=True
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)


def _render_dashboard(raw, insights, f_leads) -> None:
    """Render main dashboard"""
    
    # Header
    _render_header(insights)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "💰 Efficiency",
        "⚡ Speed-to-Lead",
        "🎯 Segments",
        "📍 Attribution"
    ])
    
    with tab1:
        _render_dashboard_overview(insights, f_leads)
    
    with tab2:
        _render_dashboard_efficiency(insights)
    
    with tab3:
        _render_dashboard_speed(insights, f_leads)
    
    with tab4:
        _render_dashboard_segments(insights)
    
    with tab5:
        _render_dashboard_attribution(insights)


def _render_copilot_ui(insights) -> None:
    """Render the copilot UI"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 AI Co‑Pilot")
        st.caption("Real-time strategic advice powered by your data")
        
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Set OPENAI_API_KEY or paste here"
        )
        model = st.selectbox("Model", options=["gpt-4o-mini", "gpt-4o"], index=0)
        temperature = st.slider("Creativity", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    client = _get_openai_client(api_key)
    
    if client is None:
        st.info("🔐 Set `OPENAI_API_KEY` to enable AI Co‑Pilot")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "copilot_client" not in st.session_state:
        st.session_state.copilot_client = client
    if "copilot_model" not in st.session_state:
        st.session_state.copilot_model = model
    if "copilot_temperature" not in st.session_state:
        st.session_state.copilot_temperature = temperature
    if "copilot_insights" not in st.session_state:
        st.session_state.copilot_insights = insights

    st.session_state.copilot_client = client
    st.session_state.copilot_model = model
    st.session_state.copilot_temperature = temperature
    st.session_state.copilot_insights = insights

    st.markdown("### 💬 Conversation")
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def _handle_copilot_input(insights) -> None:
    """Handle chat input and generate response"""
    
    if "copilot_client" not in st.session_state or st.session_state.copilot_client is None:
        return
    
    user_msg = st.chat_input("Ask about budget, strategy, operations…")
    if not user_msg:
        return

    st.session_state.messages.append({"role": "user", "content": user_msg})
    
    system_prompt = build_system_prompt(st.session_state.copilot_insights)
    client = st.session_state.copilot_client
    model = st.session_state.copilot_model
    temperature = st.session_state.copilot_temperature

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *st.session_state.messages[-12:],
                    ],
                    max_tokens=800,
                )
                answer = resp.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "Invalid_api_key" in error_msg or "Incorrect API key" in error_msg:
                    st.error("❌ Invalid or expired API key")
                    st.info("💡 Get a key from https://platform.openai.com/account/api-keys")
                else:
                    st.error(f"❌ Error: {error_msg}")
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    st.session_state.messages.pop()


# ============================================================================
# MAIN APP
# ============================================================================

def main() -> None:
    """Main app entry point"""
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    root_dir = str(Path(__file__).parent)
    cache_key = _get_data_cache_key(root_dir)
    raw = _load_all(root_dir, cache_key=cache_key)

    # Get filtered leads
    leads = raw["data"].leads.copy()
    f_leads, filter_hash = _render_filters(leads)
    
    # Filter related data
    f_lead_ids = set(f_leads["lead_id"].unique())
    f_touchpoints = raw["data"].touchpoints[raw["data"].touchpoints["lead_id"].isin(f_lead_ids)].copy()
    f_policies = raw["data"].policies[raw["data"].policies["lead_id"].isin(f_lead_ids)].copy()
    
    # Filter ad_spend by date range
    if len(f_leads) > 0:
        min_date = pd.to_datetime(f_leads["lead_created_at"]).min()
        max_date = pd.to_datetime(f_leads["lead_created_at"]).max()
        f_ad_spend = raw["data"].ad_spend_daily[
            (pd.to_datetime(raw["data"].ad_spend_daily["date"]) >= min_date) &
            (pd.to_datetime(raw["data"].ad_spend_daily["date"]) <= max_date)
        ].copy()
    else:
        f_ad_spend = raw["data"].ad_spend_daily.copy()
    
    # Compute insights
    insights = _compute_filtered_insights(f_leads, f_ad_spend, raw["data"].agents, f_touchpoints, f_policies, filter_hash)
    
    # Main layout: Dashboard + Copilot
    col_main, col_copilot = st.columns([2.2, 1], gap="large")
    
    with col_main:
        _render_dashboard(raw, insights, f_leads)
    
    with col_copilot:
        _render_copilot_ui(insights)
        _handle_copilot_input(insights)


if __name__ == "__main__":
    main()
