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


APP_TITLE = "Harper Growth Intelligence OS"
APP_SUBTITLE = "Real-time GTM optimization analytics with co-pilot support"

# Modern CSS with gradients and improved aesthetics
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700&display=swap');
    
    /* Root variables */
    :root {
        --primary: #0f172a;
        --primary-light: #1e293b;
        --accent: #10b981;
        --accent-dark: #059669;
        --accent-light: #d1fae5;
        --warning: #f59e0b;
        --danger: #ef4444;
        --success: #10b981;
        --neutral-50: #f9fafb;
        --neutral-100: #f3f4f6;
        --neutral-200: #e5e7eb;
        --neutral-300: #d1d5db;
        --neutral-600: #4b5563;
        --neutral-700: #374151;
        --neutral-900: #111827;
    }
    
    /* Main app background with gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0f9ff 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar with gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] [class*="css"] {
        color: white;
    }
    
    /* Typography - Larger, bolder headers */
    h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    h2 {
        font-family: 'Inter', sans-serif;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--primary) !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, var(--accent) 0%, transparent 100%) 1;
    }
    
    h3 {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--primary-light) !important;
        margin-top: 1.5rem;
    }
    
    /* Ensure all markdown headers are bold */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-weight: 700 !important;
    }
    
    /* Metric cards with gradient borders - Fixed alignment */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 1.25rem !important;
        border-left: 4px solid var(--accent);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 0.5rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    
    [data-testid="metric-container"]:hover {
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15);
        transform: translateY(-2px);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: var(--primary) !important;
        text-align: center !important;
        margin: 0.5rem 0 !important;
        line-height: 1.2 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: var(--neutral-600) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        text-align: center !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
    }
    
    [data-testid="stMetricDelta"] {
        text-align: center !important;
        margin-top: 0.5rem !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    
    [data-testid="stMetricDelta"] > div {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.25rem !important;
        flex-wrap: nowrap !important;
    }
    
    /* Hide all arrows in metric deltas */
    [data-testid="stMetricDelta"] svg {
        display: none !important;
    }
    
    [data-testid="stMetricDelta"] > div > span {
        display: inline-block !important;
        vertical-align: middle !important;
        line-height: 1.2 !important;
    }
    
    /* Dataframes - Light colored tables */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        background: #f9fafb !important;
    }
    
    [data-testid="stDataFrame"] table {
        background: #f9fafb !important;
    }
    
    [data-testid="stDataFrame"] thead {
        background: #f3f4f6 !important;
    }
    
    [data-testid="stDataFrame"] thead th {
        background: #f3f4f6 !important;
        color: #111827 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #e5e7eb !important;
    }
    
    [data-testid="stDataFrame"] tbody tr {
        background: #ffffff !important;
    }
    
    [data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background: #f9fafb !important;
    }
    
    [data-testid="stDataFrame"] tbody td {
        color: #374151 !important;
        border-bottom: 1px solid #e5e7eb !important;
    }
    
    /* Tabs with modern styling - BIGGER headings */
    [data-testid="stTabs"] {
        margin-top: 1rem;
    }
    
    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.5rem;
        border-bottom: 2px solid var(--neutral-200);
        padding-bottom: 0;
        margin-bottom: 1.5rem;
    }
    
    [data-testid="stTabs"] [role="tab"] {
        padding: 1.25rem 2.5rem;
        border-radius: 8px 8px 0 0;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
        transition: all 0.2s ease;
    }
    
    [data-testid="stTabs"] [aria-selected="true"] {
        color: var(--accent) !important;
        background: linear-gradient(180deg, rgba(16, 185, 129, 0.1) 0%, transparent 100%);
        border-bottom: 3px solid var(--accent) !important;
        font-size: 1.6rem !important;
        font-weight: 800 !important;
    }
    
    [data-testid="stTabs"] [aria-selected="false"] {
        color: var(--neutral-600) !important;
        font-size: 1.4rem !important;
    }
    
    [data-testid="stTabs"] [aria-selected="false"]:hover {
        color: var(--primary) !important;
        background: var(--neutral-50);
        font-size: 1.5rem !important;
    }
    
    /* Sidebar styling - ensure all text is white */
    [data-testid="stSidebar"] .stSubheader,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSubheader,
    [data-testid="stSidebar"] label {
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] .stCaption {
        color: rgba(255, 255, 255, 0.85) !important;
    }
    
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: var(--primary) !important;
    }
    
    /* Ensure sidebar markdown headers are white */
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: white !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div > div,
    .stDateInput > div > div > input {
        border-radius: 8px !important;
        border: 1px solid var(--neutral-200) !important;
        background-color: white !important;
        padding: 0.5rem 0.75rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
    }
    
    /* Secondary buttons (toggle buttons) - Better contrast */
    .stButton > button[kind="secondary"],
    button[kind="secondary"] {
        background: #ffffff !important;
        color: #111827 !important;
        border: 2px solid #e5e7eb !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }
    
    .stButton > button[kind="secondary"]:hover,
    button[kind="secondary"]:hover {
        background: #f9fafb !important;
        border-color: #10b981 !important;
        color: #10b981 !important;
        box-shadow: 0 4px 8px rgba(16, 185, 129, 0.15) !important;
    }
    
    /* Expanders */
    [data-testid="stExpander"] {
        background: white;
        border: 1px solid var(--neutral-200);
        border-radius: 12px;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stExpander"] summary {
        color: var(--primary);
        font-weight: 600;
        padding: 1rem;
    }
    
    /* Info boxes */
    [data-testid="stInfo"],
    [data-testid="stSuccess"],
    [data-testid="stWarning"],
    [data-testid="stError"] {
        border-radius: 12px;
        padding: 1rem !important;
        border-left: 4px solid;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stInfo"] {
        border-left-color: #3b82f6;
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
    }
    
    [data-testid="stSuccess"] {
        border-left-color: var(--success);
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
    }
    
    [data-testid="stWarning"] {
        border-left-color: var(--warning);
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
    }
    
    [data-testid="stError"] {
        border-left-color: var(--danger);
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
    }
    
    /* Plotly charts - Fix label visibility */
    .plotly-graph-div {
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        background: white;
        padding: 0.5rem;
    }
    
    /* Ensure plotly text is visible */
    .plotly .xtick text,
    .plotly .ytick text,
    .plotly .xaxis-title,
    .plotly .yaxis-title,
    .plotly .g-xtitle,
    .plotly .g-ytitle,
    .plotly text {
        fill: #111827 !important;
        color: #111827 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }
    
    .plotly .legendtext {
        fill: #111827 !important;
        color: #111827 !important;
    }
    
    /* Fix axis labels */
    .js-plotly-plot .plotly .xaxis .tick text,
    .js-plotly-plot .plotly .yaxis .tick text {
        fill: #374151 !important;
        color: #374151 !important;
    }
    
    /* Caption styling - ensure readability */
    .stCaption {
        color: var(--neutral-700) !important;
        font-size: 0.9rem;
        margin-top: 0.25rem;
        margin-bottom: 1rem;
    }
    
    /* Ensure all text is readable */
    p, div, span, label {
        color: var(--neutral-900) !important;
    }
    
    /* Right sidebar container for copilot chat */
    .copilot-container {
        background: white;
        border-radius: 12px;
        border: 2px solid var(--neutral-200);
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
    }
    
    .copilot-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--neutral-200);
    }
    
    .chat-messages-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem 0;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, var(--neutral-200) 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--neutral-100);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent) 0%, var(--accent-dark) 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-dark);
    }
    
    /* Fade-in animation */
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
    
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: var(--accent) !important;
        border-right-color: transparent !important;
    }
    
    /* Chat input - Translucent, no white box on hover */
    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stChatInput"]:hover,
    [data-testid="stChatInput"]:focus {
        background: rgba(255, 255, 255, 0.85) !important;
        border-color: rgba(16, 185, 129, 0.5) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.15) !important;
        transform: translateY(-2px);
    }
    
    [data-testid="stChatInput"] input {
        background: transparent !important;
        color: #111827 !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
    }
    
    [data-testid="stChatInput"] input::placeholder {
        color: #6b7280 !important;
        opacity: 0.8 !important;
    }
    
    /* Ensure text is always visible in chat input */
    [data-testid="stChatInput"] input:not(:placeholder-shown) {
        color: #111827 !important;
    }
    
    [data-testid="stChatInput"] input:focus {
        color: #111827 !important;
    }
    
    /* Sidebar toggle buttons - positioned on sides */
    .sidebar-toggle-btn {
        position: fixed;
        top: 50%;
        left: 0;
        transform: translateY(-50%);
        z-index: 1001;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
        color: white;
        border: none;
        border-radius: 0 8px 8px 0;
        padding: 1rem 0.5rem;
        font-weight: 600;
        font-size: 1.2rem;
        box-shadow: 2px 0 12px rgba(0, 0, 0, 0.15);
        cursor: pointer;
        transition: all 0.3s ease;
        writing-mode: vertical-rl;
        text-orientation: mixed;
    }
    
    .sidebar-toggle-btn:hover {
        left: 4px;
        box-shadow: 4px 0 20px rgba(16, 185, 129, 0.3);
    }
    
    .copilot-toggle-btn {
        position: fixed;
        top: 50%;
        right: 0;
        transform: translateY(-50%);
        z-index: 1001;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px 0 0 8px;
        padding: 1rem 0.5rem;
        font-weight: 600;
        font-size: 1.2rem;
        box-shadow: -2px 0 12px rgba(0, 0, 0, 0.15);
        cursor: pointer;
        transition: all 0.3s ease;
        writing-mode: vertical-rl;
        text-orientation: mixed;
    }
    
    .copilot-toggle-btn:hover {
        right: 4px;
        box-shadow: -4px 0 20px rgba(59, 130, 246, 0.3);
    }
    
    /* Align comments at bottom of columns */
    .column-container {
        display: flex;
        flex-direction: column;
        min-height: 100%;
    }
    
    .column-comment {
        margin-top: auto;
        padding-top: 1rem;
    }
</style>
"""


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


def _filter_useful_columns(df: pd.DataFrame, table_type: str) -> pd.DataFrame:
    """Remove non-useful columns from dataframes to keep tables focused and clean."""
    df = df.copy()
    
    # Define which columns to keep for each table type (only essential/useful ones)
    useful_columns = {
        "funnel": ["stage", "count", "rate_vs_leads", "conversion_lift"],  # Remove premium/egp24 (zeros for non-bound stages)
        "stq_by_vertical": ["vertical", "avg_stq_ratio", "quoted"],  # Remove stq_count
        "lead_score_intensity": ["vertical", "leads", "avg_intent_score", "avg_risk_score"],  # Remove opportunity_score and avg_egp24
        "vertical_opportunity_score": ["vertical", "leads", "avg_premium", "opportunity_score", "bound_rate", "avg_egp24"],  # Added avg_egp24 for Expected Profit per Lead (24-mo)
        "lead_quality_by_channel": ["source_channel", "leads", "bound_rate", "avg_risk_score", "roi_egp24", "quality_score"],  # Remove total_egp24, total_cac, high_risk_pct, avg_egp24
        "paid_efficiency_by_channel": ["channel", "leads", "bound", "spend_usd", "cpl", "cpb", "roi_egp24", "premium"],  # Remove cost_per_1k_premium
        "paid_efficiency_by_campaign": ["campaign_id", "channel", "leads", "bound", "spend_usd", "cpl", "roi_egp24"],  # Remove cpb, premium (less useful for campaigns)
        "ai_vs_human_perf": ["ai_contact_type", "leads", "contacted_rate", "bound_rate"],  # Remove quoted_rate, swapped contacted_rate and bound_rate positions
        "golden_window_leakage": ["leads_contacted_after_5m", "leakage_egp24_usd", "potential_egp24_if_fixed"],  # Keep all
        "speed_to_lead": ["ttc_bucket", "contacted", "avg_p_bind", "quoted_rate"],  # Removed bound_rate column
        "carrier_hit_ratio": ["carrier_tier", "policy_type", "avg_hit_ratio", "total_premium"],  # Remove policies count
        "human_leverage_ratio": ["agent_id", "leads", "premium_per_agent"],  # Removed leverage_ratio, include leads column
        "vertical_perf": ["vertical", "leads", "egp24_per_lead", "bound_rate"],  # Bound Rate as last column
        "state_perf": ["state", "leads", "egp24_per_lead", "bound_rate"],  # Bound Rate as last column
        "churn_risk_analysis": ["churn_bucket", "customers", "total_premium_at_risk"],  # Remove avg_churn_risk
        "attribution": ["channel", "first_touch_bound", "last_touch_bound"],  # Remove first_touch_leads, last_touch_leads
        "operational_drilldown": ["ttc_bucket", "leads", "bound_rate"],  # Remove avg_ttc, avg_ttc_min
        "market_context": ["segment", "trend", "harper_advantage"]  # Keep all
    }
    
    if table_type in useful_columns:
        cols_to_keep = useful_columns[table_type]
        # Only keep columns that exist in the dataframe
        available_cols = [col for col in cols_to_keep if col in df.columns]
        if available_cols:
            df = df[available_cols]
    
    return df


def _format_channel_name(channel: str) -> str:
    """Format channel names: Meta_Retargeting -> Meta Retargeting (remove underscores)"""
    if pd.isna(channel) or not isinstance(channel, str):
        return channel
    # Replace all underscores with spaces
    return channel.replace('_', ' ')


def _format_vertical_name(vertical: str) -> str:
    """Format vertical names: Professional_Services -> Professional Services (remove underscores)"""
    if pd.isna(vertical) or not isinstance(vertical, str):
        return vertical
    # Replace all underscores with spaces
    return vertical.replace('_', ' ')


def _semantic_column_names(df: pd.DataFrame, table_type: str = "default") -> pd.DataFrame:
    """Convert technical column names to semantic, user-friendly names."""
    df = df.copy()
    
    # Format channel names for tables that have channel columns
    channel_columns = ["channel", "source_channel", "origin_channel"]
    for col in channel_columns:
        if col in df.columns:
            df[col] = df[col].apply(_format_channel_name)
    
    # Format vertical names for tables that have vertical columns
    if "vertical" in df.columns:
        df["vertical"] = df["vertical"].apply(_format_vertical_name)
    
    # Define semantic mappings for different table types
    semantic_maps = {
        "funnel": {
            "stage": "Stage",
            "count": "Count",
            "rate_vs_leads": "Conversion Rate",
            "premium_annual_usd_total": "Total Premium (Annual)",
            "egp_24m_total_usd": "Expected Gross Profit (24M)",
            "conversion_lift": "Conversion Lift"
        },
        "stq_by_vertical": {
            "vertical": "Vertical",
            "quoted": "Quoted Leads",
            "avg_stq_ratio": "Avg STQ Ratio",
            "stq_count": "STQ Count"
        },
        "lead_score_intensity": {
            "vertical": "Vertical",
            "leads": "Leads",
            "avg_intent_score": "Avg Intent Score",
            "avg_risk_score": "Avg Risk Score",
            "opportunity_score": "Opportunity Score",
            "avg_egp24": "Avg EGP24"
        },
        "vertical_opportunity_score": {
            "vertical": "Vertical",
            "leads": "Leads",
            "avg_intent_score": "Avg Intent Score",
            "avg_premium": "Avg Premium",
            "bound_rate": "Bound Rate",
            "opportunity_score": "Opportunity Score",
            "avg_egp24": "Expected Profit per Lead (24-mo)"
        },
        "lead_quality_by_channel": {
            "source_channel": "Channel",
            "leads": "Leads",
            "avg_risk_score": "Avg Risk Score",
            "high_risk_pct": "High Risk %",
            "bound_rate": "Bound Rate",
            "avg_egp24": "Avg EGP24",
            "total_egp24": "Total EGP24",
            "total_cac": "Total CAC",
            "quality_score": "Quality Score",
            "roi_egp24": "ROI"
        },
        "paid_efficiency_by_channel": {
            "channel": "Channel",
            "leads": "Leads",
            "bound": "Bound",
            "spend_usd": "Spend ($)",
            "cpl": "Cost Per Lead",
            "cpb": "Cost Per Bound",
            "premium": "Premium",
            "roi_egp24": "ROI",
            "cost_per_1k_premium": "Cost per $1K Premium"
        },
        "paid_efficiency_by_campaign": {
            "campaign_id": "Campaign ID",
            "channel": "Channel",
            "leads": "Leads",
            "bound": "Bound",
            "spend_usd": "Spend ($)",
            "cpl": "Cost Per Lead",
            "cpb": "Cost Per Bound",
            "premium": "Premium",
            "roi_egp24": "ROI"
        },
        "ai_vs_human_perf": {
            "ai_contact_type": "Contact Type",
            "leads": "Assigned Leads",
            "contacted_rate": "Contacted Rate",
            "quoted_rate": "Quoted Rate",
            "bound_rate": "Bound Rate"
        },
        "golden_window_leakage": {
            "leads_contacted_after_5m": "Leads Contacted After 5min",
            "leakage_egp24_usd": "Leakage EGP24 ($)",
            "potential_egp24_if_fixed": "Potential EGP24 if Fixed ($)"
        },
        "speed_to_lead": {
            "ttc_bucket": "Time-to-Contact",
            "leads": "Leads",
            "quoted": "Contacted Leads",
            "contacted": "Contacted Leads",
            "quoted_leads": "Contacted Leads",
            "bound_rate": "Bound Rate",
            "quoted_rate": "Quote Rate",
            "quote_rate": "Quote Rate",
            "avg_p_bind": "Lead Quality Perception",
            "avg_ttc": "Avg TTC (min)",
            "avg_ttc_min": "Avg TTC (min)"
        },
        "carrier_hit_ratio": {
            "carrier_tier": "Carrier Tier",
            "policy_type": "Policy Type",
            "policies": "Policies",
            "avg_hit_ratio": "Avg Hit Ratio",
            "total_premium": "Total Premium"
        },
        "human_leverage_ratio": {
            "agent_id": "Agent ID",
            "leads": "Leads Handled",
            "premium_per_agent": "Premium per Agent",
            "leads_per_agent": "Leads per Agent",
            "leverage_ratio": "Leverage Ratio"
        },
        "vertical_perf": {
            "vertical": "Vertical",
            "leads": "Leads",
            "bound": "Bound",
            "bind_rate": "Bind Rate",
            "bound_rate": "Bound Rate",
            "quote_rate": "Quote Rate",
            "avg_ttc_min": "Avg TTC (min)",
            "premium": "Premium",
            "premium_per_lead": "Premium per Lead",
            "egp24_per_lead": "Expected Profit per Lead (24-mo)"
        },
        "state_perf": {
            "state": "State",
            "leads": "Leads",
            "bound": "Bound",
            "bind_rate": "Bind Rate",
            "bound_rate": "Bound Rate",
            "quote_rate": "Quote Rate",
            "avg_ttc_min": "Avg TTC (min)",
            "premium": "Premium",
            "premium_per_lead": "Premium per Lead",
            "egp24_per_lead": "Expected Profit per Lead (24-mo)"
        },
        "agent_perf": {
            "agent_id": "Agent ID",
            "leads": "Leads",
            "contacted_rate": "Contacted Rate",
            "quote_rate": "Quote Rate",
            "bound_rate": "Bound Rate",
            "avg_ttc_min": "Avg TTC (min)",
            "egp24_total": "Total EGP24",
            "egp24_per_lead": "Expected Profit per Lead (24-mo)"
        },
        "churn_risk_analysis": {
            "churn_bucket": "Churn Risk",
            "customers": "Customers",
            "avg_churn_risk": "Avg Churn Risk",
            "total_premium_at_risk": "Premium at Risk ($)"
        },
        "attribution": {
            "channel": "Channel",
            "first_touch_leads": "First Touch Leads",
            "last_touch_leads": "Last Touch Leads",
            "first_touch_bound": "First Touch Bound",
            "last_touch_bound": "Last Touch Bound"
        },
        "trends_daily": {
            "date": "Date",
            "leads": "Leads",
            "bound": "Bound",
            "spend_usd": "Spend ($)",
            "premium": "Premium"
        },
        "market_context": {
            "segment": "Segment",
            "trend": "Trend",
            "harper_advantage": "Harper Advantage"
        },
        "operational_drilldown": {
            "ttc_bucket": "Time-to-Contact",
            "leads": "Leads",
            "bind_rate": "Bind Rate",
            "bound_rate": "Bound Rate",
            "avg_ttc": "Avg TTC (min)",
            "avg_ttc_min": "Avg TTC (min)"
        }
    }
    
    # Get the appropriate mapping
    mapping = semantic_maps.get(table_type, {})
    
    # Apply mapping for columns that exist in the dataframe
    rename_dict = {old: new for old, new in mapping.items() if old in df.columns}
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    return df


def _update_chart_layout(fig, height=400, showlegend=None):
    """Update plotly chart layout with visible labels and consistent styling."""
    layout_updates = {
        'height': height,
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'xaxis': dict(
            title_font=dict(size=14, color='#111827'),
            tickfont=dict(size=12, color='#374151'),
            gridcolor='#e5e7eb',
            linecolor='#d1d5db',
            title_standoff=10
        ),
        'yaxis': dict(
            title_font=dict(size=14, color='#111827'),
            tickfont=dict(size=12, color='#374151'),
            gridcolor='#e5e7eb',
            linecolor='#d1d5db',
            title_standoff=10
        ),
        'title': dict(
            font=dict(size=16, color='#111827', family='Inter'),
            x=0.5,
            xanchor='center'
        ),
        'font': dict(family='Inter', color='#111827', size=12)
    }
    if showlegend is not None:
        layout_updates['showlegend'] = showlegend
        if showlegend:
            layout_updates['legend'] = dict(
                font=dict(size=12, color='#111827'),
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#e5e7eb',
                borderwidth=1,
                x=0.5,
                xanchor='center',
                orientation='h',
                yanchor='bottom',
                y=-0.15
            )
    fig.update_layout(**layout_updates)
    
    # Center x-axis and y-axis titles
    fig.update_xaxes(title_standoff=10)
    fig.update_yaxes(title_standoff=10)
    
    return fig


def _get_openai_client(api_key: Optional[str]) -> Optional[OpenAI]:
    key = api_key or os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not key:
        return None
    # Strip whitespace in case there's any
    key = key.strip()
    if not key.startswith("sk-"):
        st.error(f"Invalid API key format. Key should start with 'sk-'. Got: {key[:10]}...")
        return None
    try:
        return OpenAI(api_key=key)
    except Exception as e:
        st.error(f"Failed to create OpenAI client: {str(e)}")
        return None


def _handle_toggle_query_params() -> None:
    """
    Make the side HTML toggle buttons functional without changing their style.
    We do this by writing a `?toggle=...` query param from JS and handling it here.
    """
    try:
        qp = st.query_params
        toggle = qp.get("toggle", None)
    except Exception:
        # Fallback for older Streamlit versions
        qp = st.experimental_get_query_params()
        toggle_list = qp.get("toggle", [])
        toggle = toggle_list[0] if toggle_list else None

    if not toggle:
        return

    if toggle == "filters":
        st.session_state.sidebar_collapsed = not st.session_state.get("sidebar_collapsed", False)
    elif toggle == "copilot":
        st.session_state.copilot_sidebar_open = not st.session_state.get("copilot_sidebar_open", False)

    # Clear the query param so refresh doesn't keep toggling
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()

    st.rerun()


def _render_header(insights) -> None:
    """Render modern header with gradient styling and improved layout"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Ensure session state exists
    if "sidebar_collapsed" not in st.session_state:
        st.session_state.sidebar_collapsed = False
    if "copilot_sidebar_open" not in st.session_state:
        st.session_state.copilot_sidebar_open = False

    # Handle query-param toggles from the styled HTML buttons
    _handle_toggle_query_params()
    
    # Sidebar toggle buttons - positioned on sides using HTML/CSS
    sidebar_state = st.session_state.get("sidebar_collapsed", False)
    copilot_open = st.session_state.get("copilot_sidebar_open", False)
    
    toggle_html = f"""
    <div style="position: fixed; top: 50%; left: 0; transform: translateY(-50%); z-index: 1001;">
        <button onclick="(function(){{ var u=new URL(window.location.href); u.searchParams.set('toggle','filters'); window.location.href=u.toString(); }})()" 
                class="sidebar-toggle-btn" 
                style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; border: none; border-radius: 0 8px 8px 0; padding: 1rem 0.5rem; font-weight: 600; font-size: 1.2rem; box-shadow: 2px 0 12px rgba(0, 0, 0, 0.15); cursor: pointer; transition: all 0.3s ease; writing-mode: vertical-rl; text-orientation: mixed;">
            ☰ Filters
        </button>
    </div>
    <div style="position: fixed; top: 50%; right: 0; transform: translateY(-50%); z-index: 1001;">
        <button onclick="(function(){{ var u=new URL(window.location.href); u.searchParams.set('toggle','copilot'); window.location.href=u.toString(); }})()" 
                class="copilot-toggle-btn" 
                style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; border: none; border-radius: 8px 0 0 8px; padding: 1rem 0.5rem; font-weight: 600; font-size: 1.2rem; box-shadow: -2px 0 12px rgba(0, 0, 0, 0.15); cursor: pointer; transition: all 0.3s ease; writing-mode: vertical-rl; text-orientation: mixed;">
            {'✕ Chat' if copilot_open else '💬 Chat'}
        </button>
    </div>
    """
    st.markdown(toggle_html, unsafe_allow_html=True)
    
    # Title section
    st.title(APP_TITLE)
    st.markdown(f'<div style="text-align: left; margin: 1rem 0 2rem 0;"><p style="color: #111827; font-size: 2.5rem; font-weight: 700; line-height: 1.3;">{APP_SUBTITLE}</p></div>', unsafe_allow_html=True)
    
    st.divider()
    
    # KPI Metrics in a responsive grid
    s = insights.summary
    metric_cols = st.columns(7)
    
    metrics_config = [
        ("📈", "Leads", f"{s['leads']:,}", None, "#3b82f6"),
        ("🎯", "Bound", f"{s['bound_leads']:,}", _pct(s["bind_rate"]), "#10b981"),
        ("⚙️", "Automated Binding", _pct(s.get("automation_rate", 0.0)), None, "#8b5cf6"),
        ("💰", "LTV:CAC", f"{s.get('cac_ltv_ratio', 0.0):.1f}:1", 
         "✅" if s.get('cac_ltv_ratio', 0) >= 3.0 else "⚠️", "#f59e0b"),
        ("💵", "Net Revenue", _money(s.get("net_commission_revenue_usd", 0)), None, "#10b981"),
        ("💸", "Spend", _money(s["total_spend_usd"]), None, "#ef4444"),
        ("🏆", "Premium", _money(s["total_premium_annual_usd"]), None, "#06b6d4"),
    ]
    
    for col, (icon, label, value, delta, color) in zip(metric_cols, metrics_config):
        with col:
            st.markdown(
                f'<div style="text-align: center; padding: 0.5rem;">'
                f'<div style="font-size: 1.5rem; margin-bottom: 0.25rem;">{icon}</div>'
                f'<div style="font-size: 1rem; color: #111827; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            col.metric("", value, delta)
    
    st.divider()


def _filter_leads(leads: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Filter leads and return filtered dataframe + hash for caching."""
    st.sidebar.markdown('<div style="padding: 1rem 0;"><h2 style="color: white; font-size: 1.3rem; margin: 0;">🔍 Filters</h2></div>', unsafe_allow_html=True)
    
    # OpenAI API Key input for copilot
    st.sidebar.divider()
    st.sidebar.markdown('<div style="padding: 1rem 0;"><h2 style="color: white; font-size: 1.3rem; margin: 0;">⚙️ Copilot Settings</h2></div>', unsafe_allow_html=True)
    
    # Initialize session state for API key
    if "user_openai_api_key" not in st.session_state:
        st.session_state.user_openai_api_key = ""
    
    # API key input
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key (optional)",
        value=st.session_state.user_openai_api_key,
        type="password",
        help="Enter your OpenAI API key to use the GTM Copilot. Leave empty to use the default demo key.",
        key="openai_key_input"
    )
    
    # Update session state when user enters key
    if api_key_input != st.session_state.user_openai_api_key:
        st.session_state.user_openai_api_key = api_key_input
        # Clear existing copilot client to force re-initialization with new key
        if "copilot_client" in st.session_state:
            del st.session_state.copilot_client
        st.rerun()
    
    if api_key_input:
        st.sidebar.success("✓ API key set")
    else:
        st.sidebar.info("💡 Using demo key. Enter your own key for full access.")
    
    st.sidebar.divider()
    
    min_date = pd.to_datetime(leads["lead_created_at"]).min().date()
    max_date = pd.to_datetime(leads["lead_created_at"]).max().date()
    date_range = st.sidebar.date_input("Lead created date", value=(min_date, max_date))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d0, d1 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        d0, d1 = pd.to_datetime(min_date), pd.to_datetime(max_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    channels = sorted(leads["source_channel"].dropna().unique().tolist())
    selected_channels = st.sidebar.multiselect("Source channel", options=channels, default=channels)

    verticals = sorted(leads["vertical"].dropna().unique().tolist())
    selected_verticals = st.sidebar.multiselect("Vertical", options=verticals, default=verticals)

    states = sorted(leads["state"].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect("State", options=states, default=states)

    es = st.sidebar.multiselect("E&S eligible", options=["All", "Yes", "No"], default=["All"])

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
    
    # Create hash for caching based on filter values
    filter_str = f"{d0}_{d1}_{sorted(selected_channels)}_{sorted(selected_verticals)}_{sorted(selected_states)}_{sorted(es)}"
    filter_hash = hashlib.md5(filter_str.encode()).hexdigest()
    
    return out, filter_hash


def _render_dashboard(raw) -> None:
    """Render dashboard using pre-computed filtered insights."""
    d = raw["data"]
    insights = raw["insights"]
    f_leads = raw.get("f_leads", d.leads.copy())

    _render_header(insights)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Overview", "💰 Paid Efficiency", "☎️ Outreach Intelligence", "🎯Target Segments", "📈 Performance Drivers"]
    )

    with tab1:
        # Funnel section
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 📊 Conversion Funnel")
            
            # Create dual-layer funnel visualization
            funnel = insights.funnel.copy()
            
            # Create funnel chart with annotations
            fig_funnel = go.Figure()
            
            # Layer 1: Primary funnel - Count at each stage
            # Use a gradient color scale for premium look
            colors = ["#3b82f6", "#2563eb", "#1d4ed8", "#1e40af"]  # Blue gradient
            
            fig_funnel.add_trace(go.Funnel(
                y=funnel["stage"],
                x=funnel["count"],
                textposition="inside",
                textinfo="value+percent initial",
                textfont=dict(size=14, color="white", family="Arial Black"),
                marker=dict(
                    color=[colors[min(i, len(colors)-1)] for i in range(len(funnel))],
                    line=dict(color="white", width=3),
                    opacity=0.9
                ),
                connector=dict(
                    line=dict(color="rgba(59, 130, 246, 0.3)", width=3, dash="dot")
                ),
                name="Count"
            ))
            
            # Layer 2: Add annotations for conversion rates and lift (overlay)
            annotations = []
            for idx, row in funnel.iterrows():
                # Calculate conversion rate for this step
                if idx == 0:
                    # First stage: show 100%
                    rate_text = "100%"
                    annotation_text = rate_text
                else:
                    # Calculate step conversion rate
                    prev_count = funnel.iloc[idx-1]["count"]
                    current_count = row["count"]
                    step_rate = (current_count / prev_count) if prev_count > 0 else 0
                    rate_text = f"{step_rate:.1%}"
                    
                    # Add conversion lift if available
                    lift = row.get('conversion_lift', 0)
                    if lift > 0 and idx > 1:
                        lift_text = f"↑{lift:.2f}x"
                        annotation_text = f"{rate_text}<br><span style='font-size:10px;'>{lift_text}</span>"
                    else:
                        annotation_text = rate_text
                
                # Position annotation on the right side of each funnel segment
                annotations.append(dict(
                    x=row['count'] * 1.15,  # Position to the right of the bar
                    y=row['stage'],
                    text=annotation_text,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="rgba(59, 130, 246, 0.6)",
                    ax=20,
                    ay=0,
                    font=dict(size=12, color="#1e40af", family="Arial Black"),
                    bgcolor="rgba(255, 255, 255, 0.95)",
                    bordercolor="#3b82f6",
                    borderwidth=2,
                    borderpad=6,
                    xref="x",
                    yref="y",
                    xanchor="left",
                    yanchor="middle"
                ))
            
            fig_funnel.update_layout(
                title="",
                height=450,
                margin=dict(l=100, r=150, t=20, b=40),
                annotations=annotations,
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                hovermode="y unified"
            )
            
            # Update axes
            fig_funnel.update_xaxes(
                title_text="<b>Count</b>",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                title_font=dict(size=13, color="#111827", family="Arial"),
                tickfont=dict(size=11, color="#6b7280")
            )
            
            fig_funnel.update_yaxes(
                title_text="<b>Stage</b>",
                showgrid=False,
                title_font=dict(size=13, color="#111827", family="Arial"),
                tickfont=dict(size=12, color="#111827", family="Arial")
            )
            
            _update_chart_layout(fig_funnel, height=450, showlegend=False)
            st.plotly_chart(fig_funnel, use_container_width=True)
            
            # Optional: Show table below for detailed view
            funnel_table = _filter_useful_columns(insights.funnel.copy(), "funnel")
            funnel_table = _semantic_column_names(funnel_table, "funnel")
            with st.expander("📋 View Detailed Funnel Data", expanded=False):
                st.dataframe(funnel_table, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 💡 Quick Wins")
            if insights.recommendations:
                for i, r in enumerate(insights.recommendations[:3]):
                    with st.expander(f"#{i+1} {r.get('title', 'Recommendation')}", expanded=(i==0)):
                        st.markdown(f"**Why:** {r.get('why')}")
                        st.markdown(f"**Action:** {r.get('what_to_do')}")
            else:
                st.info("No recommendations available")
        
        st.divider()
        
        # Golden Window Leakage section
        st.markdown("### ⏰ Golden Window Leakage")
        st.caption("Revenue lost from leads contacted after 5 minutes | 'Burning platform' for engineering")
        if not insights.golden_window_leakage.empty:
            leakage = insights.golden_window_leakage.iloc[0]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("💰 Leakage", _money(leakage.get("leakage_egp24_usd", 0)), delta=None)
            with col2:
                st.metric("🎯 Potential if Fixed", _money(leakage.get("potential_egp24_if_fixed", 0)), delta=None)
        else:
            st.info("Golden window leakage data not available.")
        
        st.divider()
        
        # STQ Ratio section
        st.markdown("### ⏱️ Submission-to-Quote (STQ) Ratio by Vertical")
        st.caption("Lower = faster carrier response | Identify bottlenecks")
        if not insights.stq_by_vertical.empty:
            col1, col2 = st.columns([1, 1])
            with col1:
                stq_filtered = _filter_useful_columns(insights.stq_by_vertical.copy(), "stq_by_vertical")
                st.dataframe(_semantic_column_names(stq_filtered, "stq_by_vertical"), use_container_width=True, hide_index=True)
            with col2:
                # Use semantic column names for chart
                stq_chart = insights.stq_by_vertical.head(10).copy()
                stq_chart = stq_chart.rename(columns={"vertical": "Vertical", "avg_stq_ratio": "Avg STQ Ratio"})
                # Format vertical names in chart
                if "Vertical" in stq_chart.columns:
                    stq_chart["Vertical"] = stq_chart["Vertical"].apply(_format_vertical_name)
                fig_stq = px.bar(
                    stq_chart,
                    x="Vertical",
                    y="Avg STQ Ratio",
                    color="Avg STQ Ratio",
                    color_continuous_scale="RdYlGn_r",
                    title="STQ Ratio by Vertical",
                    labels={"Vertical": "Vertical", "Avg STQ Ratio": "Avg STQ Ratio"}
                )
                _update_chart_layout(fig_stq, height=400, showlegend=False)
                st.plotly_chart(fig_stq, use_container_width=True)
        else:
            st.info("STQ data not available.")
        
        st.divider()
        
        st.markdown("### 🎯 Lead Score Intensity Map")
        st.caption("Intent + Risk score distribution | Find high-value opportunities")
        if not insights.lead_score_intensity.empty:
            col1, col2 = st.columns([1, 1])
            with col1:
                intensity_filtered = _filter_useful_columns(insights.lead_score_intensity.copy(), "lead_score_intensity")
                st.dataframe(_semantic_column_names(intensity_filtered, "lead_score_intensity"), use_container_width=True, hide_index=True)
            with col2:
                # Use semantic column names for chart
                intensity_chart = insights.lead_score_intensity.copy()
                intensity_chart = intensity_chart.rename(columns={
                    "avg_intent_score": "Avg Intent Score",
                    "avg_risk_score": "Avg Risk Score",
                    "leads": "Leads",
                    "opportunity_score": "Opportunity Score",
                    "vertical": "Vertical"
                })
                fig_intensity = px.scatter(
                    intensity_chart,
                    x="Avg Intent Score",
                    y="Avg Risk Score",
                    size="Leads",
                    color="Opportunity Score",
                    hover_data=["Vertical"],
                    color_continuous_scale="Viridis",
                    title="Intent vs Risk by Vertical",
                    labels={"Avg Intent Score": "Avg Intent Score", "Avg Risk Score": "Avg Risk Score"}
                )
                _update_chart_layout(fig_intensity, height=400, showlegend=True)
                st.plotly_chart(fig_intensity, use_container_width=True)
        else:
            st.info("Lead score intensity data not available.")

    with tab2:
        st.markdown("### 💰 Paid Efficiency by Channel")
        ch = insights.paid_efficiency_by_channel.copy()
        
        # Use semantic column names for chart - convert to grouped bar chart
        ch_chart = ch.copy()
        ch_chart = ch_chart.rename(columns={
            "cpl": "Cost Per Lead",
            "roi_egp24": "ROI",
            "spend_usd": "Spend ($)",
            "channel": "Channel",
            "cpb": "Cost Per Bound",
            "cost_per_1k_premium": "Cost per $1K Premium",
            "leads": "Leads",
            "bound": "Bound",
            "premium": "Premium"
        })
        # Format channel names in chart
        if "Channel" in ch_chart.columns:
            ch_chart["Channel"] = ch_chart["Channel"].apply(_format_channel_name)
        
        # Convert ROI ratio to actual dollar value comparable to Cost Per Lead
        # ROI Value per Lead = ROI * Cost Per Lead (gives expected gross profit per lead in dollars)
        ch_chart["ROI Value per Lead"] = ch_chart["ROI"] * ch_chart["Cost Per Lead"]
        
        # Prepare data for grouped bar chart
        # Create a long format dataframe with metric type and value
        ch_melted = pd.melt(
            ch_chart,
            id_vars=["Channel"],
            value_vars=["Cost Per Lead", "ROI Value per Lead"],
            var_name="Metric",
            value_name="Value"
        )
        
        # Create grouped bar chart
        fig = px.bar(
            ch_melted,
            x="Channel",
            y="Value",
            color="Metric",
            barmode="group",
            title="Channel Tradeoffs: Cost Per Lead vs ROI",
            labels={"Channel": "Channel", "Value": "Value", "Metric": "Metric"},
            color_discrete_map={"Cost Per Lead": "#ef4444", "ROI Value per Lead": "#10b981"}
        )
        # Position legend at bottom right
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="right",
                x=1.0,
                title_text=""  # Remove "Metric" title from legend
            ),
            margin=dict(b=80)  # Add bottom margin for legend
        )
        _update_chart_layout(fig, height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table below the chart
        ch_filtered = _filter_useful_columns(ch, "paid_efficiency_by_channel")
        st.dataframe(_semantic_column_names(ch_filtered, "paid_efficiency_by_channel"), use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.markdown("### 🔍 Lead Quality by Channel")
        st.caption("ROI + Risk Score analysis | Prevent funnel clogging with uninsurable leads")
        if not insights.lead_quality_by_channel.empty:
            col1, col2 = st.columns([1, 1])
            with col1:
                quality_filtered = _filter_useful_columns(insights.lead_quality_by_channel.copy(), "lead_quality_by_channel")
                st.dataframe(_semantic_column_names(quality_filtered, "lead_quality_by_channel"), use_container_width=True, hide_index=True)
            with col2:
                # Use semantic column names for chart
                quality_chart = insights.lead_quality_by_channel.copy()
                quality_chart = quality_chart.rename(columns={
                    "avg_risk_score": "Avg Risk Score",
                    "roi_egp24": "ROI",
                    "leads": "Leads",
                    "quality_score": "Quality Score",
                    "source_channel": "Channel"
                })
                # Format channel names in chart
                if "Channel" in quality_chart.columns:
                    quality_chart["Channel"] = quality_chart["Channel"].apply(_format_channel_name)
                fig_quality = px.scatter(
                    quality_chart,
                    x="Avg Risk Score",
                    y="ROI",
                    size="Leads",
                    color="Quality Score",
                    hover_data=["Channel"],
                    color_continuous_scale="RdYlGn",
                    title="Risk Score vs ROI by Channel",
                    labels={"Avg Risk Score": "Avg Risk Score", "ROI": "ROI"}
                )
                _update_chart_layout(fig_quality, height=400, showlegend=True)
                st.plotly_chart(fig_quality, use_container_width=True)
        else:
            st.info("Lead quality data not available.")
        
        st.divider()
        
        st.markdown("### 🏆 Top Campaigns (by spend)")
        camp = insights.paid_efficiency_by_campaign.copy().head(25)
        camp_filtered = _filter_useful_columns(camp, "paid_efficiency_by_campaign")
        st.dataframe(_semantic_column_names(camp_filtered, "paid_efficiency_by_campaign"), use_container_width=True, hide_index=True)

    with tab3:
        # Outreach Intelligence Impact and AI Voice vs Human Performance in side-by-side columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Use a container div with flexbox to align comment at bottom
            st.markdown('<div class="column-container">', unsafe_allow_html=True)
            st.markdown("### ⚡ Time-to-lead impact")
            speed = insights.speed_to_lead.copy()
            
            # Define time bucket order (fastest to slowest)
            time_order = ["<=5m", "5-15m", "15-60m", "1-4h", "4-24h", "1-7d"]
            
            # Filter out 1-7d row
            speed = speed[speed["ttc_bucket"] != "1-7d"].copy()
            
            # Create a mapping for sorting
            speed["_sort_order"] = speed["ttc_bucket"].map({bucket: idx for idx, bucket in enumerate(time_order)})
            speed = speed.sort_values("_sort_order", na_position="last").drop(columns=["_sort_order"])
            
            # Calculate contacted leads (leads * contacted_rate)
            if "contacted_rate" in speed.columns and "leads" in speed.columns:
                speed["contacted"] = (speed["leads"] * speed["contacted_rate"]).round().astype(int)
                # Keep quote_rate and avg_p_bind for display, replace leads with contacted
                speed_display = speed.copy()
                speed_display = speed_display.drop(columns=["leads"])
                # Keep contacted column for semantic mapping
            else:
                speed_display = speed.copy()
            
            # Chart on top
            speed_chart = speed.copy()
            speed_chart = speed_chart.rename(columns={
                "ttc_bucket": "Time-to-Contact",
                "bound_rate": "Bound Rate"
            })
            # Ensure chart uses the sorted order
            speed_chart["Time-to-Contact"] = pd.Categorical(speed_chart["Time-to-Contact"], categories=time_order, ordered=True)
            speed_chart = speed_chart.sort_values("Time-to-Contact")
            
            fig = px.bar(
                speed_chart,
                x="Time-to-Contact",
                y="Bound Rate",
                color="Bound Rate",
                color_continuous_scale="Greens",
                title="Bind Rate by Time-to-Contact",
                labels={"Time-to-Contact": "Time-to-Contact", "Bound Rate": "Bound Rate"},
                category_orders={"Time-to-Contact": time_order}
            )
            _update_chart_layout(fig, height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Table below (already sorted) - use speed_display which has contacted leads, quote_rate, and avg_p_bind
            speed_filtered = _filter_useful_columns(speed_display, "speed_to_lead")
            speed_semantic = _semantic_column_names(speed_filtered, "speed_to_lead")
            # Ensure "contacted" or "quoted" is displayed as "Contacted Leads"
            if "quoted_leads" in speed_semantic.columns:
                speed_semantic = speed_semantic.rename(columns={"quoted_leads": "Contacted Leads"})
            elif "quoted" in speed_semantic.columns:
                speed_semantic = speed_semantic.rename(columns={"quoted": "Contacted Leads"})
            elif "contacted" in speed_semantic.columns:
                speed_semantic = speed_semantic.rename(columns={"contacted": "Contacted Leads"})
            # Also check for "Leads" column and rename if needed
            if "Leads" in speed_semantic.columns:
                speed_semantic = speed_semantic.rename(columns={"Leads": "Contacted Leads"})
            st.dataframe(speed_semantic, use_container_width=True, hide_index=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Use a container div with flexbox to align comment at bottom
            st.markdown('<div class="column-container">', unsafe_allow_html=True)
            st.markdown("### 🤖 AI Voice vs Human Performance")
            if not insights.ai_vs_human_perf.empty:
                # Chart on top
                ai_chart = insights.ai_vs_human_perf.copy()
                # Map contact type values to readable format
                contact_type_map = {
                    "ai_voice": "AI Voice",
                    "ai_email": "AI Email",
                    "mixed": "Mixed",
                    "human": "Human"
                }
                ai_chart["ai_contact_type"] = ai_chart["ai_contact_type"].map(contact_type_map).fillna(ai_chart["ai_contact_type"])
                ai_chart = ai_chart.rename(columns={
                    "ai_contact_type": "Contact Type",
                    "bound_rate": "Bound Rate",
                    "contacted_rate": "Contacted Rate"
                })
                fig_ai = px.bar(
                    ai_chart,
                    x="Contact Type",
                    y="Bound Rate",
                    color="Contacted Rate",
                    color_continuous_scale="Blues",
                    title="AI vs Human: Bound Rate",
                    labels={"Contact Type": "Contact Type", "Bound Rate": "Bound Rate"}
                )
                _update_chart_layout(fig_ai, height=400, showlegend=False)
                st.plotly_chart(fig_ai, use_container_width=True)
                
                # Table below - also map contact types
                ai_filtered = _filter_useful_columns(insights.ai_vs_human_perf.copy(), "ai_vs_human_perf")
                if "ai_contact_type" in ai_filtered.columns:
                    ai_filtered["ai_contact_type"] = ai_filtered["ai_contact_type"].map(contact_type_map).fillna(ai_filtered["ai_contact_type"])
                st.dataframe(_semantic_column_names(ai_filtered, "ai_vs_human_perf"), use_container_width=True, hide_index=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("AI vs Human performance data not available.")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Comments aligned above Agent Productivity Index
        col_comment1, col_comment2 = st.columns([1, 1])
        with col_comment1:
            st.markdown("💡 Contacting within 5 minutes drives a ~5× higher bound rate vs 4–24h delays.")
        with col_comment2:
            st.markdown("💡 AI Voice delivers ~1.2× higher bound rate than human outreach with higher contact coverage")
        
        st.markdown("### 💼 Agent Productivity Index")
        st.caption("Premium managed per agent with AI support | Target: 1:1,000 ratio")
        if not insights.human_leverage_ratio.empty:
            col1, col2 = st.columns([1, 1])
            with col1:
                leverage_filtered = _filter_useful_columns(insights.human_leverage_ratio.head(30).copy(), "human_leverage_ratio")
                leverage_semantic = _semantic_column_names(leverage_filtered, "human_leverage_ratio")
                
                # Calculate target value (median) for Premium per Agent
                if "Premium per Agent" in leverage_semantic.columns:
                    premium_values = pd.to_numeric(leverage_semantic["Premium per Agent"], errors='coerce')
                    target_value = premium_values.median()  # Use median as target
                    tolerance = 0.15  # 15% tolerance for "around target" (yellow zone)
                    
                    def color_premium_per_agent(val):
                        """Color code Premium per Agent based on target value"""
                        if pd.isna(val):
                            return ''
                        try:
                            val_float = float(val) if isinstance(val, str) else val
                            # Green: above target
                            if val_float > target_value * (1 + tolerance):
                                return 'background-color: #10b981; color: white; font-weight: bold'
                            # Yellow: around target (within tolerance)
                            elif val_float >= target_value * (1 - tolerance):
                                return 'background-color: #f59e0b; color: white; font-weight: bold'
                            # Red: below target
                            else:
                                return 'background-color: #ef4444; color: white; font-weight: bold'
                        except (ValueError, TypeError):
                            return ''
                    
                    # Apply color coding
                    styled_df = leverage_semantic.style.applymap(
                        color_premium_per_agent,
                        subset=["Premium per Agent"]
                    )
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(leverage_semantic, use_container_width=True, hide_index=True)
            with col2:
                # Use semantic column names for chart
                leverage_chart = insights.human_leverage_ratio.head(20).copy()
                leverage_chart = leverage_chart.rename(columns={
                    "agent_id": "Agent ID",
                    "leverage_ratio": "Leverage Ratio"
                })
                fig_leverage = px.bar(
                    leverage_chart,
                    x="Agent ID",
                    y="Leverage Ratio",
                    color="Leverage Ratio",
                    color_continuous_scale="Purples",
                    title="Agent Productivity Index",
                    labels={"Agent ID": "Agent ID", "Leverage Ratio": "Leverage Ratio"}
                )
                _update_chart_layout(fig_leverage, height=400, showlegend=False)
                st.plotly_chart(fig_leverage, use_container_width=True)
        else:
            st.info("Human leverage ratio data not available.")

    with tab4:
        st.markdown("### 🎯 Vertical Opportunity")
        if not insights.vertical_opportunity_score.empty:
            # Table at top
            opp_filtered = _filter_useful_columns(insights.vertical_opportunity_score.copy(), "vertical_opportunity_score")
            st.dataframe(_semantic_column_names(opp_filtered, "vertical_opportunity_score"), use_container_width=True, hide_index=True)
            
            # Two graphs side by side below the table
            col1, col2 = st.columns([1, 1])
            with col1:
                # Top Vertical Opportunities graph
                opp_chart = insights.vertical_opportunity_score.head(12).copy()
                opp_chart = opp_chart.rename(columns={
                    "vertical": "Vertical",
                    "opportunity_score": "Opportunity Score"
                })
                # Format vertical names in chart
                if "Vertical" in opp_chart.columns:
                    opp_chart["Vertical"] = opp_chart["Vertical"].apply(_format_vertical_name)
                fig_opp = px.bar(
                    opp_chart,
                    x="Vertical",
                    y="Opportunity Score",
                    color="Opportunity Score",
                    color_continuous_scale="Greens",
                    title="Top Vertical Opportunities",
                    labels={"Vertical": "Vertical", "Opportunity Score": "Opportunity Score"}
                )
                _update_chart_layout(fig_opp, height=400, showlegend=False)
                st.plotly_chart(fig_opp, use_container_width=True)
            
            with col2:
                # Top Verticals by Expected Profit per Lead graph
                v = insights.vertical_perf.copy()
                v_chart = v.sort_values("egp24_per_lead", ascending=False).head(12).copy()
                v_chart = v_chart.rename(columns={
                    "vertical": "Vertical",
                    "egp24_per_lead": "Expected Profit per Lead (24-mo)"
                })
                # Format vertical names in chart
                if "Vertical" in v_chart.columns:
                    v_chart["Vertical"] = v_chart["Vertical"].apply(_format_vertical_name)
                fig = px.bar(
                    v_chart,
                    x="Vertical",
                    y="Expected Profit per Lead (24-mo)",
                    color="Expected Profit per Lead (24-mo)",
                    color_continuous_scale="Blues",
                    title="Top Verticals by Expected Profit per Lead (24-mo)",
                    labels={"Vertical": "Vertical", "Expected Profit per Lead (24-mo)": "Expected Profit per Lead (24-mo)"}
                )
                _update_chart_layout(fig, height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Vertical opportunity score data not available.")
        
        st.divider()
        
        st.divider()
        
        st.markdown("### 🎯 Carrier Hits")
        st.caption("Which carrier_tier binds most frequently | Optimizes Market-Making Engine")
        if not insights.carrier_hit_ratio.empty:
            col1, col2 = st.columns([1, 1])
            with col1:
                carrier_filtered = _filter_useful_columns(insights.carrier_hit_ratio.copy(), "carrier_hit_ratio")
                st.dataframe(_semantic_column_names(carrier_filtered, "carrier_hit_ratio"), use_container_width=True, hide_index=True)
            with col2:
                # Use semantic column names for chart
                carrier_chart = insights.carrier_hit_ratio.head(15).copy()
                carrier_chart = carrier_chart.rename(columns={
                    "carrier_tier": "Carrier Tier",
                    "avg_hit_ratio": "Avg Hit Ratio",
                    "policy_type": "Policy Type"
                })
                fig_carrier = px.bar(
                    carrier_chart,
                    x="Carrier Tier",
                    y="Avg Hit Ratio",
                    color="Policy Type",
                    title="Carrier Hits by Tier",
                    labels={"Carrier Tier": "Carrier Tier", "Avg Hit Ratio": "Avg Hit Ratio"}
                )
                _update_chart_layout(fig_carrier, height=400, showlegend=True)
                st.plotly_chart(fig_carrier, use_container_width=True)
        else:
            st.info("Carrier hit ratio data not available.")
        
        st.divider()
        
        st.markdown("### 🗺️ State Performance")
        s = insights.state_perf.copy()
        s_filtered = _filter_useful_columns(s, "state_perf")
        st.dataframe(_semantic_column_names(s_filtered, "state_perf"), use_container_width=True, hide_index=True)

    with tab5:
        st.markdown("### ⚠️ Churn Risk Analysis (Predictive)")
        st.caption("Flags customers likely to defect | Enables 90-day renewal alerts")
        if not insights.churn_risk_analysis.empty:
            col1, col2 = st.columns([1, 1])
            with col1:
                churn_filtered = _filter_useful_columns(insights.churn_risk_analysis.copy(), "churn_risk_analysis")
                st.dataframe(_semantic_column_names(churn_filtered, "churn_risk_analysis"), use_container_width=True, hide_index=True)
            with col2:
                # Use semantic column names for chart
                churn_chart = insights.churn_risk_analysis.copy()
                churn_chart = churn_chart.rename(columns={
                    "churn_bucket": "Churn Risk",
                    "total_premium_at_risk": "Premium at Risk ($)"
                })
                fig_churn = px.bar(
                    churn_chart,
                    x="Churn Risk",
                    y="Premium at Risk ($)",
                    color="Churn Risk",
                    color_discrete_map={
                        "Low": "#10b981",
                        "Medium": "#f59e0b",
                        "High": "#ef4444",
                        "Critical": "#991b1b"
                    },
                    title="Premium at Risk by Churn Bucket",
                    labels={"Churn Risk": "Churn Risk", "Premium at Risk ($)": "Premium at Risk ($)"}
                )
                _update_chart_layout(fig_churn, height=400, showlegend=False)
                st.plotly_chart(fig_churn, use_container_width=True)
            
            high_risk = insights.churn_risk_analysis[insights.churn_risk_analysis["churn_bucket"].isin(["High", "Critical"])]
            if not high_risk.empty:
                total_at_risk = high_risk["total_premium_at_risk"].sum()
                st.warning(f"⚠️ **${total_at_risk:,.0f}** in premium at risk from High/Critical churn customers")
        else:
            st.info("Churn risk analysis data not available.")
        
        st.divider()
        
        st.markdown("### 🌍 Market Pricing Context")
        st.caption("External trends integration | Identify Harper's rate advantages")
        market_context = pd.DataFrame([
            {"segment": "Umbrella Coverage", "trend": "+11.5%", "harper_advantage": "Find cheaper rates"},
            {"segment": "E&S Construction", "trend": "+8.2%", "harper_advantage": "Better carrier access"},
            {"segment": "Tech SaaS", "trend": "-2.1%", "harper_advantage": "Competitive pricing"},
        ])
        market_filtered = _filter_useful_columns(market_context, "market_context")
        st.dataframe(_semantic_column_names(market_filtered, "market_context"), use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.markdown("### 📈 Daily Trends")
        tr = insights.trends_daily.copy()
        col1, col2 = st.columns(2)
        with col1:
            # Use semantic column names for chart
            tr_chart = tr.copy()
            tr_chart = tr_chart.rename(columns={
                "date": "Date",
                "leads": "Leads",
                "bound": "Bound"
            })
            fig = px.line(
                tr_chart,
                x="Date",
                y=["Leads", "Bound"],
                color_discrete_map={"Leads": "#3b82f6", "Bound": "#10b981"},
                title="Leads and Bound Over Time",
                labels={"Date": "Date", "value": "Count"}
            )
            _update_chart_layout(fig, height=350, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Use semantic column names for chart
            tr2_chart = tr.copy()
            tr2_chart = tr2_chart.rename(columns={
                "date": "Date",
                "spend_usd": "Spend ($)",
                "premium": "Premium"
            })
            fig2 = px.line(
                tr2_chart,
                x="Date",
                y=["Spend ($)", "Premium"],
                color_discrete_map={"Spend ($)": "#ef4444", "Premium": "#10b981"},
                title="Spend and Premium Over Time",
                labels={"Date": "Date", "value": "Amount ($)"}
            )
            _update_chart_layout(fig2, height=350, showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)
        


def _render_copilot_chat_panel() -> None:
    """Render the copilot chat panel in a right column when open."""
    if not st.session_state.get("copilot_sidebar_open", False):
        return
    
    with st.container():
        st.markdown('<div class="copilot-container">', unsafe_allow_html=True)
        
        # Header with close button
        col_header, col_close = st.columns([5, 1])
        with col_header:
            st.markdown("### 🤖 GTM Co‑Pilot Chat")
        with col_close:
            if st.button("✕", key="closeCopilotBtn", help="Close chat", use_container_width=True):
                st.session_state.copilot_sidebar_open = False
                st.rerun()
        
        st.caption("Ask questions about your GTM data and get actionable insights.")
        st.divider()
        
        # Render messages
        if "messages" in st.session_state and st.session_state.messages:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
        else:
            st.info("💡 Start a conversation by asking a question in the input below.")
        
        # Show spinner if processing
        if st.session_state.get("processing_query", False):
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    st.markdown("")
        
        st.markdown('</div>', unsafe_allow_html=True)


def _get_copilot_client() -> tuple[Optional[OpenAI], str, float]:
    """Get copilot client with API key from session state, environment, Streamlit secrets, or fallback demo key."""
    # Try session state first (user-entered key from UI)
    api_key = st.session_state.get("user_openai_api_key", None)
    if api_key and api_key.strip():  # Only use if non-empty
        api_key = api_key.strip()
    else:
        api_key = None
    
    # Then try environment variable
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            api_key = api_key.strip()
    
    # Then try Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
            if api_key:
                api_key = api_key.strip()
        except:
            api_key = None
    
    # Fallback: If no key found, return None (user must provide key via Streamlit secrets or UI)
    # For Streamlit Cloud deployment, set OPENAI_API_KEY in Streamlit Cloud secrets
    if not api_key:
        api_key = None
    
    model = "gpt-4o-mini"
    temperature = 0.2
    
    client = _get_openai_client(api_key)
    return client, model, temperature




def _handle_copilot_input(raw, insights) -> None:
    """Handle chat input at bottom and generate response. Opens right sidebar when question is sent."""
    
    # Initialize session state
    if "copilot_sidebar_open" not in st.session_state:
        st.session_state.copilot_sidebar_open = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_query" not in st.session_state:
        st.session_state.processing_query = False
    
    # Get client from session state (API key is embedded)
    if "copilot_client" not in st.session_state or st.session_state.copilot_client is None:
        return
    
    # Chat input at bottom
    user_msg = st.chat_input("Ask the GTM copilot…")
    if not user_msg:
        return

    # Open sidebar when user sends a message
    if not st.session_state.copilot_sidebar_open:
        st.session_state.copilot_sidebar_open = True
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_msg})
    st.session_state.processing_query = True
    
    # Generate response (without displaying in main area)
    system_prompt = build_system_prompt(insights)
    client = st.session_state.copilot_client
    model = st.session_state.copilot_model
    temperature = st.session_state.copilot_temperature

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                *st.session_state.messages[-12:],
            ],
            max_tokens=800,  # Encourage conciseness
        )
        answer = resp.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Invalid_api_key" in error_msg or "Incorrect API key" in error_msg:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⚠️ **API Key Error**: Your OpenAI API key is invalid or expired. Please check your API key."
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ **Error**: {error_msg}"
            })
    
    # Mark processing as complete and rerun to show response in sidebar
    st.session_state.processing_query = False
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "Harper Growth Intelligence OS - Operational dashboard for AI-native brokerage optimization"
        }
    )

    root_dir = str(Path(__file__).parent)
    cache_key = _get_data_cache_key(root_dir)
    raw = _load_all(root_dir, cache_key=cache_key)

    # Compute filtered insights once for both dashboard and copilot
    leads = raw["data"].leads.copy()
    f_leads, filter_hash = _filter_leads(leads)
    
    # Filter related data based on filtered leads
    f_lead_ids = set(f_leads["lead_id"].unique())
    f_touchpoints = raw["data"].touchpoints[raw["data"].touchpoints["lead_id"].isin(f_lead_ids)].copy()
    f_policies = raw["data"].policies[raw["data"].policies["lead_id"].isin(f_lead_ids)].copy()
    
    # Filter ad_spend by date range from filtered leads
    if len(f_leads) > 0:
        min_date = pd.to_datetime(f_leads["lead_created_at"]).min()
        max_date = pd.to_datetime(f_leads["lead_created_at"]).max()
        f_ad_spend = raw["data"].ad_spend_daily[
            (pd.to_datetime(raw["data"].ad_spend_daily["date"]) >= min_date) &
            (pd.to_datetime(raw["data"].ad_spend_daily["date"]) <= max_date)
        ].copy()
    else:
        f_ad_spend = raw["data"].ad_spend_daily.copy()
    
    # Compute insights from filtered data (shared between dashboard and copilot)
    insights = _compute_filtered_insights(f_leads, f_ad_spend, raw["data"].agents, f_touchpoints, f_policies, filter_hash)
    
    # Update raw dict to include filtered data for dashboard
    raw_with_filtered = {"data": raw["data"], "insights": insights, "f_leads": f_leads}
    
    # Get copilot client with embedded API key
    client, model, temperature = _get_copilot_client()
    
    # Initialize copilot session state
    if client is not None:
        if "copilot_client" not in st.session_state:
            st.session_state.copilot_client = client
        if "copilot_model" not in st.session_state:
            st.session_state.copilot_model = model
        if "copilot_temperature" not in st.session_state:
            st.session_state.copilot_temperature = temperature
        if "copilot_insights" not in st.session_state:
            st.session_state.copilot_insights = insights
        
        # Update if changed
        st.session_state.copilot_client = client
        st.session_state.copilot_model = model
        st.session_state.copilot_temperature = temperature
        st.session_state.copilot_insights = insights
    
    # Render dashboard with conditional layout
    if st.session_state.get("copilot_sidebar_open", False) and client is not None:
        # Split layout when chat is open
        col_dashboard, col_copilot = st.columns([2, 1])
        with col_dashboard:
            _render_dashboard(raw_with_filtered)
        with col_copilot:
            _render_copilot_chat_panel()
    else:
        # Full width dashboard when chat is closed
        _render_dashboard(raw_with_filtered)
    
    # Chat input at bottom - must be at top level
    _handle_copilot_input(raw, insights)


if __name__ == "__main__":
    main()

