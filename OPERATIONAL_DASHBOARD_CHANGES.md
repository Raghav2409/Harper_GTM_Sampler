# Operational Dashboard Transformation Summary

## Overview
Transformed the Harper GTM dashboard from a "reporting tool" to an "operating system" focused on operational leverage and market-making efficiency for an AI-native brokerage.

## New Data Columns Added

### `leads.csv` (7 new columns)
- `automation_rate`: % of tasks handled by AI (0-1)
- `ai_contact_type`: "ai_voice", "human", "ai_email", "mixed"
- `stq_ratio`: Submission-to-Quote ratio (carrier response rate)
- `cac_usd`: Customer Acquisition Cost per lead
- `ltv_usd`: Lifetime Value per lead
- `churn_risk_score`: Predictive churn risk (0-1)
- `net_commission_revenue_usd`: Net commission revenue (premium × commission_rate)

### `touchpoints.csv` (1 new column)
- `is_ai_handled`: Boolean flag for AI vs human touchpoints

### `policies.csv` (1 new column)
- `carrier_hit_ratio`: Carrier acceptance rate (0-1)

## Header Metrics Changes

**Before:** Leads, Bound, Spend, Premium, EGP(24m)

**After:** 
- Leads
- Bound (with bind rate %)
- **Automation Rate %** (NEW)
- **CAC:LTV Ratio** (NEW - shows ✅ if ≥3:1)
- **Net Commission Revenue** (NEW - replaces EGP)
- Spend
- Premium (annual)

## Tab-by-Tab Changes

### Tab 1: Overview → "Operational Flywheel"

**Added:**
- **Submission-to-Quote (STQ) Ratio by Vertical**: Identifies carrier friction points
- **Lead Score Intensity Map**: Intent vs Risk score distribution by vertical (scatter plot)
- **Conversion Lift**: Added to funnel table (shows lift between stages)

**Removed:**
- Generic funnel counts (replaced with conversion lift focus)

### Tab 2: Paid Efficiency → "Micro-Segment Targeting"

**Added:**
- **Vertical Opportunity Score**: Find + Flood micro-segments ranking (vertical + intent + premium)
- **Lead Quality by Channel**: Risk score + ROI analysis (prevents funnel clogging)

**Kept:**
- Paid efficiency by channel
- Top campaigns

### Tab 3: Speed-to-Lead → "Human-AI Collaboration"

**Added:**
- **AI Voice vs Human Performance**: Contact rate, quoted rate, bound rate comparison
- **Golden Window Leakage**: Revenue lost from leads contacted after 5 minutes (burning platform metric)

**Kept:**
- Speed-to-lead lift buckets
- Operational drilldown

### Tab 4: Segments → "Carrier Intelligence"

**Added:**
- **Carrier Hit Ratio**: Which carrier_tier binds most frequently for which vertical/policy type
- **Human Leverage Ratio**: Premium managed per agent with AI support (target 1:1,000)

**Kept:**
- Vertical performance
- State performance

### Tab 5: Attribution & Trends → "Predictive Insights"

**Added:**
- **Churn Risk Analysis**: Predictive churn buckets (Low/Medium/High/Critical) with premium at risk
- **Market Pricing Context**: External trends integration (synthetic data showing market rate changes)

**Kept:**
- Attribution snapshot
- Daily trends

## New Metrics Computation Functions

All new functions in `gtm/metrics.py`:

1. `compute_stq_by_vertical()`: STQ ratio by vertical
2. `compute_lead_score_intensity()`: Intent + Risk distribution
3. `compute_vertical_opportunity_score()`: Opportunity ranking
4. `compute_lead_quality_by_channel()`: Risk + ROI analysis
5. `compute_ai_vs_human_perf()`: AI vs Human comparison
6. `compute_golden_window_leakage()`: Revenue leakage calculation
7. `compute_carrier_hit_ratio()`: Carrier acceptance rates
8. `compute_human_leverage_ratio()`: Agent leverage metrics
9. `compute_churn_risk_analysis()`: Predictive churn analysis

## Copilot Updates

The GTM co-pilot chatbot now has access to all new operational metrics:
- STQ by vertical
- Lead score intensity
- Vertical opportunity scores
- Lead quality by channel
- AI vs Human performance
- Golden window leakage
- Carrier hit ratios
- Human leverage ratios
- Churn risk analysis

## Strategic Value

| Component | GTM Strategic Value |
|-----------|---------------------|
| **Automation Rate %** | Tracks progress toward "fully autonomous brokerage" goal |
| **STQ Ratio** | Identifies friction points with external carriers |
| **Lead Risk Score by Channel** | Prevents funnel clogging with uninsurable leads |
| **AI Voice vs Human Pods** | Validates ROI of internal "AI Grid" infrastructure |
| **Carrier Hit Ratio** | Optimizes carrier matching, Harper's core technical moat |
| **Churn Prediction** | Increases LTV by automating 90-day renewal alerts |
| **Golden Window Leakage** | Creates "burning platform" for engineering to ship automation |
| **Human Leverage Ratio** | Measures operational leverage (target 1:1,000) |

## Files Modified

1. `enrich_data_for_ops.py` - Script to add new columns (run once)
2. `gtm/metrics.py` - Added 9 new computation functions + updated InsightsPack
3. `app.py` - Updated all tabs + header metrics
4. `gtm/copilot.py` - Updated context to include new metrics
5. `leads.csv`, `touchpoints.csv`, `policies.csv` - Enriched with new columns

## Backup Files Created

- `leads.csv.backup`
- `touchpoints.csv.backup`
- `policies.csv.backup`

## Next Steps

1. Run `streamlit run app.py` to see the new operational dashboard
2. The dashboard now focuses on "what to do next" rather than "what happened"
3. All metrics are designed for Growth Marketers, Growth Engineers, and FDEs to "spec the fix and ship it same-day"
