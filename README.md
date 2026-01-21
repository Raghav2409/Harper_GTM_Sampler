# Harper GTM Co-Pilot Dashboard

A comprehensive **Streamlit-based GTM (Go-to-Market) analytics dashboard** with an integrated AI copilot, designed for AI-native commercial insurance brokerages like Harper. This dashboard transforms raw GTM data into actionable insights, focusing on **operational leverage** and **market-making efficiency**.

## 🚀 Features

### 📊 Real-Time Dashboard
- **7 Key Performance Indicators (KPIs)** in the header:
  - Leads, Bound, Automation Rate, LTV:CAC, Net Revenue, Spend, Premium
- **5 Comprehensive Tabs** with detailed analytics:
  - **📊 Overview**: Conversion funnel, Golden Window Leakage, STQ Ratio, Lead Score Intensity
  - **💰 Paid Efficiency**: Vertical opportunities, Lead quality by channel, Paid efficiency metrics, Top campaigns
  - **⚡ Speed-to-Lead**: AI vs Human performance, Speed-to-lead impact, Operational drilldown
  - **🎯 Segments**: Carrier hit ratio, Human leverage ratio, Vertical & State performance
  - **📈 Trends & Attribution**: Churn risk analysis, Market pricing context, Daily trends, Attribution snapshot

### 🤖 AI GTM Co-Pilot
- **Intelligent Chat Interface**: Ask questions about your GTM data and get actionable insights
- **Context-Aware**: Grounded with real-time metrics, funnel data, and performance tables
- **Natural Language**: Conversational responses like an informed GTM expert
- **Right-Sidebar Panel**: Opens automatically when you ask a question

### 🎨 Modern UI/UX
- **Beautiful Gradients**: Modern color schemes and visual design
- **Semantic Labels**: User-friendly column names instead of technical database names
- **Responsive Layout**: Adaptive columns and sidebar panels
- **Interactive Charts**: 12+ Plotly visualizations with centered titles and legends
- **Filtering**: Sidebar filters for date range, channels, verticals, states, and E&S eligibility

## 📁 Dataset Overview

This dataset is **fully synthetic** and designed to mimic a high-growth commercial insurance GTM funnel like Harper's:
**paid acquisition → lead → contact/quote → bind → premium + retention**,
with multi-touch journeys across **Google / Meta / TikTok + Direct/Organic/Partner/Referral**, and sales touches across **Email + Phone**.

### Files
- `leads.csv` — 1 row per lead (SMB), including features + funnel outcomes
- `touchpoints.csv` — multi-touch event stream (marketing + sales + conversion)
- `ad_spend_daily.csv` — daily spend, impressions, clicks by channel/campaign
- `policies.csv` — bound policies (1–3 per bound lead) with premium split + E&S tier
- `agents.csv` — agent roster with quality + speed multipliers

### Key IDs
- `lead_id` — primary key joining all files
- `campaign_id` — joins touchpoints to `ad_spend_daily.csv` for paid analysis
- `agent_id` — joins leads to `agents.csv` for productivity / routing analysis

### Core Columns (leads.csv)
- **Acquisition**: `source_channel`, `source_campaign_id`, `origin_channel`, `device`, `landing_page_variant`
- **Business features**: `vertical`, `state`, `employees`, `annual_revenue_usd`, `years_in_business`, `prior_claims_3y`, `risk_score`, `is_es_eligible`
- **Funnel outcomes**: `is_contacted`, `is_quoted`, `is_bound`, timestamps, `time_to_first_contact_min`, `time_to_quote_hours`, `time_to_bind_days`
- **Value**: `bound_premium_annual_usd`, `bound_policy_count`, `commission_rate`, `retention_prob_12m`, `retention_prob_24m`, `expected_gross_profit_24m_usd`
- **Benchmark helper**: `p_bind_synthetic` (the true synthetic bind probability used to generate outcomes)

### Core Columns (touchpoints.csv)
- `event_time`, `event_type`, `event_stage` (marketing / sales / conversion)
- `channel`, `campaign_id`, `adset_id`, `creative_id`
- Phone metadata: `call_outcome`, `call_duration_sec`
- Email metadata: `email_template`

### Built-in Synthetic "Real-World" Patterns
- **Attribution bias**: many conversions have TOF touches (Meta/TikTok) but last-touch skews to Search/Direct.
- **Speed-to-lead effect**: faster `time_to_first_contact_min` correlates strongly with bind rate.
- **Vertical economics**: higher-premium verticals (Construction/Trucking/Manufacturing) tend to be harder to bind.
- **E&S complexity**: `is_es_eligible=1` increases premium but reduces bind probability and increases time-to-quote/bind.
- **Bundling**: larger businesses more likely to bind 2–3 policies and have higher retention.

### Dataset Counts (v1)
- **Leads**: 102,174
- **Touchpoints**: 723,995
- **Bound leads**: 14,330
- **Policies**: 20,727

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Raghav2409/Harper_GTM_Sampler.git
   cd Harper_GTM_Sampler
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API Key** (for Co-Pilot functionality)
   
   Option 1: Environment variable
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Option 2: Streamlit secrets (recommended)
   Create `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

   The dashboard will open in your browser at `http://localhost:8501`

## 📊 Dashboard Sections

### 📊 Overview Tab
- **Conversion Funnel**: Track lead progression with conversion lift insights
- **Quick Wins**: Top 3 actionable recommendations
- **Golden Window Leakage**: Revenue lost from leads contacted after 5 minutes
- **STQ Ratio by Vertical**: Submission-to-Quote ratio to identify carrier bottlenecks
- **Lead Score Intensity Map**: Intent vs Risk score distribution to find high-value opportunities

### 💰 Paid Efficiency Tab
- **Vertical Opportunity Score**: Rank verticals by opportunity (intent + premium)
- **Lead Quality by Channel**: ROI + Risk Score analysis to prevent funnel clogging
- **Paid Efficiency by Channel**: CPL vs ROI tradeoffs with channel comparison
- **Top Campaigns**: Top 25 campaigns ranked by spend

### ⚡ Speed-to-Lead Tab
- **AI Voice vs Human Performance**: Validates ROI of internal 'AI Grid' infrastructure
- **Speed-to-Lead Impact**: Bind rate by time-to-contact buckets
- **Operational Drilldown**: Filtered leads analysis by time buckets

### 🎯 Segments Tab
- **Carrier Hit Ratio**: Which carrier tier binds most frequently (optimizes Market-Making Engine)
- **Human Leverage Ratio**: Premium managed per agent with AI support (Target: 1:1,000 ratio)
- **Vertical Performance**: Performance metrics by vertical
- **State Performance**: Performance metrics by state

### 📈 Trends & Attribution Tab
- **Churn Risk Analysis**: Predictive churn flags enabling 90-day renewal alerts
- **Market Pricing Context**: External trends integration to identify Harper's rate advantages
- **Daily Trends**: Leads/Bound and Spend/Premium over time
- **Attribution Snapshot**: First-touch vs Last-touch comparison

## 🤖 GTM Co-Pilot Usage

The AI Co-Pilot is accessible via the chat input at the bottom of the dashboard:

1. **Ask a question** in the "Ask the GTM copilot…" input box
2. **Sidebar opens automatically** with your question and the AI response
3. **Get insights** about:
   - Performance metrics and trends
   - Channel efficiency comparisons
   - Vertical opportunities
   - Speed-to-lead impact
   - Churn risk analysis
   - And more!

**Example Questions:**
- "What's our bind rate by vertical?"
- "Which channel has the best ROI?"
- "How does speed-to-contact affect conversion?"
- "What's the churn risk for high-value customers?"

## 🎨 UI Features

- **Semantic Column Names**: All tables display user-friendly labels
- **Filtered Columns**: Only essential columns shown in tables
- **Centered Chart Labels**: All chart titles and legends are centered
- **Responsive Design**: Adapts to different screen sizes
- **Sidebar Toggles**: Easy access to filters and copilot chat
- **Modern Aesthetics**: Gradient backgrounds, clean typography, smooth animations

## 📝 Key Metrics Explained

- **STQ Ratio**: Submission-to-Quote ratio (lower = faster carrier response)
- **LTV:CAC**: Lifetime Value to Customer Acquisition Cost ratio (target: ≥3:1)
- **EGP24**: Expected Gross Profit over 24 months
- **Automation Rate**: Percentage of leads handled by AI
- **Human Leverage Ratio**: Premium managed per agent (target: 1:1,000)

## 🔧 Configuration

### Filters
The left sidebar provides filters for:
- **Date Range**: Filter leads by creation date
- **Source Channel**: Filter by marketing channel
- **Vertical**: Filter by business vertical
- **State**: Filter by geographic location
- **E&S Eligible**: Filter by Excess & Surplus eligibility

### Co-Pilot Settings
- **Model**: gpt-4o-mini (default) or gpt-4o
- **Temperature**: 0.2 (default) for consistent, data-driven responses
- **API Key**: Set via environment variable or Streamlit secrets

## 📚 Project Structure

```
Harper_GTM_Sampler/
├── app.py                      # Main Streamlit application
├── gtm/
│   ├── __init__.py
│   ├── data.py                 # Data loading functions
│   ├── metrics.py              # GTM metrics computation
│   └── copilot.py              # AI copilot system prompt
├── leads.csv                   # Lead data
├── touchpoints.csv             # Touchpoint events
├── ad_spend_daily.csv          # Daily ad spend
├── policies.csv                # Policy data
├── agents.csv                  # Agent roster
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── README_STREAMLIT.md         # Streamlit-specific documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for use.

## 🙏 Acknowledgments

Designed for AI-native commercial insurance brokerages, inspired by Harper's GTM operations.
