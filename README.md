# Harper GTM Co-Pilot Dashboard

A Streamlit-based GTM analytics dashboard with integrated AI copilot for AI-native commercial insurance brokerages. Transforms raw GTM data into actionable insights focused on operational leverage and market-making efficiency.

## Features

### Dashboard Analytics
- **7 Key Performance Indicators**: Leads, Bound, Automation Rate, LTV:CAC, Net Revenue, Spend, Premium
- **5 Analytics Tabs**:
  - **Overview**: Conversion funnel, Golden Window Leakage, STQ Ratio, Lead Score Intensity
  - **Paid Efficiency**: Vertical opportunities, Lead quality by channel, Paid efficiency metrics, Top campaigns
  - **Speed-to-Lead**: AI vs Human performance, Speed-to-lead impact, Operational drilldown
  - **Segments**: Carrier hit ratio, Human leverage ratio, Vertical & State performance
  - **Trends & Attribution**: Churn risk analysis, Market pricing context, Daily trends, Attribution snapshot

### AI GTM Co-Pilot
- Context-aware chat interface grounded with real-time metrics and performance data
- Natural language responses for GTM insights and recommendations
- Right-sidebar panel that opens automatically on query

### User Interface
- Semantic column labels for improved readability
- Interactive Plotly visualizations with centered titles
- Sidebar filters for date range, channels, verticals, states, and E&S eligibility
- Responsive layout with modern design

## Dataset

Synthetic dataset designed to mimic a high-growth commercial insurance GTM funnel: **paid acquisition → lead → contact/quote → bind → premium + retention**, with multi-touch journeys across Google/Meta/TikTok and Direct/Organic/Partner/Referral channels.

### Files
- `leads.csv` — Lead records with features and funnel outcomes
- `touchpoints.csv` — Multi-touch event stream (marketing + sales + conversion)
- `ad_spend_daily.csv` — Daily spend, impressions, clicks by channel/campaign
- `policies.csv` — Bound policies (1–3 per bound lead) with premium and E&S tier
- `agents.csv` — Agent roster with quality and speed multipliers

### Key Identifiers
- `lead_id` — Primary key joining all files
- `campaign_id` — Joins touchpoints to ad spend for paid analysis
- `agent_id` — Joins leads to agents for productivity analysis

### Core Columns (leads.csv)
- **Acquisition**: `source_channel`, `source_campaign_id`, `origin_channel`, `device`, `landing_page_variant`
- **Business features**: `vertical`, `state`, `employees`, `annual_revenue_usd`, `years_in_business`, `prior_claims_3y`, `risk_score`, `is_es_eligible`
- **Funnel outcomes**: `is_contacted`, `is_quoted`, `is_bound`, timestamps, `time_to_first_contact_min`, `time_to_quote_hours`, `time_to_bind_days`
- **Value**: `bound_premium_annual_usd`, `bound_policy_count`, `commission_rate`, `retention_prob_12m`, `retention_prob_24m`, `expected_gross_profit_24m_usd`

### Core Columns (touchpoints.csv)
- `event_time`, `event_type`, `event_stage` (marketing / sales / conversion)
- `channel`, `campaign_id`, `adset_id`, `creative_id`
- Phone metadata: `call_outcome`, `call_duration_sec`
- Email metadata: `email_template`

### Synthetic Patterns
- **Attribution bias**: TOF touches (Meta/TikTok) with last-touch skewing to Search/Direct
- **Speed-to-lead effect**: Faster `time_to_first_contact_min` correlates with bind rate
- **Vertical economics**: Higher-premium verticals (Construction/Trucking/Manufacturing) harder to bind
- **E&S complexity**: `is_es_eligible=1` increases premium but reduces bind probability
- **Bundling**: Larger businesses more likely to bind 2–3 policies with higher retention

### Dataset Counts
- Leads: 102,174
- Touchpoints: 723,995
- Bound leads: 14,330
- Policies: 20,727

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository
   ```bash
   git clone https://github.com/Raghav2409/Harper_GTM_Sampler.git
   cd Harper_GTM_Sampler
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Configure OpenAI API Key (for Co-Pilot functionality)
   
   Option 1: Environment variable
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Option 2: Streamlit secrets (recommended)
   Create `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

4. Run the dashboard
   ```bash
   streamlit run app.py
   ```
   
   Dashboard opens at `http://localhost:8501`

## Usage

### Dashboard Navigation
Use the sidebar filters to filter by date range, channels, verticals, states, and E&S eligibility. All metrics and visualizations update dynamically based on selected filters.

### GTM Co-Pilot
Enter questions in the chat input at the bottom of the dashboard. The copilot sidebar opens automatically with responses grounded in current dashboard data.

Example queries:
- "What's our bind rate by vertical?"
- "Which channel has the best ROI?"
- "How does speed-to-contact affect conversion?"

## Key Metrics

- **STQ Ratio**: Submission-to-Quote ratio (lower indicates faster carrier response)
- **LTV:CAC**: Lifetime Value to Customer Acquisition Cost ratio (target: ≥3:1)
- **EGP24**: Expected Gross Profit over 24 months
- **Automation Rate**: Percentage of leads handled by AI
- **Human Leverage Ratio**: Premium managed per agent (target: 1:1,000)

## Project Structure

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
└── README.md                   # This file
```

## Contributing

Contributions are welcome. Please submit a Pull Request.

## License

This project is open source and available for use.

## Acknowledgments

Designed for AI-native commercial insurance brokerages, inspired by Harper's GTM operations.
