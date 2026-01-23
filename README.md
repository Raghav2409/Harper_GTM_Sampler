# Harper Growth Intelligence OS

A Streamlit-based GTM analytics dashboard with integrated AI copilot for commercial insurance brokerages. Transforms raw GTM data into actionable insights focused on operational leverage and market-making efficiency.

## Features

### Dashboard Analytics
- **7 Key Performance Indicators**: Leads, Bound, Automation Rate, LTV:CAC, Net Revenue, Spend, Premium
- **5 Analytics Tabs**: Overview, Paid Efficiency, Speed-to-Lead, Target Segments, Performance Drivers
- Interactive visualizations with semantic column labels and dynamic filtering

### AI GTM Co-Pilot
- Context-aware chat interface grounded with real-time metrics
- Natural language responses for GTM insights and recommendations
- Right-sidebar panel that opens automatically on query

## Dataset

Synthetic dataset mimicking a high-growth commercial insurance GTM funnel: **paid acquisition → lead → contact/quote → bind → premium + retention**, with multi-touch journeys across Google/Meta/TikTok and Direct/Organic/Partner/Referral channels.

### Files
- `leads.csv` — Lead records with features and funnel outcomes
- `touchpoints.csv` — Multi-touch event stream (marketing + sales + conversion)
- `ad_spend_daily.csv` — Daily spend, impressions, clicks by channel/campaign
- `policies.csv` — Bound policies with premium and E&S tier
- `agents.csv` — Agent roster with quality and speed multipliers

**Key Identifiers**: `lead_id` (primary key), `campaign_id`, `agent_id`

**Dataset Size**: 102K leads, 724K touchpoints, 14K bound leads, 21K policies

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
   
   **Option 1**: Environment variable
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   **Option 2**: Streamlit secrets (recommended)
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

Use sidebar filters to filter by date range, channels, verticals, states, and E&S eligibility. All metrics update dynamically.

**GTM Co-Pilot**: Enter questions in the chat input at the bottom. The copilot sidebar opens automatically with responses grounded in current dashboard data.

**Example queries**:
- "What's our bind rate by vertical?"
- "Which channel has the best ROI?"
- "How does speed-to-contact affect conversion?"

## Key Metrics

- **STQ Ratio**: Submission-to-Quote ratio (lower = faster carrier response)
- **LTV:CAC**: Lifetime Value to Customer Acquisition Cost (target: ≥3:1)
- **EGP24**: Expected Gross Profit over 24 months
- **Automation Rate**: Percentage of leads handled by AI
- **Human Leverage Ratio**: Premium managed per agent (target: 1:1,000)

## Project Structure

```
Harper_GTM_Sampler/
├── app.py                      # Main Streamlit application
├── gtm/
│   ├── data.py                 # Data loading functions
│   ├── metrics.py              # GTM metrics computation
│   └── copilot.py              # AI copilot system prompt
├── *.csv                       # Data files (leads, touchpoints, ad_spend, policies, agents)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## License

Open source and available for use.

## Acknowledgments

Designed for AI-native commercial insurance brokerages, inspired by Harper's GTM operations.
