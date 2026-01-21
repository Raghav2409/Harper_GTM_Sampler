# Harper-like Synthetic GTM Dataset (v1)

This dataset is **fully synthetic** and designed to mimic a high-growth commercial insurance GTM funnel like Harper's:
**paid acquisition → lead → contact/quote → bind → premium + retention**,
with multi-touch journeys across **Google / Meta / TikTok + Direct/Organic/Partner/Referral**, and sales touches across **Email + Phone**.

## Files
- `leads.csv` — 1 row per lead (SMB), including features + funnel outcomes
- `touchpoints.csv` — multi-touch event stream (marketing + sales + conversion)
- `ad_spend_daily.csv` — daily spend, impressions, clicks by channel/campaign
- `policies.csv` — bound policies (1–3 per bound lead) with premium split + E&S tier
- `agents.csv` — agent roster with quality + speed multipliers

## Key IDs
- `lead_id` — primary key joining all files
- `campaign_id` — joins touchpoints to `ad_spend_daily.csv` for paid analysis
- `agent_id` — joins leads to `agents.csv` for productivity / routing analysis

## Core columns (leads.csv)
- Acquisition: `source_channel`, `source_campaign_id`, `origin_channel`, `device`, `landing_page_variant`
- Business features: `vertical`, `state`, `employees`, `annual_revenue_usd`, `years_in_business`, `prior_claims_3y`, `risk_score`, `is_es_eligible`
- Funnel outcomes: `is_contacted`, `is_quoted`, `is_bound`, timestamps, `time_to_first_contact_min`, `time_to_quote_hours`, `time_to_bind_days`
- Value: `bound_premium_annual_usd`, `bound_policy_count`, `commission_rate`, `retention_prob_12m`, `retention_prob_24m`, `expected_gross_profit_24m_usd`
- Benchmark helper: `p_bind_synthetic` (the true synthetic bind probability used to generate outcomes)

## Core columns (touchpoints.csv)
- `event_time`, `event_type`, `event_stage` (marketing / sales / conversion)
- `channel`, `campaign_id`, `adset_id`, `creative_id`
- Phone metadata: `call_outcome`, `call_duration_sec`
- Email metadata: `email_template`

## Built-in synthetic "real-world" patterns
- **Attribution bias**: many conversions have TOF touches (Meta/TikTok) but last-touch skews to Search/Direct.
- **Speed-to-lead effect**: faster `time_to_first_contact_min` correlates strongly with bind rate.
- **Vertical economics**: higher-premium verticals (Construction/Trucking/Manufacturing) tend to be harder to bind.
- **E&S complexity**: `is_es_eligible=1` increases premium but reduces bind probability and increases time-to-quote/bind.
- **Bundling**: larger businesses more likely to bind 2–3 policies and have higher retention.

## Suggested MVP analyses
1) **Paid channel efficiency**: cost per lead, cost per bound, cost per $1k premium by channel/campaign.
2) **Multi-touch attribution**: compare last-touch vs Markov/Time-decay using `touchpoints.csv` (filter `event_stage='marketing'`).
3) **Lead scoring**: predict `is_bound` from `leads.csv` features; produce decile lift and value concentration curves.
4) **Routing simulation**: compare bind rate if top decile leads get fast follow-up (use `time_to_first_contact_min` bucket analysis).
5) **EGPL**: compute `expected_value = P(bind)*E(premium)*commission*(1+ret12)` and use it for value-based bidding.

## Counts (v1)
- Leads: 102,174
- Touchpoints: 723,995
- Bound leads: 14,330
- Policies: 20,727
