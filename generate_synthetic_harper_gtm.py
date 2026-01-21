"""
generate_synthetic_harper_gtm.py

Creates a Harper-like synthetic GTM dataset for paid growth + multi-touch attribution + lead scoring.

Outputs:
- leads.csv
- touchpoints.csv
- ad_spend_daily.csv
- policies.csv
- agents.csv
- README.md

This is fully synthetic (no real customer data).
"""

import os
import zipfile
import numpy as np
import pandas as pd

def main(
    out_dir: str = "harper_synthetic_gtm_dataset_v1",
    start_date: str = "2025-10-01",
    end_date: str = "2025-12-31",
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    dates = pd.date_range(start_date, end_date, freq="D")

    channels_paid = [
        "Google_Search_NonBrand",
        "Google_Search_Brand",
        "Meta_Prospecting",
        "Meta_Retargeting",
        "TikTok_Prospecting",
        "TikTok_Retargeting",
        "Partner_Embedded",
    ]
    channels_unpaid = ["Direct", "Organic_Search", "Referral"]

    channel_cpl = {
        "Google_Search_NonBrand": 92.0,
        "Google_Search_Brand": 38.0,
        "Meta_Prospecting": 105.0,
        "Meta_Retargeting": 72.0,
        "TikTok_Prospecting": 112.0,
        "TikTok_Retargeting": 85.0,
        "Partner_Embedded": 28.0,
    }
    channel_cpc = {
        "Google_Search_NonBrand": 6.5,
        "Google_Search_Brand": 3.8,
        "Meta_Prospecting": 2.6,
        "Meta_Retargeting": 2.2,
        "TikTok_Prospecting": 1.9,
        "TikTok_Retargeting": 1.7,
        "Partner_Embedded": 0.0,
    }
    channel_ctr = {
        "Google_Search_NonBrand": 0.040,
        "Google_Search_Brand": 0.055,
        "Meta_Prospecting": 0.012,
        "Meta_Retargeting": 0.018,
        "TikTok_Prospecting": 0.010,
        "TikTok_Retargeting": 0.013,
        "Partner_Embedded": 0.0,
    }
    base_spend = {
        "Google_Search_NonBrand": 18000,
        "Google_Search_Brand": 6000,
        "Meta_Prospecting": 16000,
        "Meta_Retargeting": 7000,
        "TikTok_Prospecting": 9000,
        "TikTok_Retargeting": 4000,
        "Partner_Embedded": 2500,
    }

    verticals = [
        "Construction",
        "Restaurants",
        "Retail",
        "Professional_Services",
        "Technology_SaaS",
        "Healthcare",
        "Manufacturing",
        "Real_Estate",
        "Transportation_Trucking",
        "Cleaning_Janitorial",
    ]
    base_premium = {
        "Technology_SaaS": 1200,
        "Professional_Services": 1400,
        "Real_Estate": 1600,
        "Retail": 2200,
        "Cleaning_Janitorial": 2500,
        "Restaurants": 3200,
        "Healthcare": 4200,
        "Manufacturing": 5200,
        "Construction": 6200,
        "Transportation_Trucking": 7800,
    }
    vertical_bind_adj = {
        "Technology_SaaS": +0.25,
        "Professional_Services": +0.18,
        "Real_Estate": +0.10,
        "Retail": +0.05,
        "Cleaning_Janitorial": +0.02,
        "Restaurants": -0.02,
        "Healthcare": -0.08,
        "Manufacturing": -0.12,
        "Construction": -0.15,
        "Transportation_Trucking": -0.22,
    }

    states = ["CA","TX","FL","NY","IL","PA","OH","GA","NC","MI","NJ","VA","WA","AZ","MA","TN","IN","MO","MD","CO"]
    state_probs = np.array([0.12,0.11,0.09,0.08,0.05,0.05,0.05,0.05,0.05,0.04,0.04,0.04,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.02])
    state_probs = state_probs / state_probs.sum()

    devices = ["mobile","desktop"]
    device_probs = [0.58, 0.42]
    landing_variants = ["LP_A_short", "LP_B_long", "LP_C_verticalized"]
    landing_probs = [0.45, 0.35, 0.20]

    # campaigns
    campaigns = {}
    for ch in channels_paid:
        if ch in ["Google_Search_NonBrand","Meta_Prospecting","TikTok_Prospecting"]:
            campaigns[ch] = [f"{ch}_Generic"] + [f"{ch}_{v}" for v in ["Construction","Restaurants","Retail","Professional","Trucking","Healthcare"]]
        elif ch in ["Meta_Retargeting","TikTok_Retargeting"]:
            campaigns[ch] = [f"{ch}_7d", f"{ch}_30d"]
        elif ch == "Google_Search_Brand":
            campaigns[ch] = [f"{ch}_Core", f"{ch}_Competitor", f"{ch}_Local"]
        elif ch == "Partner_Embedded":
            campaigns[ch] = [f"{ch}_Payroll", f"{ch}_Formation", f"{ch}_Accounting"]
        else:
            campaigns[ch] = [f"{ch}_Generic"]

    # Spend
    spend_rows = []
    for d in dates:
        weekday = d.weekday()
        season = 1.0 + (0.12 if weekday < 5 else -0.10)
        holiday = 1.0
        if d.month == 11 and d.day >= 25:
            holiday *= 0.92
        if d.month == 12 and d.day >= 20:
            holiday *= 0.90

        for ch in channels_paid:
            for camp in campaigns[ch]:
                weight = 1.0
                if "_Generic" in camp:
                    weight = 1.2
                if ch == "Google_Search_Brand" and "Core" in camp:
                    weight = 1.4
                noise = rng.lognormal(mean=0, sigma=0.25)
                spend = (base_spend[ch] * season * holiday * noise) / len(campaigns[ch]) * weight
                spend = float(max(0, spend))

                if channel_cpc[ch] > 0 and channel_ctr[ch] > 0:
                    clicks = int(rng.poisson(lam=max(1, spend / channel_cpc[ch])))
                    impressions = int(max(clicks, 1) / channel_ctr[ch])
                else:
                    clicks = 0
                    impressions = 0
                spend_rows.append((d, ch, camp, spend, impressions, clicks))

    ad_spend = pd.DataFrame(spend_rows, columns=["date","channel","campaign_id","spend_usd","impressions","clicks"])

    # Leads from spend
    camp_to_vertical = {}
    for ch, camp_list in campaigns.items():
        for camp in camp_list:
            v = None
            if "Construction" in camp: v = "Construction"
            elif "Restaurants" in camp: v = "Restaurants"
            elif "Retail" in camp: v = "Retail"
            elif "Professional" in camp: v = "Professional_Services"
            elif "Trucking" in camp: v = "Transportation_Trucking"
            elif "Healthcare" in camp: v = "Healthcare"
            camp_to_vertical[camp] = v

    vertical_probs = np.array([0.11,0.10,0.12,0.16,0.12,0.08,0.07,0.08,0.08,0.08])
    vertical_probs = vertical_probs / vertical_probs.sum()

    def sample_vertical_for_campaign(camp):
        target = camp_to_vertical.get(camp)
        if target is None:
            return rng.choice(verticals, p=vertical_probs)
        if rng.random() < 0.70:
            return target
        return rng.choice(verticals, p=vertical_probs)

    lead_mult = {
        "Google_Search_NonBrand": 1.00,
        "Google_Search_Brand": 1.20,
        "Meta_Prospecting": 0.95,
        "Meta_Retargeting": 1.05,
        "TikTok_Prospecting": 0.90,
        "TikTok_Retargeting": 1.00,
        "Partner_Embedded": 1.10,
    }

    lead_rows = []
    lead_id_counter = 1
    for _, r in ad_spend.iterrows():
        ch = r["channel"]
        camp = r["campaign_id"]
        spend = r["spend_usd"]
        lam = (spend / channel_cpl[ch]) * lead_mult[ch]
        lam *= rng.lognormal(mean=0, sigma=0.15)
        n = int(rng.poisson(lam=max(lam, 0.05)))
        if n == 0:
            continue
        base_dt = pd.Timestamp(r["date"])
        minutes = rng.integers(0, 24*60, size=n)
        seconds = rng.integers(0, 60, size=n)
        created_ats = base_dt + pd.to_timedelta(minutes, unit="m") + pd.to_timedelta(seconds, unit="s")

        adset_ids = [f"as_{rng.integers(1000,9999)}" for _ in range(n)]
        creative_ids = [f"cr_{rng.integers(10000,99999)}" for _ in range(n)]

        for i in range(n):
            lead_id = f"L{lead_id_counter:07d}"
            lead_id_counter += 1
            v = sample_vertical_for_campaign(camp)
            st = rng.choice(states, p=state_probs)
            device = rng.choice(devices, p=device_probs)
            lp = rng.choice(landing_variants, p=landing_probs)

            if "Google" in ch:
                click_id = f"gclid_{rng.integers(10**11,10**12-1)}"
            elif "Meta" in ch:
                click_id = f"fbclid_{rng.integers(10**11,10**12-1)}"
            elif "TikTok" in ch:
                click_id = f"ttclid_{rng.integers(10**11,10**12-1)}"
            else:
                click_id = f"pid_{rng.integers(10**11,10**12-1)}"

            lead_rows.append((lead_id, created_ats[i], ch, camp, adset_ids[i], creative_ids[i], click_id, v, st, device, lp))

    leads = pd.DataFrame(lead_rows, columns=[
        "lead_id","lead_created_at","source_channel","source_campaign_id",
        "adset_id","creative_id","click_id","vertical","state","device","landing_page_variant"
    ])

    # add unpaid leads (~14%)
    n_paid = len(leads)
    n_unpaid = int(n_paid * 0.14)
    unpaid_channels = rng.choice(["Organic_Search","Direct","Referral"], size=n_unpaid, p=[0.55,0.30,0.15])
    unpaid_days = rng.choice(dates, size=n_unpaid)
    unpaid_minutes = rng.integers(0, 24*60, size=n_unpaid)
    unpaid_seconds = rng.integers(0, 60, size=n_unpaid)
    unpaid_created = unpaid_days + pd.to_timedelta(unpaid_minutes, unit="m") + pd.to_timedelta(unpaid_seconds, unit="s")
    unpaid_leads = pd.DataFrame({
        "lead_id": [f"L{lead_id_counter+i:07d}" for i in range(n_unpaid)],
        "lead_created_at": unpaid_created,
        "source_channel": unpaid_channels,
        "source_campaign_id": None,
        "adset_id": None,
        "creative_id": None,
        "click_id": None,
        "vertical": rng.choice(verticals, size=n_unpaid, p=vertical_probs),
        "state": rng.choice(states, size=n_unpaid, p=state_probs),
        "device": rng.choice(devices, size=n_unpaid, p=device_probs),
        "landing_page_variant": rng.choice(landing_variants, size=n_unpaid, p=landing_probs),
    })
    lead_id_counter += n_unpaid
    leads = pd.concat([leads, unpaid_leads], ignore_index=True)
    leads = leads.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Business features
    emp = rng.lognormal(mean=1.6, sigma=0.7, size=len(leads))
    employees = np.clip(np.round(emp).astype(int), 1, 250)
    rev_per_emp = rng.lognormal(mean=np.log(90000), sigma=0.35, size=len(leads))
    annual_revenue = np.clip(employees * rev_per_emp, 40000, 50_000_000)
    years = rng.gamma(shape=2.0, scale=3.0, size=len(leads))
    years_in_business = np.clip(np.round(years).astype(int), 0, 35)

    rev_band_edges = [0, 100_000, 250_000, 500_000, 1_000_000, 5_000_000, 20_000_000, float("inf")]
    rev_band_labels = ["<100k","100-250k","250-500k","500k-1M","1-5M","5-20M",">20M"]
    revenue_band = pd.cut(annual_revenue, bins=rev_band_edges, labels=rev_band_labels, right=False).astype(str)

    vertical_risk_base = {
        "Technology_SaaS": 0.25,
        "Professional_Services": 0.26,
        "Real_Estate": 0.30,
        "Retail": 0.36,
        "Cleaning_Janitorial": 0.40,
        "Restaurants": 0.46,
        "Healthcare": 0.52,
        "Manufacturing": 0.56,
        "Construction": 0.62,
        "Transportation_Trucking": 0.72,
    }
    base_risk = np.array([vertical_risk_base[v] for v in leads["vertical"].values])
    lam_claims = np.clip((base_risk * (years_in_business + 1)) / 8.0, 0.02, 2.5)
    prior_claims = rng.poisson(lam=lam_claims)

    noise = rng.normal(0, 0.08, size=len(leads))
    risk_score = np.clip(base_risk + 0.05*np.log1p(employees) + 0.10*prior_claims + noise, 0, 1)

    es_vert_boost = np.array([
        0.35 if v in ["Transportation_Trucking","Construction"] else
        0.20 if v in ["Manufacturing","Healthcare"] else
        0.10 if v in ["Restaurants"] else
        0.05 if v in ["Cleaning_Janitorial","Retail"] else 0.02
        for v in leads["vertical"].values
    ])
    logit_es = -2.2 + 2.8*risk_score + es_vert_boost + 0.02*np.log1p(employees) + 0.12*prior_claims
    p_es = 1/(1+np.exp(-logit_es))
    is_es_eligible = rng.random(len(leads)) < p_es

    leads["employees"] = employees
    leads["annual_revenue_usd"] = np.round(annual_revenue,0).astype(int)
    leads["revenue_band"] = revenue_band
    leads["years_in_business"] = years_in_business
    leads["prior_claims_3y"] = prior_claims
    leads["risk_score"] = np.round(risk_score,3)
    leads["is_es_eligible"] = is_es_eligible.astype(int)

    # Agents + intent + routing
    n_agents = 60
    agent_ids = [f"A{idx:03d}" for idx in range(1, n_agents+1)]
    agent_quality = np.clip(rng.normal(1.0, 0.12, size=n_agents), 0.75, 1.30)
    agent_speed = np.clip(rng.normal(1.0, 0.18, size=n_agents), 0.60, 1.50)
    agent_team = rng.choice(["SMB_West","SMB_East"], size=n_agents, p=[0.45,0.55])
    agents = pd.DataFrame({
        "agent_id": agent_ids,
        "team": agent_team,
        "quality_score": np.round(agent_quality,3),
        "speed_multiplier": np.round(agent_speed,3),
    })
    west_states = set(["CA","WA","AZ","CO"])
    lead_team = np.where(leads["state"].isin(list(west_states)), "SMB_West", "SMB_East")

    base_intent_by_source = {
        "Google_Search_Brand": 0.82,
        "Google_Search_NonBrand": 0.66,
        "Meta_Retargeting": 0.58,
        "Meta_Prospecting": 0.42,
        "TikTok_Retargeting": 0.52,
        "TikTok_Prospecting": 0.38,
        "Partner_Embedded": 0.72,
        "Organic_Search": 0.62,
        "Direct": 0.70,
        "Referral": 0.86,
    }
    base_intent = leads["source_channel"].map(base_intent_by_source).fillna(0.55).values
    lp_boost = leads["landing_page_variant"].map({
        "LP_A_short": 0.00,
        "LP_B_long": 0.03,
        "LP_C_verticalized": 0.06,
    }).values
    intent_score = np.clip(base_intent + lp_boost + rng.normal(0, 0.08, size=len(leads)), 0.05, 0.98)
    leads["intent_score"] = np.round(intent_score,3)

    priority_flag = (intent_score > 0.68) | (leads["annual_revenue_usd"].values > 1_000_000) | (leads["employees"].values >= 12)

    assigned_agents = []
    for t, pr in zip(lead_team, priority_flag):
        pool = agents[agents["team"] == t].copy()
        pool = pool.sort_values("quality_score", ascending=False).reset_index(drop=True)
        top_k = max(4, int(0.30 * len(pool)))
        if pr:
            agent = pool.loc[rng.integers(0, top_k), "agent_id"] if rng.random() < 0.65 else pool.loc[rng.integers(0, len(pool)), "agent_id"]
        else:
            agent = pool.loc[rng.integers(0, top_k), "agent_id"] if rng.random() < 0.20 else pool.loc[rng.integers(0, len(pool)), "agent_id"]
        assigned_agents.append(agent)
    leads["assigned_agent_id"] = assigned_agents

    agent_map_quality = agents.set_index("agent_id")["quality_score"].to_dict()
    agent_map_speed = agents.set_index("agent_id")["speed_multiplier"].to_dict()
    agent_q = leads["assigned_agent_id"].map(agent_map_quality).values
    agent_speed = leads["assigned_agent_id"].map(agent_map_speed).values

    created_ts = pd.to_datetime(leads["lead_created_at"])
    hour = created_ts.dt.hour.values
    weekday = created_ts.dt.weekday.values
    after_hours = (hour < 8) | (hour > 18)
    is_weekend = weekday >= 5

    median_minutes = np.clip(140 - 160*intent_score, 8, 160)
    median_minutes = median_minutes * (1.0 + 0.25*after_hours + 0.20*is_weekend)
    median_minutes = median_minutes * agent_speed
    sigma = 0.85
    mu = np.log(np.maximum(median_minutes, 1))
    ttc = rng.lognormal(mean=mu, sigma=sigma)
    ttc = np.clip(ttc, 1, 7*24*60).astype(int)
    leads["time_to_first_contact_min"] = ttc

    logit_contact = (-0.40 + 2.10*intent_score - 0.0032*ttc - 0.55*leads["risk_score"].values + 0.55*(agent_q - 1.0)
                     + np.where(leads["source_channel"].isin(["Referral","Partner_Embedded"]), 0.35, 0.0)
                     + np.where(leads["source_channel"].isin(["Google_Search_Brand"]), 0.20, 0.0))
    p_contact = 1/(1+np.exp(-logit_contact))
    is_contacted = rng.random(len(leads)) < p_contact
    leads["is_contacted"] = is_contacted.astype(int)
    contacted_at = pd.to_datetime(leads["lead_created_at"]) + pd.to_timedelta(ttc, unit="m")
    leads["contacted_at"] = pd.to_datetime(np.where(is_contacted, contacted_at.astype("datetime64[ns]"), np.datetime64("NaT")))

    base_quote_median = np.where(leads["is_es_eligible"].values==1, 40, 14)
    base_quote_median = base_quote_median * (1.0 + 0.9*leads["risk_score"].values) * (1.0 + 0.10*is_weekend)
    base_quote_median = base_quote_median * np.clip(1.08 - 0.25*(agent_q-1.0), 0.80, 1.20)

    quote_sigma = 0.70
    quote_mu = np.log(np.maximum(base_quote_median, 1))
    ttq_hours = rng.lognormal(mean=quote_mu, sigma=quote_sigma)
    ttq_hours = np.clip(ttq_hours, 0.5, 240).round(1)

    logit_quote = (0.65 + 1.10*intent_score - 1.25*leads["risk_score"].values - 0.35*leads["is_es_eligible"].values
                   + 0.45*(agent_q - 1.0) + np.array([vertical_bind_adj[v] for v in leads["vertical"].values])*0.2)
    p_quote = 1/(1+np.exp(-logit_quote))
    is_quoted = is_contacted & (rng.random(len(leads)) < p_quote)
    leads["is_quoted"] = is_quoted.astype(int)
    quoted_at = pd.to_datetime(leads["contacted_at"]) + pd.to_timedelta(ttq_hours, unit="h")
    leads["quoted_at"] = pd.to_datetime(np.where(is_quoted, quoted_at.astype("datetime64[ns]"), np.datetime64("NaT")))
    leads["time_to_quote_hours"] = np.where(is_quoted, ttq_hours, np.nan)

    base_bind_median_days = np.where(leads["is_es_eligible"].values==1, 7.5, 4.0)
    vert_bind_mult = np.array([1.25 if v in ["Construction","Transportation_Trucking","Manufacturing"] else
                               1.10 if v in ["Healthcare","Restaurants"] else 1.0 for v in leads["vertical"].values])
    base_bind_median_days = base_bind_median_days * vert_bind_mult * (1.0 + 0.6*leads["risk_score"].values)
    base_bind_median_days = base_bind_median_days * np.clip(1.05 - 0.18*(agent_q-1.0), 0.85, 1.15)

    bind_sigma = 0.65
    bind_mu = np.log(np.maximum(base_bind_median_days, 0.2))
    ttb_days = rng.lognormal(mean=bind_mu, sigma=bind_sigma)
    ttb_days = np.clip(ttb_days, 0.2, 45).round(2)

    channel_bind_adj = leads["source_channel"].map({
        "Google_Search_Brand": 0.30,
        "Google_Search_NonBrand": 0.15,
        "Meta_Retargeting": 0.08,
        "Meta_Prospecting": -0.05,
        "TikTok_Retargeting": 0.02,
        "TikTok_Prospecting": -0.08,
        "Partner_Embedded": 0.20,
        "Referral": 0.35,
        "Direct": 0.18,
        "Organic_Search": 0.12,
    }).fillna(0.0).values

    logit_bind = (-0.95 + 1.35*intent_score - 0.95*leads["risk_score"].values - 0.20*leads["is_es_eligible"].values
                  + 0.65*(agent_q - 1.0) + np.array([vertical_bind_adj[v] for v in leads["vertical"].values])
                  + channel_bind_adj - 0.035*ttb_days)
    p_bind = 1/(1+np.exp(-logit_bind))
    is_bound = is_quoted & (rng.random(len(leads)) < p_bind)
    leads["is_bound"] = is_bound.astype(int)
    leads["p_bind_synthetic"] = np.round(p_bind,4)

    bound_at = pd.to_datetime(leads["quoted_at"]) + pd.to_timedelta(ttb_days, unit="D")
    leads["bound_at"] = pd.to_datetime(np.where(is_bound, bound_at.astype("datetime64[ns]"), np.datetime64("NaT")))
    leads["time_to_bind_days"] = np.where(is_bound, ttb_days, np.nan)

    # Premium + retention
    bp = np.array([base_premium[v] for v in leads["vertical"].values])
    emp_factor = 1.0 + 0.10*np.log1p(leads["employees"].values)
    rev_factor = 1.0 + 0.06*(leads["annual_revenue_usd"].values > 1_000_000) + 0.12*(leads["annual_revenue_usd"].values > 5_000_000)
    risk_factor = 1.0 + 0.45*leads["risk_score"].values
    es_factor = np.where(leads["is_es_eligible"].values==1, 1.28, 1.00)
    ch_prem_factor = leads["source_channel"].map({
        "Partner_Embedded": 1.08,
        "Referral": 1.12,
        "Google_Search_Brand": 1.05,
        "Google_Search_NonBrand": 1.00,
        "Meta_Retargeting": 0.98,
        "Meta_Prospecting": 0.92,
        "TikTok_Retargeting": 0.96,
        "TikTok_Prospecting": 0.90,
        "Organic_Search": 1.00,
        "Direct": 1.02,
    }).fillna(1.0).values

    prem_mean = bp * emp_factor * rev_factor * risk_factor * es_factor * ch_prem_factor
    prem_noise = rng.lognormal(mean=0, sigma=0.35, size=len(leads))
    premium_annual = np.clip(prem_mean * prem_noise, 400, 250_000)

    bound_idx = leads["is_bound"].values.astype(bool)
    leads["bound_premium_annual_usd"] = np.where(bound_idx, np.round(premium_annual,0).astype(int), 0)

    bundle_base = 0.25 + 0.03*np.log1p(leads["employees"].values) + 0.12*(leads["annual_revenue_usd"].values > 1_000_000)
    bundle_base += np.array([0.10 if v in ["Construction","Restaurants","Retail","Manufacturing"] else 0.06 if v in ["Healthcare"] else 0.04 for v in leads["vertical"].values])
    bundle_base += 0.06*(agent_q-1.0)
    bundle_base = np.clip(bundle_base, 0.05, 0.70)

    bundle_count = np.ones(len(leads), dtype=int)
    u = rng.random(len(leads))
    bundle_count = np.where(bound_idx & (u < bundle_base), 2, 1)
    u2 = rng.random(len(leads))
    bundle_count = np.where(bound_idx & (bundle_count==2) & (u2 < (bundle_base*0.35)), 3, bundle_count)
    leads["bound_policy_count"] = np.where(bound_idx, bundle_count, 0)

    commission_rate = np.where(leads["is_es_eligible"].values==1, 0.15, 0.12) + rng.normal(0, 0.01, size=len(leads))
    commission_rate = np.clip(commission_rate, 0.08, 0.22)
    leads["commission_rate"] = np.where(bound_idx, np.round(commission_rate,3), np.nan)

    ret_base = 0.80 + 0.03*(leads["bound_policy_count"].values >= 2) + 0.02*(leads["bound_policy_count"].values >= 3)
    ret_base -= 0.05*leads["is_es_eligible"].values
    ret_base -= 0.02*(leads["prior_claims_3y"].values >= 2)
    ret_base += 0.04*(agent_q-1.0)
    ret_base += rng.normal(0, 0.04, size=len(leads))
    ret_12 = np.clip(ret_base, 0.45, 0.95)
    ret_24 = np.clip(ret_12 * (0.85 + rng.normal(0,0.04,size=len(leads))), 0.30, 0.92)

    leads["retention_prob_12m"] = np.where(bound_idx, np.round(ret_12,3), np.nan)
    leads["retention_prob_24m"] = np.where(bound_idx, np.round(ret_24,3), np.nan)
    egp_24m = leads["bound_premium_annual_usd"].values * commission_rate * (1 + ret_12)
    leads["expected_gross_profit_24m_usd"] = np.where(bound_idx, np.round(egp_24m,0).astype(int), 0)

    # Policies
    def choose_primary(vertical):
        if vertical == "Technology_SaaS":
            return "Professional_Liability"
        if vertical in ["Restaurants","Retail","Cleaning_Janitorial","Real_Estate","Professional_Services"]:
            return "BOP"
        if vertical in ["Construction","Manufacturing","Healthcare","Transportation_Trucking"]:
            return "General_Liability"
        return "BOP"

    def choose_secondary(vertical, employees):
        if employees >= 2 and vertical in ["Construction","Restaurants","Retail","Cleaning_Janitorial","Manufacturing","Healthcare","Transportation_Trucking"]:
            return "Workers_Comp"
        if vertical == "Technology_SaaS":
            return "Cyber"
        if vertical == "Professional_Services":
            return "Professional_Liability"
        return "Workers_Comp" if employees >= 5 else "Cyber"

    def choose_third(vertical):
        if vertical == "Technology_SaaS":
            return rng.choice(["Cyber","BOP"], p=[0.65,0.35])
        if vertical in ["Construction","Transportation_Trucking"]:
            return rng.choice(["Workers_Comp","BOP"], p=[0.70,0.30])
        if vertical in ["Restaurants","Retail"]:
            return rng.choice(["Workers_Comp","General_Liability"], p=[0.55,0.45])
        return rng.choice(["Cyber","Professional_Liability","BOP"], p=[0.25,0.35,0.40])

    policy_rows = []
    pid = 1
    bound_leads = leads.loc[bound_idx, ["lead_id","bound_at","bound_premium_annual_usd","bound_policy_count","vertical","employees","is_es_eligible","commission_rate","state"]]
    for row in bound_leads.itertuples(index=False):
        lid = row.lead_id
        total_prem = int(row.bound_premium_annual_usd)
        cnt = int(row.bound_policy_count)
        vert = row.vertical
        empc = int(row.employees)
        is_es = int(row.is_es_eligible)
        comm = float(row.commission_rate)
        bound_at_ts = row.bound_at

        types = [choose_primary(vert)]
        if cnt >= 2:
            sec = choose_secondary(vert, empc)
            types.append(sec if sec not in types else "Workers_Comp")
        if cnt >= 3:
            third = choose_third(vert)
            types.append(third if third not in types else "Cyber")

        split = rng.dirichlet(alpha=np.ones(len(types))*1.3)
        prem_parts = np.maximum(200, np.round(total_prem * split, 0)).astype(int)
        prem_parts[0] += int(total_prem - prem_parts.sum())

        for t, p in zip(types, prem_parts):
            policy_id = f"P{pid:07d}"
            pid += 1
            carrier_tier = "E&S" if is_es==1 else "Standard"
            policy_rows.append((policy_id, lid, bound_at_ts, t, int(p), carrier_tier, is_es, comm, row.state))

    policies = pd.DataFrame(policy_rows, columns=[
        "policy_id","lead_id","bound_at","policy_type","premium_annual_usd","carrier_tier","is_es","commission_rate","state"
    ])

    # Touchpoints (multi-touch + sales + conversion)
    origin_probs_by_source = {
        "Google_Search_Brand": (["Google_Search_Brand","Meta_Prospecting","TikTok_Prospecting","Organic_Search","Direct","Referral","Partner_Embedded","Google_Search_NonBrand"],
                                [0.36,0.26,0.12,0.10,0.08,0.03,0.03,0.02]),
        "Google_Search_NonBrand": (["Google_Search_NonBrand","Meta_Prospecting","TikTok_Prospecting","Organic_Search","Direct","Google_Search_Brand"],
                                   [0.58,0.18,0.08,0.08,0.04,0.04]),
        "Meta_Prospecting": (["Meta_Prospecting","TikTok_Prospecting","Google_Search_NonBrand","Organic_Search","Direct"],
                             [0.72,0.10,0.08,0.05,0.05]),
        "Meta_Retargeting": (["Meta_Prospecting","Meta_Retargeting","Google_Search_NonBrand","TikTok_Prospecting","Direct","Organic_Search"],
                             [0.42,0.26,0.14,0.08,0.06,0.04]),
        "TikTok_Prospecting": (["TikTok_Prospecting","Meta_Prospecting","Google_Search_NonBrand","Organic_Search","Direct"],
                               [0.74,0.08,0.08,0.05,0.05]),
        "TikTok_Retargeting": (["TikTok_Prospecting","TikTok_Retargeting","Meta_Prospecting","Google_Search_NonBrand","Direct"],
                               [0.45,0.27,0.12,0.10,0.06]),
        "Partner_Embedded": (["Partner_Embedded","Google_Search_Brand","Direct","Referral"],
                             [0.82,0.10,0.05,0.03]),
        "Organic_Search": (["Organic_Search","Meta_Prospecting","Google_Search_NonBrand","Direct"],
                           [0.72,0.10,0.10,0.08]),
        "Direct": (["Direct","Meta_Prospecting","Google_Search_Brand","Organic_Search"],
                   [0.70,0.12,0.10,0.08]),
        "Referral": (["Referral","Google_Search_Brand","Direct"],
                     [0.80,0.12,0.08]),
    }

    def pick_origin(source_channel):
        chans, probs = origin_probs_by_source.get(source_channel, (["Direct"], [1.0]))
        probs = np.array(probs) / np.sum(probs)
        return rng.choice(chans, p=probs)

    def choose_campaign_for_touch(channel, vertical, fallback_source_campaign=None):
        if channel not in campaigns:
            return None, None, None
        if fallback_source_campaign is not None and fallback_source_campaign.startswith(channel):
            return fallback_source_campaign, None, None

        camp_list = campaigns[channel]
        suffix_map = {
            "Construction": "Construction",
            "Restaurants": "Restaurants",
            "Retail": "Retail",
            "Professional_Services": "Professional",
            "Transportation_Trucking": "Trucking",
            "Healthcare": "Healthcare",
        }
        target_suffix = suffix_map.get(vertical)
        candidates = [c for c in camp_list if target_suffix and target_suffix in c]
        generic = [c for c in camp_list if "Generic" in c] or camp_list
        camp = rng.choice(candidates) if (candidates and rng.random() < 0.60) else rng.choice(generic)
        adset = f"as_{rng.integers(1000,9999)}"
        creative = f"cr_{rng.integers(10000,99999)}"
        return camp, adset, creative

    touch_rows = []
    eid = 1
    origin_channels = []

    for row in leads.itertuples(index=False):
        lid = row.lead_id
        lead_time = pd.Timestamp(row.lead_created_at)
        source_ch = row.source_channel
        source_camp = row.source_campaign_id if isinstance(row.source_campaign_id, str) else None
        vert = row.vertical
        dev = row.device

        origin = pick_origin(source_ch)
        origin_channels.append(origin)

        path = [origin]
        is_tof = origin in ["Meta_Prospecting","TikTok_Prospecting"]

        if origin == "Meta_Prospecting":
            if rng.random() < 0.55: path.append("Meta_Retargeting")
            if rng.random() < 0.42: path.append(rng.choice(["Direct","Organic_Search"], p=[0.6,0.4]))
            if rng.random() < 0.35: path.append("Google_Search_Brand")
            if rng.random() < 0.18: path.append("Google_Search_NonBrand")
        elif origin == "TikTok_Prospecting":
            if rng.random() < 0.52: path.append("TikTok_Retargeting")
            if rng.random() < 0.40: path.append(rng.choice(["Direct","Organic_Search"], p=[0.6,0.4]))
            if rng.random() < 0.28: path.append("Google_Search_Brand")
            if rng.random() < 0.16: path.append("Meta_Retargeting")
        elif origin == "Google_Search_NonBrand":
            if rng.random() < 0.25: path.append("Direct")
            if rng.random() < 0.18: path.append("Google_Search_Brand")
            if rng.random() < 0.12: path.append("Meta_Retargeting")
        elif origin == "Google_Search_Brand":
            if rng.random() < 0.20: path.append("Direct")
            if rng.random() < 0.12: path.append("Meta_Retargeting")
        elif origin == "Partner_Embedded":
            if rng.random() < 0.18: path.append("Direct")
            if rng.random() < 0.10: path.append("Google_Search_Brand")
        elif origin == "Organic_Search":
            if rng.random() < 0.25: path.append("Direct")
            if rng.random() < 0.15: path.append("Google_Search_Brand")
            if rng.random() < 0.10: path.append("Meta_Retargeting")
        elif origin == "Referral":
            if rng.random() < 0.20: path.append("Direct")
            if rng.random() < 0.12: path.append("Google_Search_Brand")
        else:
            if rng.random() < 0.18: path.append("Google_Search_NonBrand")
            if rng.random() < 0.12: path.append("Meta_Retargeting")

        if path[-1] != source_ch:
            path.append(source_ch)

        if "Meta_Retargeting" in path and rng.random() < 0.25:
            path.insert(max(1, len(path)-1), "Meta_Retargeting")
        if "TikTok_Retargeting" in path and rng.random() < 0.22:
            path.insert(max(1, len(path)-1), "TikTok_Retargeting")

        cleaned = [path[0]]
        for c in path[1:]:
            if c == cleaned[-1] and rng.random() < 0.65:
                continue
            cleaned.append(c)
        path = cleaned

        n_marketing = max(1, len(path)-1)
        window_days = 14 if is_tof else 7
        hrs = np.sort(np.clip(rng.exponential(scale=window_days*24/3.5, size=n_marketing) + 0.4, 0.4, window_days*24))[::-1]

        for i in range(n_marketing):
            ch = path[i]
            event_time = lead_time - pd.to_timedelta(float(hrs[i]), unit="h")
            call_outcome = None
            call_dur = None
            email_tmpl = None

            if ch in channels_paid:
                camp, adset, creative = choose_campaign_for_touch(ch, vert, None)
                etype = "ad_click"
                stage = "marketing"
            elif ch in ["Direct","Organic_Search"]:
                camp, adset, creative = None, None, None
                etype = "site_visit"
                stage = "marketing"
            elif ch == "Referral":
                camp, adset, creative = None, None, None
                etype = "referral_click"
                stage = "marketing"
            elif ch == "Partner_Embedded":
                camp, adset, creative = choose_campaign_for_touch(ch, vert, None)
                etype = "partner_referral"
                stage = "marketing"
            else:
                camp, adset, creative = None, None, None
                etype = "touch"
                stage = "marketing"

            touch_rows.append((f"E{eid:09d}", lid, event_time, etype, stage, ch, camp, adset, creative, dev, call_outcome, call_dur, email_tmpl))
            eid += 1

        # form start / submit
        form_start_time = lead_time - pd.to_timedelta(int(rng.integers(1,6)), unit="m")
        touch_rows.append((f"E{eid:09d}", lid, form_start_time, "form_start", "marketing", source_ch, source_camp, row.adset_id, row.creative_id, dev, None, None, None)); eid += 1
        touch_rows.append((f"E{eid:09d}", lid, lead_time, "form_submit", "marketing", source_ch, source_camp, row.adset_id, row.creative_id, dev, None, None, None)); eid += 1

        # sales
        email1_time = lead_time + pd.to_timedelta(int(rng.integers(20,120)), unit="m")
        touch_rows.append((f"E{eid:09d}", lid, email1_time, "email_sent", "sales", "Email", None, None, None, dev, None, None, "nurture_1")); eid += 1

        call_attempt_prob = 0.55 + 0.35*row.intent_score + (0.10 if source_ch in ["Google_Search_Brand","Referral","Partner_Embedded"] else 0.0)
        call_attempt_prob = min(call_attempt_prob, 0.98)
        if rng.random() < call_attempt_prob:
            call_time = lead_time + pd.to_timedelta(int(row.time_to_first_contact_min), unit="m")
            if row.is_contacted == 1:
                outcome = "connected"
                dur = int(np.clip(rng.gamma(shape=2.2, scale=220), 45, 2400))
                touch_rows.append((f"E{eid:09d}", lid, call_time, "call_attempt", "sales", "Phone", None, None, None, dev, outcome, dur, None)); eid += 1
                touch_rows.append((f"E{eid:09d}", lid, call_time + pd.to_timedelta(5, unit="s"), "call_connected", "sales", "Phone", None, None, None, dev, outcome, dur, None)); eid += 1
            else:
                outcome = rng.choice(["no_answer","voicemail"], p=[0.65,0.35])
                dur = int(np.clip(rng.gamma(shape=1.4, scale=35), 10, 180))
                touch_rows.append((f"E{eid:09d}", lid, call_time, "call_attempt", "sales", "Phone", None, None, None, dev, outcome, dur, None)); eid += 1

        if row.is_quoted == 0 and rng.random() < 0.65:
            email2_time = lead_time + pd.to_timedelta(int(rng.integers(18,72)), unit="h")
            touch_rows.append((f"E{eid:09d}", lid, email2_time, "email_sent", "sales", "Email", None, None, None, dev, None, None, "nurture_2")); eid += 1

        if row.is_quoted == 1 and pd.notna(row.quoted_at):
            qt = pd.Timestamp(row.quoted_at)
            touch_rows.append((f"E{eid:09d}", lid, qt, "quote_sent", "sales", "Email", None, None, None, dev, None, None, "quote_pdf")); eid += 1

        if row.is_bound == 1 and pd.notna(row.bound_at):
            bt = pd.Timestamp(row.bound_at)
            touch_rows.append((f"E{eid:09d}", lid, bt, "bind_complete", "conversion", "Direct", None, None, None, dev, None, None, None)); eid += 1

    touchpoints = pd.DataFrame(touch_rows, columns=[
        "event_id","lead_id","event_time","event_type","event_stage","channel",
        "campaign_id","adset_id","creative_id","device","call_outcome","call_duration_sec","email_template"
    ])
    leads["origin_channel"] = origin_channels

    # Write outputs
    os.makedirs(out_dir, exist_ok=True)
    leads.sort_values("lead_created_at").to_csv(os.path.join(out_dir, "leads.csv"), index=False)
    touchpoints.sort_values(["lead_id","event_time"]).to_csv(os.path.join(out_dir, "touchpoints.csv"), index=False)
    ad_spend.sort_values(["date","channel","campaign_id"]).to_csv(os.path.join(out_dir, "ad_spend_daily.csv"), index=False)
    policies.sort_values(["lead_id","policy_id"]).to_csv(os.path.join(out_dir, "policies.csv"), index=False)
    agents.to_csv(os.path.join(out_dir, "agents.csv"), index=False)

    readme = f\"\"\"# Harper-like Synthetic GTM Dataset (v1)\n\nFully synthetic dataset intended for paid growth + attribution + lead scoring MVPs.\n\nCounts:\n- Leads: {{len(leads):,}}\n- Touchpoints: {{len(touchpoints):,}}\n- Bound leads: {{int(leads['is_bound'].sum()):,}}\n- Policies: {{len(policies):,}}\n\"\"\"
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(readme)

    # Zip
    zip_path = out_dir + ".zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in ["leads.csv","touchpoints.csv","ad_spend_daily.csv","policies.csv","agents.csv","README.md"]:
            z.write(os.path.join(out_dir, fn), arcname=fn)
    print("Wrote:", zip_path)

if __name__ == "__main__":
    main()
