"""
Enrich existing CSV files with operational columns needed for AI-native brokerage dashboard.
Adds columns for automation tracking, AI vs human performance, carrier intelligence, etc.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def enrich_leads(leads: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add operational columns to leads.csv"""
    rng = np.random.default_rng(seed)
    out = leads.copy()
    
    # 1. Automation rate: % of tasks handled by AI (0-100%)
    # Higher automation for lower-risk, higher-intent leads
    automation_base = 0.15 + 0.20 * out["intent_score"].fillna(0.5) - 0.15 * out["risk_score"].fillna(0.5)
    automation_base = np.clip(automation_base, 0.05, 0.85)
    automation_noise = rng.normal(0, 0.08, size=len(out))
    out["automation_rate"] = np.clip(automation_base + automation_noise, 0.0, 1.0)
    
    # 2. AI contact type: "ai_voice", "human", "ai_email", "mixed"
    # AI voice more likely for high-intent, low-risk leads
    ai_voice_prob = 0.20 + 0.30 * out["intent_score"].fillna(0.5) - 0.20 * out["risk_score"].fillna(0.5)
    ai_voice_prob = np.clip(ai_voice_prob, 0.05, 0.70)
    contact_types = []
    for i in range(len(out)):
        p = ai_voice_prob.iloc[i] if hasattr(ai_voice_prob, 'iloc') else ai_voice_prob[i]
        rand = rng.random()
        if rand < p:
            contact_types.append("ai_voice")
        elif rand < p + 0.15:
            contact_types.append("ai_email")
        elif rand < p + 0.25:
            contact_types.append("mixed")
        else:
            contact_types.append("human")
    out["ai_contact_type"] = contact_types
    
    # 3. Submission-to-Quote (STQ) ratio: whether quote was received from carrier
    # Only applies to quoted leads
    # Vary by vertical - some verticals have better carrier response rates
    vertical_stq_base = {
        "Technology_SaaS": 0.85,
        "Professional_Services": 0.88,
        "Real_Estate": 0.82,
        "Retail": 0.90,
        "Cleaning_Janitorial": 0.87,
        "Restaurants": 0.92,
        "Healthcare": 0.80,
        "Manufacturing": 0.75,
        "Construction": 0.72,
        "Transportation_Trucking": 0.70,
    }
    
    stq_base = np.array([vertical_stq_base.get(v, 0.85) for v in out["vertical"].values])
    # Add some noise but keep it realistic
    stq_noise = rng.normal(0, 0.08, size=len(out))
    stq_ratio = np.clip(stq_base + stq_noise, 0.50, 0.98)
    
    # Only apply to quoted leads, others get NaN
    stq_ratio = np.where(out["is_quoted"] == 1, stq_ratio, np.nan)
    out["stq_ratio"] = stq_ratio
    
    # 4. CAC (Customer Acquisition Cost) - synthetic based on channel
    channel_cac = {
        "Google_Search_NonBrand": 92.0,
        "Google_Search_Brand": 38.0,
        "Meta_Prospecting": 105.0,
        "Meta_Retargeting": 72.0,
        "TikTok_Prospecting": 112.0,
        "TikTok_Retargeting": 85.0,
        "Partner_Embedded": 28.0,
        "Organic_Search": 0.0,
        "Direct": 0.0,
        "Referral": 15.0,
    }
    cac_base = out["source_channel"].map(channel_cac).fillna(50.0)
    cac_noise = rng.lognormal(mean=0, sigma=0.15, size=len(out))
    out["cac_usd"] = cac_base * cac_noise
    
    # 5. LTV (Lifetime Value) - based on premium, retention, commission
    # LTV = premium * commission_rate * (1 + retention_12m + retention_24m * retention_12m)
    bound_idx = out["is_bound"] == 1
    ltv = np.zeros(len(out))
    for i in range(len(out)):
        if bound_idx.iloc[i] if hasattr(bound_idx, 'iloc') else bound_idx[i]:
            prem = out["bound_premium_annual_usd"].iloc[i] if hasattr(out["bound_premium_annual_usd"], 'iloc') else out["bound_premium_annual_usd"][i]
            comm = out["commission_rate"].iloc[i] if pd.notna(out["commission_rate"].iloc[i]) else 0.12
            ret12 = out["retention_prob_12m"].iloc[i] if pd.notna(out["retention_prob_12m"].iloc[i]) else 0.80
            ret24 = out["retention_prob_24m"].iloc[i] if pd.notna(out["retention_prob_24m"].iloc[i]) else 0.70
            # Simplified: 2 years of premium with retention
            ltv[i] = prem * comm * (1 + ret12 + ret24 * ret12)
        else:
            ltv[i] = 0.0
    out["ltv_usd"] = ltv
    
    # 6. Churn risk score (0-1, higher = more likely to churn)
    # Based on prior claims, risk score, retention prob
    churn_base = 0.20 + 0.15 * out["prior_claims_3y"].fillna(0) + 0.20 * out["risk_score"].fillna(0.5)
    churn_base -= 0.15 * out["retention_prob_12m"].fillna(0.8)
    churn_base = np.clip(churn_base, 0.05, 0.95)
    churn_noise = rng.normal(0, 0.05, size=len(out))
    out["churn_risk_score"] = np.clip(churn_base + churn_noise, 0.0, 1.0)
    
    # 7. Net commission revenue (premium * commission_rate, for bound leads)
    out["net_commission_revenue_usd"] = np.where(
        bound_idx,
        out["bound_premium_annual_usd"] * out["commission_rate"].fillna(0.12),
        0.0
    )
    
    return out


def enrich_touchpoints(touchpoints: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add operational columns to touchpoints.csv"""
    rng = np.random.default_rng(seed)
    out = touchpoints.copy()
    
    # Add AI vs human flag for sales touchpoints
    # AI more likely for email, less likely for calls
    ai_prob = np.where(
        out["event_type"] == "email_sent",
        0.65,  # Most emails are AI
        np.where(
            out["event_type"] == "call_connected",
            0.15,  # Fewer calls are AI
            0.30  # Default
        )
    )
    out["is_ai_handled"] = rng.random(len(out)) < ai_prob
    
    return out


def enrich_policies(policies: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add operational columns to policies.csv"""
    rng = np.random.default_rng(seed)
    out = policies.copy()
    
    # Carrier hit ratio: whether carrier accepted the submission (0-1)
    # Higher for standard tier, lower for E&S
    hit_base = np.where(out["carrier_tier"] == "Standard", 0.75, 0.55)
    hit_noise = rng.normal(0, 0.10, size=len(out))
    out["carrier_hit_ratio"] = np.clip(hit_base + hit_noise, 0.20, 0.95)
    
    return out


def main():
    root = Path(__file__).parent
    
    print("Loading data files...")
    leads = pd.read_csv(root / "leads.csv", parse_dates=["lead_created_at", "contacted_at", "quoted_at", "bound_at"])
    touchpoints = pd.read_csv(root / "touchpoints.csv", parse_dates=["event_time"])
    policies = pd.read_csv(root / "policies.csv", parse_dates=["bound_at"])
    
    print("Enriching leads.csv...")
    leads_enriched = enrich_leads(leads)
    
    print("Enriching touchpoints.csv...")
    touchpoints_enriched = enrich_touchpoints(touchpoints)
    
    print("Enriching policies.csv...")
    policies_enriched = enrich_policies(policies)
    
    # Backup original files
    print("Backing up original files...")
    leads.to_csv(root / "leads.csv.backup", index=False)
    touchpoints.to_csv(root / "touchpoints.csv.backup", index=False)
    policies.to_csv(root / "policies.csv.backup", index=False)
    
    # Write enriched files
    print("Writing enriched files...")
    leads_enriched.to_csv(root / "leads.csv", index=False)
    touchpoints_enriched.to_csv(root / "touchpoints.csv", index=False)
    policies_enriched.to_csv(root / "policies.csv", index=False)
    
    print("✅ Data enrichment complete!")
    print(f"Leads: {len(leads_enriched)} rows, {len(leads_enriched.columns)} columns")
    print(f"Touchpoints: {len(touchpoints_enriched)} rows, {len(touchpoints_enriched.columns)} columns")
    print(f"Policies: {len(policies_enriched)} rows, {len(policies_enriched.columns)} columns")
    print("\nNew columns added:")
    print("  leads.csv: automation_rate, ai_contact_type, stq_ratio, cac_usd, ltv_usd, churn_risk_score, net_commission_revenue_usd")
    print("  touchpoints.csv: is_ai_handled")
    print("  policies.csv: carrier_hit_ratio")


if __name__ == "__main__":
    main()
