from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class GTMData:
    leads: pd.DataFrame
    touchpoints: pd.DataFrame
    ad_spend_daily: pd.DataFrame
    policies: pd.DataFrame
    agents: pd.DataFrame


def _read_csv(path: Path, *, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=parse_dates or [])


def load_gtm_data(root_dir: str | Path) -> GTMData:
    root = Path(root_dir)

    leads = _read_csv(
        root / "leads.csv",
        parse_dates=["lead_created_at", "contacted_at", "quoted_at", "bound_at"],
    )
    touchpoints = _read_csv(root / "touchpoints.csv", parse_dates=["event_time"])
    ad_spend_daily = _read_csv(root / "ad_spend_daily.csv", parse_dates=["date"])
    policies = _read_csv(root / "policies.csv", parse_dates=["bound_at"])
    agents = _read_csv(root / "agents.csv")

    return GTMData(
        leads=leads,
        touchpoints=touchpoints,
        ad_spend_daily=ad_spend_daily,
        policies=policies,
        agents=agents,
    )

