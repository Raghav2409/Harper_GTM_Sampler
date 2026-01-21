# Harper GTM Co‑Pilot (Streamlit)

This repo contains a **Streamlit dashboard + GTM co‑pilot chatbot** powered by the synthetic Harper-like GTM dataset:

- `leads.csv`
- `touchpoints.csv`
- `ad_spend_daily.csv`
- `policies.csv`
- `agents.csv`

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option A (recommended): export key as env var
export OPENAI_API_KEY="YOUR_KEY"

streamlit run app.py
```

## OpenAI key (important)

- **Do not hardcode keys in code or commit them to git.**
- Preferred: set `OPENAI_API_KEY` as an environment variable.
- Alternative: Streamlit secrets in `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "YOUR_KEY"
```

## What you get

- **Dashboard**: funnel, paid efficiency, speed-to-lead impact, segments (vertical/state), agent performance, attribution snapshot, daily trends.
- **GTM Co‑Pilot**: chat assistant that answers using the computed tables/insights from the dataset.

