# FifaPredictor

A fork of [MiroFish](https://github.com/666ghj/MiroFish) adapted for **FIFA World Cup 2026** match and tournament prediction.

Uses multi-agent LLM simulation (Claude) fed with live data from SofaScore + FIFA + YouTube video analysis to produce:
- Match outcome probabilities (win/draw/loss)
- Expected goals and over/under lines
- Goal timing distributions (per 15-min bucket)
- Likely goalscorers per match
- Tournament winner / finalist / top-scorer probabilities

All predictions are exported as **Polymarket-compatible YES/NO questions** ranked by betting edge.

---

## Architecture

```
SofaScore API ──┐
FIFA data       ├──▶ Orchestrator ──▶ Team/Player models
YouTube API ────┘
                            │
                            ▼
                   Claude match simulator
                   (N simulations per match)
                            │
                            ▼
                   Monte Carlo tournament
                   simulator (1000 runs)
                            │
                            ▼
                   Polymarket question mapper
                   (yes_prob, edge, recommended_side)
```

---

## Quick Start

```bash
cd backend
cp ../.env.example .env
# Fill in ANTHROPIC_API_KEY (required), YOUTUBE_API_KEY (optional)

pip install -r requirements.txt

# Run the API
python run.py

# Or run CLI predictions directly
cd ..
python scripts/run_predictions.py --sims 50 --tournament-sims 500 --out data/output/predictions.json
```

API runs on `http://localhost:5001`.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status check |
| POST | `/api/refresh-data` | Re-fetch SofaScore data |
| GET | `/api/teams` | List all 48 WC teams |
| GET | `/api/matches` | List all WC matches |
| POST | `/api/predict/match` | Predict a single match |
| POST | `/api/predict/tournament` | Full tournament simulation |
| GET | `/api/polymarket/top-bets` | Best-edge Polymarket questions |

---

## Key Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude API key (required) | — |
| `YOUTUBE_API_KEY` | YouTube Data API v3 (optional, for video analysis) | — |
| `SIMULATION_RUNS` | Simulations per match | 200 |
| `CLAUDE_MODEL` | Claude model to use | claude-sonnet-4-6 |
| `CACHE_DIR` | Where to store scraped data | ./data/cache |

---

## Cost Estimate

Running 50 simulations per match × 104 WC matches = ~5,200 Claude API calls.  
At ~800 tokens/call (input+output), that is ~4.2M tokens ≈ **$12–$15 total** with claude-sonnet-4-6.

Use `--sims 10` for quick/cheap runs during development.

---

## Upstream Webpage

Predictions JSON at `data/output/predictions.json` is intended to be served on
[RafaelOrtiz.github.io](https://rafaelortiz.github.io) as an interactive dashboard.
