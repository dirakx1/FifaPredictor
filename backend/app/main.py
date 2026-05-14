"""FifaPredictor API — FastAPI backend."""
import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .data.orchestrator import build_dataset
from .data.scrapers.sofascore import fetch_all_wc_data
from .predictions.match_predictor import predict_match
from .predictions.tournament_simulator import simulate_tournament
from .polymarket.market_mapper import (
    match_to_polymarket_questions,
    tournament_to_polymarket_questions,
    get_top_value_bets,
)

app = FastAPI(
    title="FifaPredictor",
    description="FIFA World Cup 2026 prediction engine — powered by multi-agent LLM simulation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory dataset (loaded on startup)
_dataset: dict = {}
_match_predictions: dict = {}   # match_id -> MatchPrediction
_tournament_pred = None


@app.on_event("startup")
async def startup():
    global _dataset
    print("[startup] Loading dataset...")
    _dataset = await build_dataset(use_video=False)
    print(f"[startup] {len(_dataset['teams'])} teams, {len(_dataset['matches'])} matches")


@app.get("/health")
async def health():
    return {"status": "ok", "teams": len(_dataset.get("teams", {})), "matches": len(_dataset.get("matches", []))}


@app.post("/api/refresh-data")
async def refresh_data():
    """Re-fetch SofaScore data and rebuild dataset."""
    global _dataset
    cache_dir = os.getenv("CACHE_DIR", "./data/cache")
    raw = await fetch_all_wc_data(cache_dir)
    _dataset = await build_dataset(use_video=False)
    return {"status": "refreshed", "teams": len(_dataset["teams"]), "matches": len(_dataset["matches"])}


@app.get("/api/teams")
async def list_teams():
    return [t.dict() for t in _dataset.get("teams", {}).values()]


@app.get("/api/matches")
async def list_matches():
    return [m.dict() for m in _dataset.get("matches", [])]


class PredictMatchRequest(BaseModel):
    match_id: str
    n_simulations: int = 50


@app.post("/api/predict/match")
async def predict_match_endpoint(req: PredictMatchRequest):
    teams = _dataset.get("teams", {})
    matches = {m.id: m for m in _dataset.get("matches", [])}

    if req.match_id not in matches:
        raise HTTPException(404, f"Match {req.match_id} not found")

    match = matches[req.match_id]
    home_team = teams.get(match.home_team_id)
    away_team = teams.get(match.away_team_id)

    if not home_team or not away_team:
        raise HTTPException(404, "Team data missing")

    home_insights = _dataset.get("video_insights", {}).get(match.home_team_id, "")
    away_insights = _dataset.get("video_insights", {}).get(match.away_team_id, "")

    pred = predict_match(
        match=match,
        home_team=home_team,
        away_team=away_team,
        home_players=[],
        away_players=[],
        home_video_insights=home_insights,
        away_video_insights=away_insights,
        n_simulations=req.n_simulations,
    )
    _match_predictions[req.match_id] = pred

    # Generate Polymarket questions
    questions = match_to_polymarket_questions(pred, home_team.name, away_team.name)
    top_bets = get_top_value_bets(questions, min_edge=0.05)

    return {
        "prediction": pred.dict(),
        "polymarket_questions": [q.to_dict() for q in questions],
        "top_value_bets": top_bets,
    }


@app.post("/api/predict/tournament")
async def predict_tournament(n_simulations: int = Query(default=200, le=2000)):
    """Run full tournament Monte Carlo simulation."""
    global _tournament_pred

    teams = _dataset.get("teams", {})
    if not teams:
        raise HTTPException(400, "No team data loaded. Call /api/refresh-data first.")

    # Build groups from SofaScore group standings
    # For now, fall back to dividing teams alphabetically if groups not parsed
    groups: dict[str, list] = {}
    raw_groups = _dataset.get("groups", [])

    if raw_groups:
        for standing in raw_groups:
            group_name = standing.get("name", "Unknown")
            team_rows = standing.get("rows", [])
            group_teams = []
            for row in team_rows:
                tid = str(row.get("team", {}).get("id", ""))
                if tid in teams:
                    group_teams.append(teams[tid])
            if group_teams:
                groups[group_name] = group_teams
    else:
        # Fallback: 12 groups of 4 from available teams
        team_list = list(teams.values())
        for i in range(0, min(48, len(team_list)), 4):
            g = chr(ord("A") + i // 4)
            groups[f"Group {g}"] = team_list[i : i + 4]

    _tournament_pred = simulate_tournament(
        groups=groups,
        match_predictions=_match_predictions,
        n_runs=n_simulations,
    )

    team_names = {tid: t.name for tid, t in teams.items()}
    questions = tournament_to_polymarket_questions(_tournament_pred, team_names)
    top_bets = get_top_value_bets(questions, min_edge=0.05)

    return {
        "prediction": _tournament_pred.dict(),
        "polymarket_questions": [q.to_dict() for q in questions],
        "top_value_bets": top_bets,
    }


@app.get("/api/polymarket/top-bets")
async def top_bets(min_edge: float = Query(default=0.1)):
    """Return the highest-edge bets across all computed predictions."""
    all_questions = []

    teams = _dataset.get("teams", {})
    matches = {m.id: m for m in _dataset.get("matches", [])}

    for match_id, pred in _match_predictions.items():
        m = matches.get(match_id)
        if not m:
            continue
        home_name = teams.get(m.home_team_id, m.home_team_id).name if isinstance(teams.get(m.home_team_id), object) else str(m.home_team_id)
        away_name = teams.get(m.away_team_id, m.away_team_id).name if isinstance(teams.get(m.away_team_id), object) else str(m.away_team_id)
        all_questions.extend(match_to_polymarket_questions(pred, home_name, away_name))

    if _tournament_pred:
        team_names = {tid: t.name for tid, t in teams.items()}
        all_questions.extend(tournament_to_polymarket_questions(_tournament_pred, team_names))

    return get_top_value_bets(all_questions, min_edge=min_edge, top_n=50)
