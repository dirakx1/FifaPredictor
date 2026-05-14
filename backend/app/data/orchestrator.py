"""
Orchestrates data fetching from SofaScore + video analysis,
normalizes everything into Team/Player/Match models.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from .models import Team, TeamStats, Player, PlayerStats, Match
from .scrapers.sofascore import fetch_all_wc_data, load_cache
from .scrapers.video_analyzer import get_team_video_insights

CACHE_DIR = os.getenv("CACHE_DIR", "./data/cache")


def _parse_team(team_raw: dict, stats_raw: dict, team_id_str: str) -> Team:
    s = stats_raw
    return Team(
        id=team_id_str,
        name=team_raw.get("name", team_id_str),
        country_code=team_raw.get("nameCode", ""),
        sofascore_id=team_raw.get("id"),
        stats=TeamStats(
            avg_goals_scored=s.get("goalsScored", 0) / max(1, s.get("matchesPlayed", 1)),
            avg_goals_conceded=s.get("goalsConceded", 0) / max(1, s.get("matchesPlayed", 1)),
            avg_possession=s.get("avgBallPossession", 50.0),
            avg_shots_on_target=s.get("onTargetScoringAttempts", 0) / max(1, s.get("matchesPlayed", 1)),
            clean_sheet_rate=s.get("cleanSheet", 0) / max(1, s.get("matchesPlayed", 1)),
            xg_for=s.get("expectedGoals", 0.0),
            xg_against=s.get("expectedGoalsAgainst", 0.0),
        ),
    )


def _parse_match(event: dict) -> Optional[Match]:
    mid = str(event.get("id", ""))
    home = event.get("homeTeam", {})
    away = event.get("awayTeam", {})
    if not mid or not home or not away:
        return None

    round_info = event.get("roundInfo", {})
    tournament_round = round_info.get("name", "Group Stage")

    return Match(
        id=mid,
        home_team_id=str(home.get("id", "")),
        away_team_id=str(away.get("id", "")),
        stage=tournament_round,
        venue=event.get("venue", {}).get("stadium", {}).get("name", ""),
        city=event.get("venue", {}).get("city", {}).get("name", ""),
    )


async def build_dataset(use_video: bool = False) -> dict:
    """
    Full data pipeline. Returns:
    {
        "teams": {team_id: Team},
        "players": {team_id: [Player]},
        "matches": [Match],
        "video_insights": {team_id: str},
    }
    """
    raw = load_cache(CACHE_DIR)
    if not raw:
        print("[orchestrator] Cache miss — fetching from SofaScore...")
        raw = await fetch_all_wc_data(CACHE_DIR)

    # Build team map
    teams: dict[str, Team] = {}
    for event in raw.get("matches", []):
        for side in ("homeTeam", "awayTeam"):
            team_raw = event.get(side, {})
            tid = str(team_raw.get("id", ""))
            if tid and tid not in teams:
                stats_raw = raw.get("teams", {}).get(int(tid), {})
                teams[tid] = _parse_team(team_raw, stats_raw, tid)

    # Parse matches
    matches: list[Match] = []
    for event in raw.get("matches", []):
        m = _parse_match(event)
        if m:
            matches.append(m)

    # Gather video insights if requested
    video_insights: dict[str, str] = {}
    if use_video:
        tasks = {tid: get_team_video_insights(t.name, CACHE_DIR) for tid, t in teams.items()}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for tid, result in zip(tasks.keys(), results):
            video_insights[tid] = result if isinstance(result, str) else ""

    return {
        "teams": teams,
        "players": {},  # populated separately via get_team_players
        "matches": matches,
        "video_insights": video_insights,
    }
