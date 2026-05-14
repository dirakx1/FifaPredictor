"""
SofaScore unofficial API scraper.
FIFA World Cup 2026: unique-tournament ID 16, season to be fetched dynamically.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import httpx
from cachetools import TTLCache

SOFASCORE_BASE = "https://api.sofascore.com/api/v1"
WC_TOURNAMENT_ID = 16  # FIFA World Cup

_cache: TTLCache = TTLCache(maxsize=512, ttl=3600)

HEADERS = {
    "User-Agent": os.getenv(
        "SOFASCORE_USER_AGENT",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    ),
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
}


async def _get(client: httpx.AsyncClient, path: str) -> dict:
    cache_key = path
    if cache_key in _cache:
        return _cache[cache_key]
    resp = await client.get(f"{SOFASCORE_BASE}{path}", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    _cache[cache_key] = data
    return data


async def get_wc2026_season_id(client: httpx.AsyncClient) -> int:
    data = await _get(client, f"/unique-tournament/{WC_TOURNAMENT_ID}/seasons")
    # Pick the most recent season (highest id)
    seasons = data.get("seasons", [])
    seasons.sort(key=lambda s: s["id"], reverse=True)
    return seasons[0]["id"]


async def get_tournament_groups(client: httpx.AsyncClient, season_id: int) -> list[dict]:
    """Return group standings for the WC."""
    data = await _get(client, f"/unique-tournament/{WC_TOURNAMENT_ID}/season/{season_id}/standings/total")
    return data.get("standings", [])


async def get_tournament_matches(client: httpx.AsyncClient, season_id: int) -> list[dict]:
    """Return all matches (events) for the WC season."""
    data = await _get(client, f"/unique-tournament/{WC_TOURNAMENT_ID}/season/{season_id}/events")
    return data.get("events", [])


async def get_team_stats(client: httpx.AsyncClient, team_id: int, season_id: int) -> dict:
    try:
        data = await _get(
            client,
            f"/team/{team_id}/unique-tournament/{WC_TOURNAMENT_ID}/season/{season_id}/statistics/overall",
        )
        return data.get("statistics", {})
    except httpx.HTTPStatusError:
        return {}


async def get_team_players(client: httpx.AsyncClient, team_id: int) -> list[dict]:
    data = await _get(client, f"/team/{team_id}/players")
    return data.get("players", [])


async def get_player_stats(client: httpx.AsyncClient, player_id: int, season_id: int) -> dict:
    try:
        data = await _get(
            client,
            f"/player/{player_id}/unique-tournament/{WC_TOURNAMENT_ID}/season/{season_id}/statistics/overall",
        )
        return data.get("statistics", {})
    except httpx.HTTPStatusError:
        return {}


async def get_team_recent_form(client: httpx.AsyncClient, team_id: int) -> list[dict]:
    """Last 10 matches for the team across all competitions."""
    data = await _get(client, f"/team/{team_id}/events/last/0")
    return data.get("events", [])[:10]


async def fetch_all_wc_data(cache_dir: str = "./data/cache") -> dict:
    """
    Top-level function: fetches season, groups, matches, and team stats.
    Saves everything to cache_dir as JSON files.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    result: dict = {}

    async with httpx.AsyncClient() as client:
        season_id = await get_wc2026_season_id(client)
        result["season_id"] = season_id
        print(f"[sofascore] WC 2026 season_id={season_id}")

        groups = await get_tournament_groups(client, season_id)
        result["groups"] = groups

        matches = await get_tournament_matches(client, season_id)
        result["matches"] = matches
        print(f"[sofascore] Found {len(matches)} matches, {len(groups)} group tables")

        # Collect unique team IDs
        team_ids: set[int] = set()
        for event in matches:
            home = event.get("homeTeam", {})
            away = event.get("awayTeam", {})
            if home.get("id"):
                team_ids.add(home["id"])
            if away.get("id"):
                team_ids.add(away["id"])

        # Fetch team stats in batches of 5
        teams_data: dict[int, dict] = {}
        team_ids_list = list(team_ids)
        for i in range(0, len(team_ids_list), 5):
            batch = team_ids_list[i : i + 5]
            tasks = [get_team_stats(client, tid, season_id) for tid in batch]
            stats_list = await asyncio.gather(*tasks)
            for tid, stats in zip(batch, stats_list):
                teams_data[tid] = stats
            await asyncio.sleep(0.5)  # be polite

        result["teams"] = teams_data
        print(f"[sofascore] Fetched stats for {len(teams_data)} teams")

    _save_cache(result, cache_dir)
    return result


def _save_cache(data: dict, cache_dir: str) -> None:
    out = Path(cache_dir) / "wc2026_sofascore.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[sofascore] Cached to {out}")


def load_cache(cache_dir: str = "./data/cache") -> Optional[dict]:
    path = Path(cache_dir) / "wc2026_sofascore.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
