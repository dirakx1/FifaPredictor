"""
AgentProfileGenerator — equivalent to MiroFish's OasisProfileGenerator.

Converts Team + Player data (from SofaScore/FIFA) into rich LLM agent personas.
These personas are injected into FootballAgent instances before simulation starts.

MiroFish pattern:
  entities (from Zep graph) → LLM enrichment → OasisAgentProfile
FifaPredictor pattern:
  teams/players (from SofaScore) → LLM enrichment → FootballAgentProfile
"""
import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from ..data.models import Team, Player

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
_client: Optional[anthropic.AsyncAnthropic] = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    return _client


@dataclass
class FootballAgentProfile:
    """
    Enriched persona for a football agent.
    Analogous to MiroFish's OasisAgentProfile.
    """
    agent_id: str
    name: str
    agent_type: str        # "team" | "player" | "referee"
    side: str              # "home" | "away" | "neutral"

    # Core persona — injected into system prompt
    persona: str = ""
    playing_style_summary: str = ""
    psychological_profile: str = ""   # pressure handling, aggression level, etc.
    tactical_tendencies: str = ""     # preferred formations, set pieces, etc.
    threat_vectors: list[str] = field(default_factory=list)   # e.g. ["left flank pace", "set pieces"]
    vulnerability_vectors: list[str] = field(default_factory=list)

    # Metadata
    source_team_id: Optional[str] = None
    source_player_name: Optional[str] = None


async def generate_team_profile(
    team: Team,
    side: str,
    video_insights: str = "",
) -> FootballAgentProfile:
    """
    Use Claude to enrich a team's raw stats into a vivid tactical persona.
    This gives the TeamAgent a much richer behavioral basis than raw numbers.
    """
    s = team.stats
    raw_data = f"""Team: {team.name}
FIFA Rank: #{s.fifa_ranking}
Confederation: {team.confederation}
Coach: {team.coach}
Form (last 5): {s.form_string}
Goals scored/game: {s.avg_goals_scored:.2f}
Goals conceded/game: {s.avg_goals_conceded:.2f}
xG for: {s.xg_for:.2f}  xG against: {s.xg_against:.2f}
Avg possession: {s.avg_possession:.0f}%
Shots on target/game: {s.avg_shots_on_target:.1f}
Clean sheet rate: {s.clean_sheet_rate:.0%}
Win rate last 10: {s.win_rate_last10:.0%}
Playing style: {s.playing_style}
Key players: {', '.join(team.key_players)}
Injuries: {', '.join(team.injuries) if team.injuries else 'none'}
Video insights: {video_insights[:400] if video_insights else 'none'}"""

    prompt = f"""Generate a rich football agent persona for {team.name} based on this data:

{raw_data}

Return JSON:
{{
  "persona": "<2-3 sentence vivid description of how this team plays, their identity, strengths>",
  "playing_style_summary": "<tactical summary: formation preference, press intensity, tempo>",
  "psychological_profile": "<how they handle pressure, big-game mentality, scoring/conceding reaction>",
  "tactical_tendencies": "<set piece strength, preferred build-up patterns, counter-attack threat>",
  "threat_vectors": ["<top 3 ways this team creates danger>"],
  "vulnerability_vectors": ["<top 3 ways this team can be exploited>"]
}}"""

    resp = await _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        system="You are a football scout generating agent personas for a simulation. Return valid JSON only.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    return FootballAgentProfile(
        agent_id=f"team_{team.id}_{side}",
        name=team.name,
        agent_type="team",
        side=side,
        persona=data.get("persona", s.playing_style),
        playing_style_summary=data.get("playing_style_summary", ""),
        psychological_profile=data.get("psychological_profile", ""),
        tactical_tendencies=data.get("tactical_tendencies", ""),
        threat_vectors=data.get("threat_vectors", []),
        vulnerability_vectors=data.get("vulnerability_vectors", []),
        source_team_id=team.id,
    )


async def generate_player_profile(player: Player, team_name: str, side: str) -> FootballAgentProfile:
    """Enrich a player's stats into a behavioral persona."""
    s = player.stats
    raw_data = f"""Player: {player.name}
Team: {team_name}  Position: {player.position}
Goals: {s.goals}  Assists: {s.assists}  Rating: {s.rating:.1f}
xG: {s.xg:.2f}  Goals/90: {s.goals_per_90:.2f}
Shots on target: {s.shots_on_target}
Dribble success: {s.dribbles_success_rate:.0%}
Key passes: {s.key_passes}
Recent form: {player.recent_form or 'not available'}"""

    prompt = f"""Generate a football agent persona for {player.name} based on:

{raw_data}

Return JSON:
{{
  "persona": "<2 sentence description of this player's style and identity>",
  "playing_style_summary": "<how they typically play: movement, decision-making, strengths>",
  "psychological_profile": "<big-game performance, pressure handling, goal-scoring instinct>",
  "threat_vectors": ["<top 2 ways this player creates danger>"],
  "vulnerability_vectors": ["<top 1-2 weaknesses>"]
}}"""

    resp = await _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=400,
        system="You are a football scout. Return valid JSON only.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    return FootballAgentProfile(
        agent_id=f"player_{player.name.replace(' ', '_')}_{side}",
        name=player.name,
        agent_type="player",
        side=side,
        persona=data.get("persona", ""),
        playing_style_summary=data.get("playing_style_summary", ""),
        psychological_profile=data.get("psychological_profile", ""),
        threat_vectors=data.get("threat_vectors", []),
        vulnerability_vectors=data.get("vulnerability_vectors", []),
        source_player_name=player.name,
        source_team_id=player.team_id,
    )


async def generate_all_profiles(
    home_team: Team,
    away_team: Team,
    home_players: list[Player],
    away_players: list[Player],
    home_video: str = "",
    away_video: str = "",
    parallel_count: int = 4,
) -> dict[str, FootballAgentProfile]:
    """
    Generate all agent profiles in parallel batches.
    Mirrors MiroFish's parallel profile generation with progress tracking.
    """
    tasks = [
        generate_team_profile(home_team, "home", home_video),
        generate_team_profile(away_team, "away", away_video),
    ]
    for p in home_players[:3]:
        tasks.append(generate_player_profile(p, home_team.name, "home"))
    for p in away_players[:3]:
        tasks.append(generate_player_profile(p, away_team.name, "away"))

    # Run in batches of parallel_count
    all_profiles: list[FootballAgentProfile] = []
    for i in range(0, len(tasks), parallel_count):
        batch = tasks[i : i + parallel_count]
        results = await asyncio.gather(*batch, return_exceptions=True)
        for r in results:
            if isinstance(r, FootballAgentProfile):
                all_profiles.append(r)
            else:
                print(f"[profile_gen] warning: {r}")

    return {p.agent_id: p for p in all_profiles}
