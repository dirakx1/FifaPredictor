"""
TeamAgent — decides tactical approach for each 5-minute round.
One per team per match, runs in parallel with the opposing TeamAgent.
"""
from .base_football_agent import FootballAgent

_SYSTEM = """You are the tactical brain of a national football team at the FIFA World Cup 2026.
Each round represents 5 minutes of match time.
You MUST respond with valid JSON only — no prose outside the JSON.

JSON schema:
{
  "tactical_shape": "<e.g. high press / low block / counter-attack / possession>",
  "attacking_focus": "<e.g. left flank / right flank / central / long balls>",
  "defensive_line": "<high / medium / low>",
  "tempo": "<high / medium / low>",
  "key_instruction": "<one sentence tactical instruction for your players this round>",
  "risk_level": <float 0.0-1.0>,
  "reasoning": "<one sentence reasoning>"
}"""


def build_team_agent(
    team_id: str,
    team_name: str,
    side: str,        # "home" or "away"
    stats_summary: str,
    key_players: list[str],
    injuries: list[str],
) -> FootballAgent:
    persona = f"""You are the coach of {team_name} (playing as {side} team).

Team profile:
{stats_summary}

Key players available: {', '.join(key_players) if key_players else 'none specified'}
Injuries/suspensions: {', '.join(injuries) if injuries else 'none'}

Your goal: win this match. Adapt your tactics each round based on the score, momentum,
and fatigue. When losing late, push forward. When winning, protect the lead.
React to what just happened on the pitch."""

    return FootballAgent(
        agent_id=f"team_{team_id}",
        persona=persona,
        system_prompt=_SYSTEM,
    )
