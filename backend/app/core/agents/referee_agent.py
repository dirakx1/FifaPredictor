"""
RefereeAgent — the oracle that evaluates all player and team actions
and decides what actually happened in the round (goals, fouls, cards).

This is the key convergence point: it sees all agent actions and the match
state, then produces the authoritative RoundOutcome.

Analogous to MiroFish's ReportAgent — it synthesizes all agent outputs.
"""
import json

from .base_football_agent import FootballAgent
from .match_state import GoalRecord, RoundOutcome

_SYSTEM = """You are a senior FIFA referee and match analyst at the FIFA World Cup 2026.
Each round you receive: the current match state, both teams' tactical decisions,
and the actions of key players. You decide what ACTUALLY happened in that 5-minute period.

Be realistic: most rounds produce no goals. Goal probability in any 5-min window is ~10-15%.
Vary outcomes — sometimes a dangerous attack leads to a corner, sometimes a goal.
Use the action types, tactical shape, momentum, and fatigue to determine outcomes.

You MUST respond with valid JSON only.

JSON schema:
{
  "goals": [
    {
      "minute": <int 1-90>,
      "team": "home" or "away",
      "scorer": "<player name>",
      "assist": "<player name or null>",
      "is_penalty": <boolean>,
      "is_own_goal": <boolean>
    }
  ],
  "home_yellow_cards": <int 0-2>,
  "away_yellow_cards": <int 0-2>,
  "home_red_cards": <int 0-1>,
  "away_red_cards": <int 0-1>,
  "narrative_events": ["<short event description>", ...],
  "reasoning": "<one paragraph: what key actions led to these outcomes>"
}"""


def build_referee_agent() -> FootballAgent:
    persona = """You are a neutral, experienced FIFA World Cup referee.
You evaluate football actions objectively based on their tactical quality,
player stats, and situational pressure. You enforce the Laws of the Game."""

    return FootballAgent(
        agent_id="referee",
        persona=persona,
        system_prompt=_SYSTEM,
    )


def parse_round_outcome(raw_output: dict, round_num: int) -> RoundOutcome:
    """Convert referee JSON output to a typed RoundOutcome."""
    outcome = RoundOutcome(
        home_yellow_cards=raw_output.get("home_yellow_cards", 0),
        away_yellow_cards=raw_output.get("away_yellow_cards", 0),
        home_red_cards=raw_output.get("home_red_cards", 0),
        away_red_cards=raw_output.get("away_red_cards", 0),
        narrative_events=raw_output.get("narrative_events", []),
        referee_reasoning=raw_output.get("reasoning", ""),
    )
    for g in raw_output.get("goals", []):
        outcome.goals.append(GoalRecord(
            round_num=round_num,
            minute=g.get("minute", round_num * 5),
            team=g.get("team", "home"),
            scorer=g.get("scorer", "Unknown"),
            assist=g.get("assist"),
            is_penalty=g.get("is_penalty", False),
            is_own_goal=g.get("is_own_goal", False),
        ))
    return outcome
