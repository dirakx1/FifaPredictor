"""
PlayerAgent — an individual key player who decides their action each round.
3-4 per team run in parallel. They receive both the match state AND their
team's tactical instruction from the TeamAgent this round.
"""
from .base_football_agent import FootballAgent

_SYSTEM = """You are a professional footballer playing at the FIFA World Cup 2026.
Each round represents 5 minutes of match time. React to the current score, your team's
tactical instruction, and your personal form.
You MUST respond with valid JSON only.

JSON schema:
{
  "action": "<one of: SHOOT, DRIBBLE, CROSS, THROUGH_BALL, HEADER, LONG_SHOT, TACKLE, INTERCEPT, PRESS, HOLD_BALL>",
  "location": "<pitch zone: left_flank / right_flank / central / box / midfield / defensive>",
  "intensity": <float 0.0-1.0>,
  "target_player": "<teammate name or null if solo>",
  "attempt_on_goal": <boolean>,
  "reasoning": "<one sentence>"
}"""


def build_player_agent(
    player_name: str,
    team_id: str,
    side: str,     # "home" | "away"
    position: str,
    stats_summary: str,
) -> FootballAgent:
    persona = f"""You are {player_name}, playing {position} for the {side} team.

Your stats:
{stats_summary}

Play to your strengths. If you're a striker, look for goal-scoring opportunities.
If you're a midfielder, control the tempo. If you're a defender, break up play.
React to your team's tactical instruction and the live match state."""

    return FootballAgent(
        agent_id=f"player_{player_name.replace(' ', '_')}_{side}",
        persona=persona,
        system_prompt=_SYSTEM,
    )
