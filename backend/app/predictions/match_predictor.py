"""
Claude-based match predictor.
Builds a rich seed prompt from team/player data + video insights,
runs N simulations, and aggregates into a MatchPrediction.
"""
import json
import os
import random
from collections import defaultdict
from typing import Optional

import anthropic

from ..data.models import (
    Match, Team, Player, MatchPrediction, GoalTimingBucket
)

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

SIMULATION_RUNS = int(os.getenv("SIMULATION_RUNS", "200"))

SYSTEM_PROMPT = """You are an expert football statistician and FIFA World Cup analyst.
You simulate matches using historical data, current form, player statistics, and tactical analysis.
You ALWAYS respond with valid JSON only — no markdown, no explanation outside the JSON."""

MATCH_SIM_SCHEMA = """{
  "home_goals": <integer>,
  "away_goals": <integer>,
  "goals": [
    {
      "minute": <integer 1-90>,
      "team": "home" or "away",
      "scorer": "<player name>",
      "assist": "<player name or null>",
      "is_penalty": <boolean>,
      "is_own_goal": <boolean>
    }
  ],
  "reasoning": "<one sentence explaining the key factor that decided this simulation>"
}"""


def _build_team_context(team: Team, players: list[Player]) -> str:
    top_players = sorted(players, key=lambda p: p.stats.goals, reverse=True)[:5]
    player_lines = "\n".join(
        f"  - {p.name} ({p.position}): {p.stats.goals}g {p.stats.assists}a "
        f"rating={p.stats.rating:.1f} xG={p.stats.xg:.2f}"
        f"{' [INJURED]' if p.is_injured else ''}{' [SUSPENDED]' if p.is_suspended else ''}"
        for p in top_players
    )
    s = team.stats
    return f"""
{team.name} (FIFA rank #{s.fifa_ranking}):
  Form (last 5): {s.form_string}
  Avg goals scored: {s.avg_goals_scored:.2f} | conceded: {s.avg_goals_conceded:.2f}
  xG for: {s.xg_for:.2f} | xG against: {s.xg_against:.2f}
  Possession: {s.avg_possession:.0f}% | Shots on target/game: {s.avg_shots_on_target:.1f}
  Clean sheet rate: {s.clean_sheet_rate:.0%}
  Win rate last 10: {s.win_rate_last10:.0%}
  Style: {s.playing_style}
  Coach: {team.coach}
  Key players:
{player_lines}"""


def _build_simulation_prompt(
    match: Match,
    home_team: Team,
    away_team: Team,
    home_players: list[Player],
    away_players: list[Player],
    home_video_insights: str = "",
    away_video_insights: str = "",
    head_to_head: str = "",
    sim_index: int = 0,
) -> str:
    home_ctx = _build_team_context(home_team, home_players)
    away_ctx = _build_team_context(away_team, away_players)

    video_section = ""
    if home_video_insights:
        video_section += f"\nRecent video analysis — {home_team.name}:\n{home_video_insights}\n"
    if away_video_insights:
        video_section += f"\nRecent video analysis — {away_team.name}:\n{away_video_insights}\n"

    h2h_section = f"\nHead-to-head history:\n{head_to_head}\n" if head_to_head else ""

    return f"""Simulate FIFA World Cup 2026 match #{sim_index + 1}.
Stage: {match.stage} | Venue: {match.venue}, {match.city}

{home_ctx}

{away_ctx}
{video_section}{h2h_section}
Instructions:
- Use the statistics and form to produce a realistic, statistically-grounded simulation.
- Vary results across simulations — not every match ends the same way.
- Injured/suspended players should NOT score or assist.
- Goal minutes must be realistic (clusters around 45+, 90+ for stoppage time, etc.).
- Include 0-6 goals total for 90 minutes; knockouts can add extra time goals.

Respond ONLY with this JSON schema:
{MATCH_SIM_SCHEMA}"""


def _simulate_one(
    match: Match,
    home_team: Team,
    away_team: Team,
    home_players: list[Player],
    away_players: list[Player],
    home_video_insights: str,
    away_video_insights: str,
    head_to_head: str,
    sim_index: int,
) -> Optional[dict]:
    prompt = _build_simulation_prompt(
        match, home_team, away_team,
        home_players, away_players,
        home_video_insights, away_video_insights,
        head_to_head, sim_index,
    )
    try:
        response = _client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        print(f"[predictor] sim {sim_index} failed: {e}")
        return None


def _aggregate_simulations(
    results: list[dict],
    home_team_id: str,
    away_team_id: str,
    match_id: str,
) -> MatchPrediction:
    n = len(results)
    if n == 0:
        raise ValueError("No successful simulations")

    home_wins = draws = away_wins = 0
    total_home_goals = total_away_goals = 0
    over15 = over25 = over35 = 0
    btts = 0
    scorer_counts: defaultdict[str, int] = defaultdict(int)
    scorer_team: dict[str, str] = {}
    timing_buckets = [0] * 7  # 0-15, 16-30, 31-45, 46-60, 61-75, 76-90, ET

    first_goal_before_30 = 0
    first_goal_before_45 = 0
    matches_with_goals = 0

    for r in results:
        hg = r.get("home_goals", 0)
        ag = r.get("away_goals", 0)
        total = hg + ag

        if hg > ag:
            home_wins += 1
        elif hg < ag:
            away_wins += 1
        else:
            draws += 1

        total_home_goals += hg
        total_away_goals += ag

        if total > 1.5:
            over15 += 1
        if total > 2.5:
            over25 += 1
        if total > 3.5:
            over35 += 1
        if hg > 0 and ag > 0:
            btts += 1

        goals = r.get("goals", [])
        first_minute = None
        for g in goals:
            scorer = g.get("scorer", "Unknown")
            team = g.get("team", "home")
            team_id = home_team_id if team == "home" else away_team_id
            scorer_counts[scorer] += 1
            scorer_team[scorer] = team_id

            minute = g.get("minute", 45)
            if minute <= 15:
                timing_buckets[0] += 1
            elif minute <= 30:
                timing_buckets[1] += 1
            elif minute <= 45:
                timing_buckets[2] += 1
            elif minute <= 60:
                timing_buckets[3] += 1
            elif minute <= 75:
                timing_buckets[4] += 1
            elif minute <= 90:
                timing_buckets[5] += 1
            else:
                timing_buckets[6] += 1

            if first_minute is None:
                first_minute = minute

        if first_minute is not None:
            matches_with_goals += 1
            if first_minute <= 30:
                first_goal_before_30 += 1
            if first_minute <= 45:
                first_goal_before_45 += 1

    total_goals_in_sims = sum(timing_buckets)
    timing = GoalTimingBucket(
        min_0_15=timing_buckets[0] / max(1, total_goals_in_sims),
        min_16_30=timing_buckets[1] / max(1, total_goals_in_sims),
        min_31_45=timing_buckets[2] / max(1, total_goals_in_sims),
        min_46_60=timing_buckets[3] / max(1, total_goals_in_sims),
        min_61_75=timing_buckets[4] / max(1, total_goals_in_sims),
        min_76_90=timing_buckets[5] / max(1, total_goals_in_sims),
        extra_time=timing_buckets[6] / max(1, total_goals_in_sims),
    )

    top_scorers = sorted(scorer_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    likely_scorers = [
        {"name": name, "team_id": scorer_team.get(name, ""), "prob": count / n}
        for name, count in top_scorers
    ]

    return MatchPrediction(
        match_id=match_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_win_prob=home_wins / n,
        draw_prob=draws / n,
        away_win_prob=away_wins / n,
        expected_home_goals=total_home_goals / n,
        expected_away_goals=total_away_goals / n,
        over_1_5_prob=over15 / n,
        over_2_5_prob=over25 / n,
        over_3_5_prob=over35 / n,
        btts_prob=btts / n,
        likely_scorers=likely_scorers,
        goal_timing=timing,
        first_goal_before_30_prob=first_goal_before_30 / max(1, matches_with_goals),
        first_goal_before_45_prob=first_goal_before_45 / max(1, matches_with_goals),
        simulations_run=n,
        confidence=min(1.0, n / 100),
    )


def predict_match(
    match: Match,
    home_team: Team,
    away_team: Team,
    home_players: list[Player],
    away_players: list[Player],
    home_video_insights: str = "",
    away_video_insights: str = "",
    head_to_head: str = "",
    n_simulations: int = SIMULATION_RUNS,
) -> MatchPrediction:
    """
    Run n_simulations Claude calls and aggregate into a MatchPrediction.
    Note: For cost/speed, use n_simulations=50-100 for quick runs, 500+ for high confidence.
    """
    results = []
    for i in range(n_simulations):
        result = _simulate_one(
            match, home_team, away_team,
            home_players, away_players,
            home_video_insights, away_video_insights,
            head_to_head, i,
        )
        if result:
            results.append(result)

    return _aggregate_simulations(results, home_team.id, away_team.id, match.id)
