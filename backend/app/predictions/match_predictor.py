"""
Match predictor — now backed by the proper MiroFish-style multi-agent swarm.

Flow:
  1. AgentProfileGenerator enriches team/player data into behavioral personas
  2. MatchEngine runs a full 18-round multi-agent simulation (TeamAgents +
     PlayerAgents in parallel, RefereeAgent synthesizing outcomes)
  3. N simulations are aggregated into a MatchPrediction with probabilities
"""
import asyncio
import os
from collections import defaultdict

from ..core.agents.match_engine import simulate_match_n_times, SimulationLog
from ..core.agents.match_state import MatchState
from ..data.models import Match, Team, Player, MatchPrediction, GoalTimingBucket
from ..services.agent_profile_generator import generate_all_profiles

DEFAULT_SIMS = int(os.getenv("SIMULATION_RUNS", "10"))


def _aggregate(
    results: list[tuple[MatchState, SimulationLog]],
    home_team_id: str,
    away_team_id: str,
    match_id: str,
) -> MatchPrediction:
    n = len(results)
    if n == 0:
        raise ValueError("No simulation results to aggregate")

    home_wins = draws = away_wins = 0
    total_hg = total_ag = 0
    over15 = over25 = over35 = 0
    btts = 0
    scorer_counts: defaultdict[str, int] = defaultdict(int)
    scorer_team: dict[str, str] = {}
    timing = [0] * 7          # buckets: 0-15, 16-30, 31-45, 46-60, 61-75, 76-90, ET
    total_goals_scored = 0
    first_before_30 = first_before_45 = matches_with_goals = 0

    for state, _ in results:
        hg, ag = state.home_goals, state.away_goals
        total = hg + ag

        if hg > ag:
            home_wins += 1
        elif ag > hg:
            away_wins += 1
        else:
            draws += 1

        total_hg += hg
        total_ag += ag
        if total > 1.5: over15 += 1
        if total > 2.5: over25 += 1
        if total > 3.5: over35 += 1
        if hg > 0 and ag > 0: btts += 1

        first_min = None
        for g in state.goals:
            team_id = home_team_id if g.team == "home" else away_team_id
            scorer_counts[g.scorer] += 1
            scorer_team[g.scorer] = team_id

            m = g.minute
            if m <= 15:   timing[0] += 1
            elif m <= 30: timing[1] += 1
            elif m <= 45: timing[2] += 1
            elif m <= 60: timing[3] += 1
            elif m <= 75: timing[4] += 1
            elif m <= 90: timing[5] += 1
            else:         timing[6] += 1
            total_goals_scored += 1

            if first_min is None:
                first_min = m

        if first_min is not None:
            matches_with_goals += 1
            if first_min <= 30: first_before_30 += 1
            if first_min <= 45: first_before_45 += 1

    tg = max(1, total_goals_scored)
    gt = GoalTimingBucket(
        min_0_15=timing[0] / tg,
        min_16_30=timing[1] / tg,
        min_31_45=timing[2] / tg,
        min_46_60=timing[3] / tg,
        min_61_75=timing[4] / tg,
        min_76_90=timing[5] / tg,
        extra_time=timing[6] / tg,
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
        expected_home_goals=total_hg / n,
        expected_away_goals=total_ag / n,
        over_1_5_prob=over15 / n,
        over_2_5_prob=over25 / n,
        over_3_5_prob=over35 / n,
        btts_prob=btts / n,
        likely_scorers=likely_scorers,
        goal_timing=gt,
        first_goal_before_30_prob=first_before_30 / max(1, matches_with_goals),
        first_goal_before_45_prob=first_before_45 / max(1, matches_with_goals),
        simulations_run=n,
        confidence=min(1.0, n / 10),
    )


async def predict_match_async(
    match: Match,
    home_team: Team,
    away_team: Team,
    home_players: list[Player],
    away_players: list[Player],
    home_video_insights: str = "",
    away_video_insights: str = "",
    n_simulations: int = DEFAULT_SIMS,
    enrich_profiles: bool = True,
) -> MatchPrediction:
    """
    Full pipeline:
      1. (Optional) LLM-enrich agent personas via AgentProfileGenerator
      2. Run N multi-agent match simulations
      3. Aggregate into MatchPrediction
    """
    if enrich_profiles:
        print(f"  [predictor] generating agent profiles for {home_team.name} vs {away_team.name}...")
        profiles = await generate_all_profiles(
            home_team, away_team,
            home_players, away_players,
            home_video_insights, away_video_insights,
        )
        # Inject enriched personas back into the team/player objects
        home_profile = profiles.get(f"team_{home_team.id}_home")
        away_profile = profiles.get(f"team_{away_team.id}_away")
        if home_profile:
            home_team.stats.playing_style = home_profile.playing_style_summary or home_team.stats.playing_style
        if away_profile:
            away_team.stats.playing_style = away_profile.playing_style_summary or away_team.stats.playing_style
        for p in home_players:
            key = f"player_{p.name.replace(' ', '_')}_home"
            if key in profiles:
                p.recent_form = profiles[key].psychological_profile
        for p in away_players:
            key = f"player_{p.name.replace(' ', '_')}_away"
            if key in profiles:
                p.recent_form = profiles[key].psychological_profile

    results = await simulate_match_n_times(
        home_team, away_team,
        home_players, away_players,
        home_video_insights, away_video_insights,
        n=n_simulations,
    )
    return _aggregate(results, home_team.id, away_team.id, match.id)


def predict_match(
    match: Match,
    home_team: Team,
    away_team: Team,
    home_players: list[Player],
    away_players: list[Player],
    home_video_insights: str = "",
    away_video_insights: str = "",
    n_simulations: int = DEFAULT_SIMS,
    enrich_profiles: bool = True,
) -> MatchPrediction:
    """Sync wrapper for use in CLI scripts."""
    return asyncio.run(predict_match_async(
        match, home_team, away_team,
        home_players, away_players,
        home_video_insights, away_video_insights,
        n_simulations, enrich_profiles,
    ))
