"""
MatchEngine — the MiroFish-style multi-agent match simulator.

Each match = 18 rounds (5 min each = 90 min).
Per round:
  1. Broadcast match state to all agents
  2. Run HomeTeamAgent + AwayTeamAgent + all PlayerAgents in parallel (asyncio.gather)
  3. Feed all actions to RefereeAgent → RoundOutcome
  4. Update MatchState, log the round
  5. Repeat

This is the core OASIS pattern adapted for football.
"""
import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional

from .match_state import MatchState, RoundOutcome
from .team_agent import build_team_agent
from .player_agent import build_player_agent
from .referee_agent import build_referee_agent, parse_round_outcome
from .base_football_agent import FootballAgent
from ...data.models import Team, Player

ROUNDS_PER_MATCH = 18       # 18 × 5 min = 90 min
MAX_PLAYERS_PER_TEAM = 3    # keep cost reasonable while preserving swarm behavior


@dataclass
class SimulationLog:
    """Full log of one match simulation — equivalent to MiroFish's AgentAction log."""
    rounds: list[dict] = field(default_factory=list)
    final_state: Optional[MatchState] = None

    def add_round(
        self,
        round_num: int,
        home_tactics: dict,
        away_tactics: dict,
        home_player_actions: list[dict],
        away_player_actions: list[dict],
        outcome: RoundOutcome,
    ) -> None:
        self.rounds.append({
            "round": round_num,
            "minute": round_num * 5,
            "home_tactics": home_tactics,
            "away_tactics": away_tactics,
            "home_player_actions": home_player_actions,
            "away_player_actions": away_player_actions,
            "outcome": {
                "goals": [vars(g) for g in outcome.goals],
                "home_yellow_cards": outcome.home_yellow_cards,
                "away_yellow_cards": outcome.away_yellow_cards,
                "home_red_cards": outcome.home_red_cards,
                "away_red_cards": outcome.away_red_cards,
                "events": outcome.narrative_events,
                "reasoning": outcome.referee_reasoning,
            },
        })


def _build_referee_prompt(
    state_broadcast: str,
    home_tactics: dict,
    away_tactics: dict,
    home_actions: list[dict],
    away_actions: list[dict],
) -> str:
    return f"""{state_broadcast}

HOME TEAM TACTICS THIS ROUND:
{json.dumps(home_tactics, indent=2)}

AWAY TEAM TACTICS THIS ROUND:
{json.dumps(away_tactics, indent=2)}

HOME PLAYER ACTIONS:
{json.dumps(home_actions, indent=2)}

AWAY PLAYER ACTIONS:
{json.dumps(away_actions, indent=2)}

Based on all of the above, determine what actually happened in this 5-minute period.
Remember: goals are rare — only award one if the attacking actions genuinely warrant it."""


async def _run_agents_in_parallel(
    agents: list[FootballAgent],
    state_broadcast: str,
    extra: str = "",
) -> list[dict]:
    """Run a list of agents concurrently — the core OASIS parallel-act pattern."""
    tasks = [agent.act(state_broadcast, extra) for agent in agents]
    return await asyncio.gather(*tasks)


async def simulate_match_once(
    home_team: Team,
    away_team: Team,
    home_players: list[Player],
    away_players: list[Player],
    home_video_insights: str = "",
    away_video_insights: str = "",
) -> tuple[MatchState, SimulationLog]:
    """
    Run one complete multi-agent match simulation.
    Returns (final_state, full_log).
    """
    # --- Build agent profiles from team data ---
    home_stats = _team_stats_summary(home_team, home_video_insights)
    away_stats = _team_stats_summary(away_team, away_video_insights)

    home_team_agent = build_team_agent(
        home_team.id, home_team.name, "home",
        home_stats, home_team.key_players, home_team.injuries,
    )
    away_team_agent = build_team_agent(
        away_team.id, away_team.name, "away",
        away_stats, away_team.key_players, away_team.injuries,
    )

    home_player_agents = [
        build_player_agent(
            p.name, home_team.id, "home", p.position,
            _player_stats_summary(p),
        )
        for p in home_players[:MAX_PLAYERS_PER_TEAM]
        if not p.is_injured and not p.is_suspended
    ]
    away_player_agents = [
        build_player_agent(
            p.name, away_team.id, "away", p.position,
            _player_stats_summary(p),
        )
        for p in away_players[:MAX_PLAYERS_PER_TEAM]
        if not p.is_injured and not p.is_suspended
    ]

    referee = build_referee_agent()

    # --- Initialize match state ---
    state = MatchState(
        round_num=1,
        match_minute=0,
        home_team=home_team.name,
        away_team=away_team.name,
        home_momentum=0.5 + (0.55 - away_team.stats.win_rate_last10) * 0.2,
        away_momentum=0.5 + (0.55 - home_team.stats.win_rate_last10) * 0.2,
    )
    log = SimulationLog()

    # --- Main simulation loop ---
    for round_num in range(1, ROUNDS_PER_MATCH + 1):
        state.round_num = round_num
        state.match_minute = round_num * 5
        broadcast = state.to_broadcast()

        # Step 1: Team agents decide tactics in parallel
        home_tactics, away_tactics = await asyncio.gather(
            home_team_agent.act(broadcast),
            away_team_agent.act(broadcast),
        )

        # Step 2: Player agents act in parallel, given team tactics as extra context
        home_tactic_hint = f"Your team's instruction this round: {home_tactics.get('key_instruction', '')}"
        away_tactic_hint = f"Your team's instruction this round: {away_tactics.get('key_instruction', '')}"

        home_actions_list, away_actions_list = [], []
        if home_player_agents or away_player_agents:
            all_player_agents = (
                [(a, home_tactic_hint) for a in home_player_agents]
                + [(a, away_tactic_hint) for a in away_player_agents]
            )
            results = await asyncio.gather(
                *[agent.act(broadcast, hint) for agent, hint in all_player_agents]
            )
            home_actions_list = list(results[:len(home_player_agents)])
            away_actions_list = list(results[len(home_player_agents):])

        # Step 3: Referee evaluates all actions → round outcome
        referee_prompt = _build_referee_prompt(
            broadcast, home_tactics, away_tactics,
            home_actions_list, away_actions_list,
        )
        raw_outcome = await referee.act(referee_prompt)
        outcome = parse_round_outcome(raw_outcome, round_num)

        # Step 4: Update state and log
        state.apply_round_outcome(outcome)
        log.add_round(
            round_num, home_tactics, away_tactics,
            home_actions_list, away_actions_list, outcome,
        )

    log.final_state = state
    return state, log


async def simulate_match_n_times(
    home_team: Team,
    away_team: Team,
    home_players: list[Player],
    away_players: list[Player],
    home_video_insights: str = "",
    away_video_insights: str = "",
    n: int = 10,
) -> list[tuple[MatchState, SimulationLog]]:
    """
    Run N independent match simulations sequentially.
    (Parallel across matches would multiply costs — run sequentially instead.)
    """
    results = []
    for i in range(n):
        print(f"  [engine] simulation {i+1}/{n}: {home_team.name} vs {away_team.name}")
        state, sim_log = await simulate_match_once(
            home_team, away_team, home_players, away_players,
            home_video_insights, away_video_insights,
        )
        results.append((state, sim_log))
    return results


# ---- helpers ----------------------------------------------------------------

def _team_stats_summary(team: Team, video_insights: str = "") -> str:
    s = team.stats
    lines = [
        f"FIFA rank: #{s.fifa_ranking}",
        f"Form (last 5): {s.form_string}",
        f"Goals scored/game: {s.avg_goals_scored:.2f}  Goals conceded/game: {s.avg_goals_conceded:.2f}",
        f"xG for: {s.xg_for:.2f}  xG against: {s.xg_against:.2f}",
        f"Avg possession: {s.avg_possession:.0f}%  Shots on target/game: {s.avg_shots_on_target:.1f}",
        f"Clean sheet rate: {s.clean_sheet_rate:.0%}",
        f"Win rate last 10: {s.win_rate_last10:.0%}",
        f"Playing style: {s.playing_style}",
    ]
    if video_insights:
        lines.append(f"Recent form (video analysis): {video_insights[:300]}")
    return "\n".join(lines)


def _player_stats_summary(player: Player) -> str:
    s = player.stats
    return (
        f"Goals: {s.goals}  Assists: {s.assists}  Rating: {s.rating:.1f}  "
        f"xG: {s.xg:.2f}  Goals/90: {s.goals_per_90:.2f}  "
        f"Shots on target: {s.shots_on_target}  Key passes: {s.key_passes}"
    )
