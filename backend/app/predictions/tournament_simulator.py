"""
Full FIFA World Cup 2026 tournament Monte Carlo simulator.
WC 2026 format: 48 teams, 12 groups of 4, top 2 + 8 best 3rd-place advance (32 teams).
Then: Round of 32 → Round of 16 → QF → SF → Final.
"""
import random
from collections import defaultdict
from typing import Optional

from ..data.models import (
    Team, Player, Match, MatchPrediction,
    TournamentPrediction, TopScorerPrediction,
)
from .match_predictor import predict_match


class GroupStanding:
    def __init__(self, team: Team):
        self.team = team
        self.pts = 0
        self.gd = 0
        self.gf = 0
        self.ga = 0

    def __lt__(self, other: "GroupStanding") -> bool:
        if self.pts != other.pts:
            return self.pts > other.pts
        if self.gd != other.gd:
            return self.gd > other.gd
        return self.gf > other.gf


def _sample_result(pred: MatchPrediction) -> tuple[int, int]:
    """Sample one match result from prediction probabilities using Poisson."""
    import numpy as np
    hg = max(0, int(np.random.poisson(pred.expected_home_goals)))
    ag = max(0, int(np.random.poisson(pred.expected_away_goals)))
    return hg, ag


def _knockout_winner(
    pred: MatchPrediction,
    home_team_id: str,
    away_team_id: str,
) -> str:
    """Return winner team_id (no draws in knockout)."""
    hg, ag = _sample_result(pred)
    if hg > ag:
        return home_team_id
    elif ag > hg:
        return away_team_id
    else:
        # Penalty shootout: use win_prob as tiebreaker
        r = random.random()
        total = pred.home_win_prob + pred.away_win_prob
        if total == 0:
            return random.choice([home_team_id, away_team_id])
        return home_team_id if r < pred.home_win_prob / total else away_team_id


def simulate_group_stage(
    groups: dict[str, list[Team]],
    match_predictions: dict[str, MatchPrediction],
) -> tuple[list[Team], list[GroupStanding]]:
    """
    Returns (qualified_teams, all_standings) where qualified_teams is the 32 that advance.
    match_predictions keyed by f"{home_id}_vs_{away_id}".
    """
    group_standings: dict[str, list[GroupStanding]] = {}
    all_third_place: list[GroupStanding] = []

    for group_name, teams in groups.items():
        standings = {t.id: GroupStanding(t) for t in teams}

        # Round robin (each pair plays once)
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                home = teams[i]
                away = teams[j]
                key = f"{home.id}_vs_{away.id}"
                rev_key = f"{away.id}_vs_{home.id}"
                pred = match_predictions.get(key) or match_predictions.get(rev_key)

                if pred:
                    hg, ag = _sample_result(pred)
                else:
                    # Fallback: equal probability
                    hg = random.randint(0, 3)
                    ag = random.randint(0, 3)

                standings[home.id].gf += hg
                standings[home.id].ga += ag
                standings[home.id].gd += hg - ag
                standings[away.id].gf += ag
                standings[away.id].ga += hg
                standings[away.id].gd += ag - hg

                if hg > ag:
                    standings[home.id].pts += 3
                elif hg < ag:
                    standings[away.id].pts += 3
                else:
                    standings[home.id].pts += 1
                    standings[away.id].pts += 1

        sorted_standings = sorted(standings.values())
        group_standings[group_name] = sorted_standings

        # Top 2 qualify directly
        all_third_place.append(sorted_standings[2])

    # Best 8 third-place teams also qualify (WC 2026 rule)
    best_thirds = sorted(all_third_place)[:8]
    best_third_ids = {s.team.id for s in best_thirds}

    qualified: list[Team] = []
    all_standings: list[GroupStanding] = []
    for group_name, sorted_standings in group_standings.items():
        qualified.append(sorted_standings[0].team)
        qualified.append(sorted_standings[1].team)
        if sorted_standings[2].team.id in best_third_ids:
            qualified.append(sorted_standings[2].team)
        all_standings.extend(sorted_standings)

    return qualified, all_standings


def simulate_knockout_round(
    teams: list[Team],
    match_predictions: dict[str, MatchPrediction],
    stage_name: str,
) -> list[Team]:
    """Pair teams sequentially and return winners."""
    winners = []
    for i in range(0, len(teams), 2):
        home = teams[i]
        away = teams[i + 1] if i + 1 < len(teams) else teams[i]
        key = f"{home.id}_vs_{away.id}"
        rev_key = f"{away.id}_vs_{home.id}"
        pred = match_predictions.get(key) or match_predictions.get(rev_key)

        if pred:
            winner_id = _knockout_winner(pred, home.id, away.id)
        else:
            winner_id = random.choice([home.id, away.id])

        winner = home if winner_id == home.id else away
        winners.append(winner)
    return winners


def simulate_tournament(
    groups: dict[str, list[Team]],
    match_predictions: dict[str, MatchPrediction],
    n_runs: int = 1000,
    players_by_team: Optional[dict[str, list[Player]]] = None,
) -> TournamentPrediction:
    """
    Monte Carlo: simulate the full tournament n_runs times.
    Returns aggregated TournamentPrediction.
    """
    champion_counts: defaultdict[str, int] = defaultdict(int)
    finalist_counts: defaultdict[str, int] = defaultdict(int)
    semifinalist_counts: defaultdict[str, int] = defaultdict(int)
    scorer_goals: defaultdict[str, list[float]] = defaultdict(list)
    scorer_team: dict[str, str] = {}
    total_goals_all: list[float] = []

    for _ in range(n_runs):
        qualified, _ = simulate_group_stage(groups, match_predictions)

        # Shuffle qualified to randomize bracket
        random.shuffle(qualified)

        bracket = qualified[:]
        while len(bracket) > 2:
            bracket = simulate_knockout_round(bracket, match_predictions, "knockout")

        if len(bracket) == 2:
            finalist_counts[bracket[0].id] += 1
            finalist_counts[bracket[1].id] += 1

            key = f"{bracket[0].id}_vs_{bracket[1].id}"
            rev_key = f"{bracket[1].id}_vs_{bracket[0].id}"
            pred = match_predictions.get(key) or match_predictions.get(rev_key)
            if pred:
                winner_id = _knockout_winner(pred, bracket[0].id, bracket[1].id)
            else:
                winner_id = random.choice([bracket[0].id, bracket[1].id])
            champion_counts[winner_id] += 1

        for team in bracket[:4]:
            semifinalist_counts[team.id] += 1

    # Aggregate scorer stats from group-stage predictions
    total_goals_in_run: list[float] = []
    for pred in match_predictions.values():
        total_goals_in_run.append(
            pred.expected_home_goals + pred.expected_away_goals
        )
        for scorer_info in pred.likely_scorers:
            name = scorer_info["name"]
            scorer_goals[name].append(scorer_info["prob"] * 7)  # ~7 games if champion
            scorer_team[name] = scorer_info["team_id"]

    avg_goals = sum(total_goals_in_run) / max(1, len(total_goals_in_run))

    top_scorer_preds = []
    scorer_expected = {
        name: sum(goals) / max(1, len(goals))
        for name, goals in scorer_goals.items()
    }
    total_expected = sum(scorer_expected.values()) or 1
    for name, expected in sorted(scorer_expected.items(), key=lambda x: x[1], reverse=True)[:15]:
        top_scorer_preds.append(TopScorerPrediction(
            player_name=name,
            team_id=scorer_team.get(name, ""),
            predicted_goals=round(expected, 2),
            win_probability=min(0.99, expected / total_expected),
        ))

    return TournamentPrediction(
        simulations_run=n_runs,
        champion_probabilities={tid: c / n_runs for tid, c in champion_counts.items()},
        finalist_probabilities={tid: c / n_runs for tid, c in finalist_counts.items()},
        semifinalist_probabilities={tid: c / n_runs for tid, c in semifinalist_counts.items()},
        top_scorer_predictions=top_scorer_preds[:10],
        avg_goals_per_match=round(avg_goals, 2),
        total_goals_distribution={
            "over_2_5": sum(1 for g in total_goals_in_run if g > 2.5) / max(1, len(total_goals_in_run)),
            "over_3_5": sum(1 for g in total_goals_in_run if g > 3.5) / max(1, len(total_goals_in_run)),
        },
    )
