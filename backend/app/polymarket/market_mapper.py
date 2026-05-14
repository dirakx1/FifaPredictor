"""
Maps FifaPredictor predictions to Polymarket-compatible prediction market questions.
Each market is a binary YES/NO with a probability and recommended bet side.
"""
from dataclasses import dataclass, field
from typing import Optional

from ..data.models import MatchPrediction, TournamentPrediction


@dataclass
class PolymarketQuestion:
    """A single binary prediction market question."""
    question: str
    category: str        # "match", "tournament", "top_scorer", "goal_timing"
    yes_probability: float
    no_probability: float
    recommended_side: str  # "YES" or "NO"
    edge: float            # abs(prob - 0.5) — higher = more confident
    match_id: Optional[str] = None
    team_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "category": self.category,
            "yes_probability": round(self.yes_probability, 4),
            "no_probability": round(self.no_probability, 4),
            "recommended_side": self.recommended_side,
            "edge": round(self.edge, 4),
            "match_id": self.match_id,
            "team_ids": self.team_ids,
            "metadata": self.metadata,
        }


def _make_question(
    question: str,
    category: str,
    yes_prob: float,
    match_id: Optional[str] = None,
    team_ids: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
) -> PolymarketQuestion:
    yes_prob = max(0.01, min(0.99, yes_prob))
    no_prob = 1.0 - yes_prob
    edge = abs(yes_prob - 0.5)
    return PolymarketQuestion(
        question=question,
        category=category,
        yes_probability=yes_prob,
        no_probability=no_prob,
        recommended_side="YES" if yes_prob > 0.5 else "NO",
        edge=edge,
        match_id=match_id,
        team_ids=team_ids or [],
        metadata=metadata or {},
    )


def match_to_polymarket_questions(
    pred: MatchPrediction,
    home_name: str,
    away_name: str,
) -> list[PolymarketQuestion]:
    """Generate all Polymarket-style questions for a single match."""
    questions: list[PolymarketQuestion] = []
    mid = pred.match_id
    tids = [pred.home_team_id, pred.away_team_id]

    # --- Outcome ---
    questions.append(_make_question(
        f"Will {home_name} win vs {away_name}?",
        "match_outcome", pred.home_win_prob, mid, tids,
        {"type": "winner", "team": home_name},
    ))
    questions.append(_make_question(
        f"Will {away_name} win vs {home_name}?",
        "match_outcome", pred.away_win_prob, mid, tids,
        {"type": "winner", "team": away_name},
    ))
    questions.append(_make_question(
        f"Will {home_name} vs {away_name} end in a draw?",
        "match_outcome", pred.draw_prob, mid, tids,
        {"type": "draw"},
    ))

    # --- Goals totals ---
    questions.append(_make_question(
        f"Will {home_name} vs {away_name} have over 1.5 goals?",
        "goals_total", pred.over_1_5_prob, mid, tids,
        {"type": "over_under", "line": 1.5},
    ))
    questions.append(_make_question(
        f"Will {home_name} vs {away_name} have over 2.5 goals?",
        "goals_total", pred.over_2_5_prob, mid, tids,
        {"type": "over_under", "line": 2.5},
    ))
    questions.append(_make_question(
        f"Will {home_name} vs {away_name} have over 3.5 goals?",
        "goals_total", pred.over_3_5_prob, mid, tids,
        {"type": "over_under", "line": 3.5},
    ))

    # --- Both teams to score ---
    questions.append(_make_question(
        f"Will both {home_name} and {away_name} score?",
        "btts", pred.btts_prob, mid, tids,
        {"type": "btts"},
    ))

    # --- Team-specific ---
    home_clean_sheet = 1.0 - (pred.away_win_prob + pred.btts_prob * 0.5)
    home_clean_sheet = max(0.01, min(0.99, home_clean_sheet))
    questions.append(_make_question(
        f"Will {home_name} keep a clean sheet vs {away_name}?",
        "clean_sheet", home_clean_sheet, mid, tids,
        {"type": "clean_sheet", "team": home_name},
    ))

    # --- Goal timing ---
    t = pred.goal_timing
    questions.append(_make_question(
        f"Will there be a goal in the first 15 minutes of {home_name} vs {away_name}?",
        "goal_timing", t.min_0_15, mid, tids,
        {"type": "goal_timing_bucket", "bucket": "0-15"},
    ))
    questions.append(_make_question(
        f"Will there be a goal in the first 30 minutes of {home_name} vs {away_name}?",
        "goal_timing", pred.first_goal_before_30_prob, mid, tids,
        {"type": "first_goal_before", "minutes": 30},
    ))
    questions.append(_make_question(
        f"Will there be a goal in the second half of {home_name} vs {away_name}?",
        "goal_timing", t.min_46_60 + t.min_61_75 + t.min_76_90, mid, tids,
        {"type": "goal_in_second_half"},
    ))
    questions.append(_make_question(
        f"Will there be a 76-90 min goal in {home_name} vs {away_name}?",
        "goal_timing", t.min_76_90, mid, tids,
        {"type": "goal_timing_bucket", "bucket": "76-90"},
    ))

    # --- Top scorer for this match ---
    for scorer_info in pred.likely_scorers[:3]:
        questions.append(_make_question(
            f"Will {scorer_info['name']} score in {home_name} vs {away_name}?",
            "anytime_scorer", scorer_info["prob"], mid, tids,
            {"type": "anytime_scorer", "player": scorer_info["name"]},
        ))

    return questions


def tournament_to_polymarket_questions(
    pred: TournamentPrediction,
    team_names: dict[str, str],  # team_id -> team_name
) -> list[PolymarketQuestion]:
    """Generate Polymarket questions from tournament prediction."""
    questions: list[PolymarketQuestion] = []

    # --- Champion ---
    for team_id, prob in sorted(pred.champion_probabilities.items(), key=lambda x: x[1], reverse=True)[:10]:
        name = team_names.get(team_id, team_id)
        questions.append(_make_question(
            f"Will {name} win the FIFA World Cup 2026?",
            "tournament_winner", prob, None, [team_id],
            {"type": "champion", "team_id": team_id},
        ))

    # --- Finalist ---
    for team_id, prob in sorted(pred.finalist_probabilities.items(), key=lambda x: x[1], reverse=True)[:8]:
        name = team_names.get(team_id, team_id)
        questions.append(_make_question(
            f"Will {name} reach the final of the FIFA World Cup 2026?",
            "tournament_stage", prob, None, [team_id],
            {"type": "finalist", "team_id": team_id},
        ))

    # --- Semi-finalist ---
    for team_id, prob in sorted(pred.semifinalist_probabilities.items(), key=lambda x: x[1], reverse=True)[:8]:
        name = team_names.get(team_id, team_id)
        questions.append(_make_question(
            f"Will {name} reach the semi-finals of the FIFA World Cup 2026?",
            "tournament_stage", prob, None, [team_id],
            {"type": "semi_finalist", "team_id": team_id},
        ))

    # --- Top scorer ---
    for scorer_pred in pred.top_scorer_predictions[:5]:
        questions.append(_make_question(
            f"Will {scorer_pred.player_name} be the top scorer of the FIFA World Cup 2026?",
            "top_scorer", scorer_pred.win_probability, None, [scorer_pred.team_id],
            {"type": "top_scorer", "player": scorer_pred.player_name,
             "predicted_goals": scorer_pred.predicted_goals},
        ))

    # --- Tournament totals ---
    questions.append(_make_question(
        "Will the FIFA World Cup 2026 average over 2.5 goals per match?",
        "tournament_goals",
        pred.total_goals_distribution.get("over_2_5", 0.5),
        metadata={"type": "tournament_avg_goals", "line": 2.5},
    ))
    questions.append(_make_question(
        "Will the FIFA World Cup 2026 average over 3.5 goals per match?",
        "tournament_goals",
        pred.total_goals_distribution.get("over_3_5", 0.3),
        metadata={"type": "tournament_avg_goals", "line": 3.5},
    ))

    return questions


def get_top_value_bets(
    questions: list[PolymarketQuestion],
    min_edge: float = 0.1,
    top_n: int = 20,
) -> list[dict]:
    """Return the highest-edge bets, sorted by confidence."""
    filtered = [q for q in questions if q.edge >= min_edge]
    filtered.sort(key=lambda q: q.edge, reverse=True)
    return [q.to_dict() for q in filtered[:top_n]]
