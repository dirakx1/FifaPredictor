from pydantic import BaseModel, Field
from typing import Optional


class GoalTimingBucket(BaseModel):
    """Probability of a goal being scored in each 15-minute bucket."""
    min_0_15: float = 0.0
    min_16_30: float = 0.0
    min_31_45: float = 0.0
    min_46_60: float = 0.0
    min_61_75: float = 0.0
    min_76_90: float = 0.0
    extra_time: float = 0.0


class MatchPrediction(BaseModel):
    match_id: str
    home_team_id: str
    away_team_id: str

    # Outcome probabilities
    home_win_prob: float
    draw_prob: float
    away_win_prob: float

    # Goals
    expected_home_goals: float
    expected_away_goals: float
    over_1_5_prob: float
    over_2_5_prob: float
    over_3_5_prob: float
    btts_prob: float  # both teams to score

    # Likely scorers with probability
    likely_scorers: list[dict] = Field(default_factory=list)  # [{name, team_id, prob}]

    # Goal timing
    goal_timing: GoalTimingBucket = Field(default_factory=GoalTimingBucket)

    # First goal probability
    first_goal_before_30_prob: float = 0.0
    first_goal_before_45_prob: float = 0.0

    # Simulation metadata
    simulations_run: int = 0
    llm_reasoning: str = ""
    confidence: float = 0.0  # 0-1


class TopScorerPrediction(BaseModel):
    player_name: str
    team_id: str
    predicted_goals: float
    win_probability: float  # probability of being top scorer
    runner_up_probability: float = 0.0


class TournamentPrediction(BaseModel):
    simulations_run: int
    champion_probabilities: dict[str, float]   # team_id -> probability
    finalist_probabilities: dict[str, float]
    semifinalist_probabilities: dict[str, float]
    top_scorer_predictions: list[TopScorerPrediction]
    avg_goals_per_match: float
    total_goals_distribution: dict[str, float]  # "over_X" -> probability
