from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class GoalEvent(BaseModel):
    minute: int
    team_id: str
    scorer: str
    assist: Optional[str] = None
    is_penalty: bool = False
    is_own_goal: bool = False


class Match(BaseModel):
    id: str
    home_team_id: str
    away_team_id: str
    stage: str          # "Group A", "Round of 32", "QF", "SF", "Final"
    venue: str = ""
    city: str = ""
    scheduled_at: Optional[datetime] = None
    group: Optional[str] = None


class MatchResult(BaseModel):
    match_id: str
    home_goals: int
    away_goals: int
    goals: list[GoalEvent] = Field(default_factory=list)
    went_to_extra_time: bool = False
    went_to_penalties: bool = False
    penalty_winner_team_id: Optional[str] = None
    winner_team_id: Optional[str] = None  # None for draw (group stage only)
