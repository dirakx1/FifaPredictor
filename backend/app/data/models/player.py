from pydantic import BaseModel, Field
from typing import Optional


class PlayerStats(BaseModel):
    goals: int = 0
    assists: int = 0
    minutes_played: int = 0
    shots_on_target: int = 0
    dribbles_success_rate: float = 0.0
    key_passes: int = 0
    rating: float = 0.0
    xg: float = 0.0
    goals_per_90: float = 0.0


class Player(BaseModel):
    id: str
    name: str
    team_id: str
    position: str           # GK, DEF, MID, FWD
    sofascore_id: Optional[int] = None
    age: int = 0
    nationality: str = ""
    is_captain: bool = False
    is_injured: bool = False
    is_suspended: bool = False
    stats: PlayerStats = Field(default_factory=PlayerStats)
    recent_form: str = ""   # narrative from video/news analysis
