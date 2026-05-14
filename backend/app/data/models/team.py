from pydantic import BaseModel, Field
from typing import Optional


class TeamStats(BaseModel):
    fifa_ranking: int = 0
    avg_goals_scored: float = 0.0
    avg_goals_conceded: float = 0.0
    avg_possession: float = 50.0
    avg_shots_on_target: float = 0.0
    clean_sheet_rate: float = 0.0
    win_rate_last10: float = 0.0
    draw_rate_last10: float = 0.0
    form_string: str = ""             # e.g. "WWDLW"
    xg_for: float = 0.0
    xg_against: float = 0.0
    pressing_intensity: float = 50.0  # 0-100
    defensive_line: str = "medium"    # high/medium/low
    playing_style: str = ""           # e.g. "possession, high press"


class Team(BaseModel):
    id: str
    name: str
    country_code: str
    sofascore_id: Optional[int] = None
    confederation: str = ""           # UEFA, CONMEBOL, etc.
    coach: str = ""
    stats: TeamStats = Field(default_factory=TeamStats)
    key_players: list[str] = Field(default_factory=list)
    injuries: list[str] = Field(default_factory=list)
    suspended: list[str] = Field(default_factory=list)
