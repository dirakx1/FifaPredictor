"""
MatchState is the shared broadcast bus between all agents in a round.
Equivalent to MiroFish's OASIS "platform" that all agents observe.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GoalRecord:
    round_num: int
    minute: int
    team: str           # "home" | "away"
    scorer: str
    assist: Optional[str] = None
    is_penalty: bool = False
    is_own_goal: bool = False


@dataclass
class MatchState:
    """
    The live match state broadcast to all agents at the start of each round.
    Agents receive this as context and must react to it.
    """
    round_num: int          # 1-18 (each round = 5 match minutes)
    match_minute: int       # 5, 10, 15 … 90
    home_team: str
    away_team: str
    home_goals: int = 0
    away_goals: int = 0
    goals: list[GoalRecord] = field(default_factory=list)

    # Derived pressure signals — agents use these to adjust behavior
    home_momentum: float = 0.5   # 0-1, updated after each round
    away_momentum: float = 0.5
    home_fatigue: float = 0.0    # 0-1, increases over rounds
    away_fatigue: float = 0.0
    home_pressure: float = 0.5   # tactical pressure level
    away_pressure: float = 0.5

    home_yellow_cards: int = 0
    away_yellow_cards: int = 0
    home_red_cards: int = 0
    away_red_cards: int = 0

    # Events from the previous round (visible to all agents as "news feed")
    last_round_events: list[str] = field(default_factory=list)

    def score_line(self) -> str:
        return f"{self.home_team} {self.home_goals} - {self.away_goals} {self.away_team}"

    def to_broadcast(self) -> str:
        """Human-readable state broadcast — fed to every agent as context."""
        goals_text = ""
        for g in self.goals:
            side = self.home_team if g.team == "home" else self.away_team
            assist = f" (assist: {g.assist})" if g.assist else ""
            pen = " [pen]" if g.is_penalty else ""
            og = " [OG]" if g.is_own_goal else ""
            goals_text += f"  {g.minute}' {side} - {g.scorer}{assist}{pen}{og}\n"

        events = "\n".join(f"  • {e}" for e in self.last_round_events) if self.last_round_events else "  (none)"

        return f"""=== MATCH STATE — MINUTE {self.match_minute} ===
Score: {self.score_line()}
Goals:
{goals_text if goals_text else "  (no goals yet)"}
Momentum  : {self.home_team}={self.home_momentum:.2f}  {self.away_team}={self.away_momentum:.2f}
Fatigue   : {self.home_team}={self.home_fatigue:.2f}   {self.away_team}={self.away_fatigue:.2f}
Pressure  : {self.home_team}={self.home_pressure:.2f}  {self.away_team}={self.away_pressure:.2f}
Cards     : {self.home_team} Y={self.home_yellow_cards} R={self.home_red_cards} | \
{self.away_team} Y={self.away_yellow_cards} R={self.away_red_cards}
Last round events:
{events}
"""

    def apply_round_outcome(self, outcome: "RoundOutcome") -> None:
        """Update state from a referee outcome."""
        for g in outcome.goals:
            self.goals.append(g)
            if g.team == "home":
                self.home_goals += 1
            else:
                self.away_goals += 1

        self.home_yellow_cards += outcome.home_yellow_cards
        self.away_yellow_cards += outcome.away_yellow_cards
        self.home_red_cards += outcome.home_red_cards
        self.away_red_cards += outcome.away_red_cards

        # Update momentum based on round events
        if outcome.goals:
            last_team = outcome.goals[-1].team
            if last_team == "home":
                self.home_momentum = min(1.0, self.home_momentum + 0.15)
                self.away_momentum = max(0.0, self.away_momentum - 0.10)
            else:
                self.away_momentum = min(1.0, self.away_momentum + 0.15)
                self.home_momentum = max(0.0, self.home_momentum - 0.10)
        else:
            # Momentum drifts toward 0.5 when nothing happens
            self.home_momentum = 0.5 + (self.home_momentum - 0.5) * 0.85
            self.away_momentum = 0.5 + (self.away_momentum - 0.5) * 0.85

        # Fatigue increases as match progresses
        fatigue_step = 0.04 + (self.round_num * 0.002)
        self.home_fatigue = min(1.0, self.home_fatigue + fatigue_step)
        self.away_fatigue = min(1.0, self.away_fatigue + fatigue_step)

        self.last_round_events = outcome.narrative_events
        self.round_num += 1
        self.match_minute = min(90, self.round_num * 5)


@dataclass
class RoundOutcome:
    """What the RefereeAgent decides happened in this round."""
    goals: list[GoalRecord] = field(default_factory=list)
    home_yellow_cards: int = 0
    away_yellow_cards: int = 0
    home_red_cards: int = 0
    away_red_cards: int = 0
    narrative_events: list[str] = field(default_factory=list)
    referee_reasoning: str = ""
