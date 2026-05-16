#!/usr/bin/env python3
"""
Live end-to-end test: Argentina vs France, 1 multi-agent simulation.
Run from FifaPredictor root:
    cp .env.example .env   # add your ANTHROPIC_API_KEY
    source .venv/bin/activate
    python scripts/test_live_simulation.py
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

os.environ.setdefault("CLAUDE_MODEL", "claude-haiku-4-5-20251001")  # cheapest model for test

from app.data.models import Team, TeamStats, Player, PlayerStats
from app.core.agents.match_engine import simulate_match_once
from app.predictions.match_predictor import predict_match_async
from app.data.models import Match
from app.polymarket.market_mapper import match_to_polymarket_questions, get_top_value_bets


argentina = Team(
    id="arg", name="Argentina", country_code="ARG",
    confederation="CONMEBOL", coach="Lionel Scaloni",
    key_players=["Lionel Messi", "Julian Alvarez", "Enzo Fernandez"],
    stats=TeamStats(
        fifa_ranking=1, avg_goals_scored=2.4, avg_goals_conceded=0.8,
        avg_possession=58.0, avg_shots_on_target=5.2, clean_sheet_rate=0.55,
        win_rate_last10=0.8, form_string="WWWDW", xg_for=2.1, xg_against=0.9,
        playing_style="possession-based high press, lethal on counter",
    ),
)
france = Team(
    id="fra", name="France", country_code="FRA",
    confederation="UEFA", coach="Didier Deschamps",
    key_players=["Kylian Mbappe", "Antoine Griezmann", "Aurelien Tchouameni"],
    stats=TeamStats(
        fifa_ranking=2, avg_goals_scored=2.1, avg_goals_conceded=1.0,
        avg_possession=55.0, avg_shots_on_target=4.8, clean_sheet_rate=0.45,
        win_rate_last10=0.7, form_string="WWDWW", xg_for=1.9, xg_against=1.1,
        playing_style="structured, fast transitions, individual brilliance",
    ),
)
messi = Player(id="mes", name="Lionel Messi", team_id="arg", position="FWD", age=38,
    stats=PlayerStats(goals=12, assists=8, rating=9.1, xg=10.5, goals_per_90=0.85,
                      shots_on_target=38, key_passes=62))
alvarez = Player(id="alv", name="Julian Alvarez", team_id="arg", position="FWD", age=25,
    stats=PlayerStats(goals=9, assists=5, rating=8.3, xg=7.8, goals_per_90=0.71,
                      shots_on_target=28, key_passes=22))
mbappe = Player(id="mba", name="Kylian Mbappe", team_id="fra", position="FWD", age=27,
    stats=PlayerStats(goals=14, assists=6, rating=8.9, xg=11.2, goals_per_90=0.92,
                      shots_on_target=42, key_passes=31))
griezmann = Player(id="gri", name="Antoine Griezmann", team_id="fra", position="MID", age=35,
    stats=PlayerStats(goals=7, assists=9, rating=8.5, xg=6.1, goals_per_90=0.55,
                      shots_on_target=24, key_passes=58))

match = Match(
    id="arf_final", home_team_id="arg", away_team_id="fra",
    stage="Final", venue="MetLife Stadium", city="New York",
)


async def run_single():
    print("=" * 60)
    print("FIFAPREDICTOR — MULTI-AGENT SWARM TEST")
    print("Argentina vs France  |  FIFA World Cup 2026 Final")
    print("=" * 60)
    print("\n[1/2] Running 1 full 18-round multi-agent simulation...")
    print("      Agents: HomeTeamAgent + AwayTeamAgent + 4 PlayerAgents + RefereeAgent")
    print("      Each round fires agents in parallel via asyncio.gather\n")

    state, log = await simulate_match_once(
        argentina, france,
        [messi, alvarez], [mbappe, griezmann],
    )

    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {state.score_line()}")
    print(f"{'='*60}")
    if state.goals:
        for g in state.goals:
            side = argentina.name if g.team == "home" else france.name
            assist = f" (assist: {g.assist})" if g.assist else ""
            pen = " [pen]" if g.is_penalty else ""
            print(f"  {g.minute:2d}'  {side} — {g.scorer}{assist}{pen}")
    else:
        print("  No goals — goalless draw")

    print(f"\nCards: ARG Y={state.home_yellow_cards} R={state.home_red_cards} | "
          f"FRA Y={state.away_yellow_cards} R={state.away_red_cards}")
    print(f"Final momentum: ARG={state.home_momentum:.2f}  FRA={state.away_momentum:.2f}")

    print(f"\n--- Round-by-round sample ---")
    for rnd in log.rounds[::4]:
        home_t = rnd["home_tactics"].get("tactical_shape", "?")
        away_t = rnd["away_tactics"].get("tactical_shape", "?")
        events = rnd["outcome"].get("events", [])
        reasoning = rnd["outcome"].get("reasoning", "")[:80]
        print(f"  Round {rnd['round']:2d} ({rnd['minute']:2d}'): "
              f"ARG={home_t} | FRA={away_t}")
        if events:
            for e in events:
                print(f"    → {e}")
        if reasoning:
            print(f"    [ref] {reasoning}...")


async def run_multi(n=3):
    print(f"\n[2/2] Running {n}-simulation aggregate (enrich_profiles=False for speed)...\n")
    pred = await predict_match_async(
        match, argentina, france,
        [messi, alvarez], [mbappe, griezmann],
        n_simulations=n,
        enrich_profiles=False,
    )
    print(f"\n--- MatchPrediction after {pred.simulations_run} simulations ---")
    print(f"  Argentina win : {pred.home_win_prob:.0%}")
    print(f"  Draw          : {pred.draw_prob:.0%}")
    print(f"  France win    : {pred.away_win_prob:.0%}")
    print(f"  xG ARG={pred.expected_home_goals:.2f}  xG FRA={pred.expected_away_goals:.2f}")
    print(f"  Over 2.5 goals: {pred.over_2_5_prob:.0%}  BTTS: {pred.btts_prob:.0%}")
    print(f"\n  Top likely scorers:")
    for s in pred.likely_scorers[:4]:
        print(f"    {s['name']:25s}  {s['prob']:.0%}")

    questions = match_to_polymarket_questions(pred, "Argentina", "France")
    top_bets = get_top_value_bets(questions, min_edge=0.05, top_n=8)
    print(f"\n--- Top Polymarket bets (edge ≥ 5%) ---")
    for b in top_bets:
        side = b["recommended_side"]
        prob = b["yes_probability"] if side == "YES" else b["no_probability"]
        print(f"  [{side}] {b['question']}")
        print(f"         prob={prob:.0%}  edge={b['edge']:.0%}")


if __name__ == "__main__":
    asyncio.run(run_single())
    asyncio.run(run_multi(n=3))
    print("\nALL TESTS PASSED")
