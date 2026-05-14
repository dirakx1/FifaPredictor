#!/usr/bin/env python3
"""
CLI script to run full tournament prediction and export Polymarket-ready JSON.
Usage:
    python scripts/run_predictions.py --sims 200 --out predictions_output.json
    python scripts/run_predictions.py --match-id 12345 --sims 50
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv
load_dotenv()

from app.data.orchestrator import build_dataset
from app.predictions.match_predictor import predict_match
from app.predictions.tournament_simulator import simulate_tournament
from app.polymarket.market_mapper import (
    match_to_polymarket_questions,
    tournament_to_polymarket_questions,
    get_top_value_bets,
)


async def main(args):
    print("[FifaPredictor] Loading data...")
    dataset = await build_dataset(use_video=args.video)
    teams = dataset["teams"]
    matches_list = dataset["matches"]
    matches_by_id = {m.id: m for m in matches_list}

    print(f"[FifaPredictor] {len(teams)} teams, {len(matches_list)} matches")

    match_predictions = {}

    if args.match_id:
        # Predict a single match
        if args.match_id not in matches_by_id:
            print(f"[error] Match ID {args.match_id} not found")
            return

        m = matches_by_id[args.match_id]
        home = teams.get(m.home_team_id)
        away = teams.get(m.away_team_id)

        if not home or not away:
            print(f"[error] Could not find teams for match {args.match_id}")
            return

        print(f"[FifaPredictor] Predicting: {home.name} vs {away.name} ({args.sims} simulations)...")
        pred = predict_match(
            match=m, home_team=home, away_team=away,
            home_players=[], away_players=[],
            home_video_insights=dataset["video_insights"].get(m.home_team_id, ""),
            away_video_insights=dataset["video_insights"].get(m.away_team_id, ""),
            n_simulations=args.sims,
        )
        match_predictions[m.id] = pred

        questions = match_to_polymarket_questions(pred, home.name, away.name)
        top_bets = get_top_value_bets(questions, min_edge=0.05)

        output = {
            "type": "match_prediction",
            "match": f"{home.name} vs {away.name}",
            "prediction": pred.dict(),
            "polymarket_questions": [q.to_dict() for q in questions],
            "top_value_bets": top_bets,
        }

    else:
        # Predict all group-stage matches then simulate tournament
        print(f"[FifaPredictor] Running group-stage predictions ({args.sims} sims per match)...")
        for m in matches_list[:args.max_matches]:
            home = teams.get(m.home_team_id)
            away = teams.get(m.away_team_id)
            if not home or not away:
                continue
            print(f"  → {home.name} vs {away.name}")
            pred = predict_match(
                match=m, home_team=home, away_team=away,
                home_players=[], away_players=[],
                home_video_insights=dataset["video_insights"].get(m.home_team_id, ""),
                away_video_insights=dataset["video_insights"].get(m.away_team_id, ""),
                n_simulations=args.sims,
            )
            match_predictions[m.id] = pred

        # Tournament simulation
        print(f"[FifaPredictor] Simulating full tournament ({args.tournament_sims} runs)...")
        raw_groups = dataset.get("groups", [])
        groups = {}
        if raw_groups:
            for standing in raw_groups:
                gname = standing.get("name", "Unknown")
                rows = standing.get("rows", [])
                group_teams = [teams[str(r.get("team", {}).get("id", ""))]
                               for r in rows
                               if str(r.get("team", {}).get("id", "")) in teams]
                if group_teams:
                    groups[gname] = group_teams
        else:
            tlist = list(teams.values())
            for i in range(0, min(48, len(tlist)), 4):
                groups[f"Group {chr(65 + i // 4)}"] = tlist[i : i + 4]

        tournament_pred = simulate_tournament(
            groups=groups,
            match_predictions=match_predictions,
            n_runs=args.tournament_sims,
        )

        team_names = {tid: t.name for tid, t in teams.items()}
        t_questions = tournament_to_polymarket_questions(tournament_pred, team_names)

        all_questions = list(t_questions)
        for m_id, pred in match_predictions.items():
            m = matches_by_id[m_id]
            home_name = teams.get(m.home_team_id, m.home_team_id)
            away_name = teams.get(m.away_team_id, m.away_team_id)
            if hasattr(home_name, "name"):
                home_name = home_name.name
            if hasattr(away_name, "name"):
                away_name = away_name.name
            all_questions.extend(match_to_polymarket_questions(pred, home_name, away_name))

        top_bets = get_top_value_bets(all_questions, min_edge=0.05, top_n=50)

        output = {
            "type": "tournament_prediction",
            "tournament": "FIFA World Cup 2026",
            "simulations_run": args.tournament_sims,
            "tournament_prediction": tournament_pred.dict(),
            "match_predictions": {mid: p.dict() for mid, p in match_predictions.items()},
            "top_value_bets": top_bets,
            "all_polymarket_questions": len(all_questions),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[FifaPredictor] Output saved to {out_path}")

    print("\n=== TOP VALUE BETS FOR POLYMARKET ===")
    for bet in output.get("top_value_bets", [])[:10]:
        side = bet["recommended_side"]
        prob = bet["yes_probability"] if side == "YES" else bet["no_probability"]
        print(f"  [{side}] {bet['question']}")
        print(f"         Probability: {prob:.1%}  |  Edge: {bet['edge']:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FifaPredictor CLI")
    parser.add_argument("--match-id", type=str, default=None, help="Predict a single match by ID")
    parser.add_argument("--sims", type=int, default=50, help="Simulations per match (default 50)")
    parser.add_argument("--tournament-sims", type=int, default=500, help="Tournament Monte Carlo runs")
    parser.add_argument("--max-matches", type=int, default=10, help="Max group-stage matches to predict")
    parser.add_argument("--video", action="store_true", help="Include YouTube video analysis")
    parser.add_argument("--out", type=str, default="data/output/predictions.json", help="Output file path")
    args = parser.parse_args()
    asyncio.run(main(args))
