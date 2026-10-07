"""
Experiment 1 -- pre-match Dixon-Coles baseline, 2024/25 walk-forward backtest.

Fits the goals- and xG-target Dixon-Coles models before every 2024/25
matchweek and scores 1X2 predictions (log-loss, RPS, accuracy) against
uniform and historical-base-rate forecasts.

Usage:
    python -m src.experiments.baseline                    # tuned xi per target (OPTIMAL_XI)
    python -m src.experiments.baseline --xi 0.0019        # original xi (PROJECT_CONTEXT 7.1)
    python -m src.experiments.baseline --predict Arsenal Chelsea
    python -m src.experiments.baseline --refresh          # re-download data first
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from ..data_fetcher import DEFAULT_SEASONS, PROCESSED_DIR, fetch_data, load_matches
from ..evaluation.backtest import baseline_scores, summarise, walk_forward
from ..ingest.config import RESULTS_DIR
from ..model.dixon_coles import DixonColesModel

TEST_SEASON = 2024


def get_matches(refresh: bool) -> pd.DataFrame:
    if refresh or not (PROCESSED_DIR / "matches.csv").exists():
        fetch_data(DEFAULT_SEASONS, refresh=refresh)
    return load_matches()  # applies the analysis cutoff


def show_prediction(matches: pd.DataFrame, home: str, away: str, target: str, xi: float) -> None:
    model = DixonColesModel(xi=xi, target=target).fit(matches)
    p = model.predict_match(home, away)
    print(f"\n{home} vs {away}  (target={target}, trained to {model.ref_date.date()})")
    print(f"  expected: {p.home_xg:.2f} - {p.away_xg:.2f}   rho={model.rho:+.3f}")
    print(f"  1X2: H {p.home_win:.1%}  D {p.draw:.1%}  A {p.away_win:.1%}")
    grid = pd.DataFrame(p.score_matrix, index=pd.Index(range(7), name="home"),
                        columns=pd.Index(range(7), name="away"))
    print((grid * 100).round(1).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["goals", "xg", "both"], default="both")
    ap.add_argument("--xi", type=float, default=None,
                    help="time decay per day; default: tuned value per target (OPTIMAL_XI)")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--predict", nargs=2, metavar=("HOME", "AWAY"))
    args = ap.parse_args()

    matches = get_matches(args.refresh)
    targets = ["goals", "xg"] if args.target == "both" else [args.target]

    if args.predict:
        for t in targets:
            show_prediction(matches, *args.predict, target=t, xi=args.xi)
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    summary = {}
    for target in targets:
        xi = DixonColesModel(xi=args.xi, target=target).xi
        print(f"\nWalk-forward {TEST_SEASON}/{(TEST_SEASON + 1) % 100:02d}, target={target}, xi={xi}")
        preds = walk_forward(matches, TEST_SEASON,
                             model_factory=lambda: DixonColesModel(xi=xi, target=target))
        preds.to_csv(RESULTS_DIR / f"predictions_{target}_xi{xi}.csv", index=False)
        summary[f"dixon_coles_{target}"] = summarise(preds)

    table = pd.concat([pd.DataFrame(summary).T, baseline_scores(matches, TEST_SEASON)])
    print(f"\nOut-of-sample results, {TEST_SEASON}/{(TEST_SEASON + 1) % 100:02d}:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table[["log_loss", "rps", "accuracy"]].replace({np.nan: None}).to_string())


if __name__ == "__main__":
    main()
