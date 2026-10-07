"""
Dixon-Coles baseline for EPL: fetch data, run the 2024/25 walk-forward backtest.

Usage:
    python main.py                      # goals- and xG-target models, tuned xi per target
    python main.py --target goals --xi 0.0019
    python main.py --refresh            # re-download season results
    python main.py --predict Arsenal Chelsea
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import baseline_scores, summarise, walk_forward
from src.data_fetcher import DEFAULT_SEASONS, PROCESSED_DIR, fetch_data, load_matches
from src.dixon_coles import DixonColesModel

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TEST_SEASON = 2024


def get_matches(refresh: bool) -> pd.DataFrame:
    if refresh or not (PROCESSED_DIR / "matches.csv").exists():
        matches, _ = fetch_data(DEFAULT_SEASONS, refresh=refresh)
        return matches
    return load_matches()


def show_prediction(matches: pd.DataFrame, home: str, away: str, target: str, xi: float) -> None:
    model = DixonColesModel(xi=xi, target=target).fit(matches)
    p = model.predict_match(home, away)
    print(f"\n{home} vs {away}  (target={target}, trained to {model.ref_date.date()})")
    print(f"  expected: {p.home_xg:.2f} - {p.away_xg:.2f}   rho={model.rho:+.3f}")
    print(f"  1X2: H {p.home_win:.1%}  D {p.draw:.1%}  A {p.away_win:.1%}")
    grid = pd.DataFrame(p.score_matrix, index=pd.Index(range(7), name="home"),
                        columns=pd.Index(range(7), name="away"))
    print((grid * 100).round(1).to_string())


def compare_style(matches: pd.DataFrame, xi: float) -> None:
    """Walk-forward 2024/25: xG Dixon-Coles vs the same model + S_resid style shift.

    Archetypes and S_resid are fitted on 2019/20-2023/24 only. Differences are
    paired by match, so their standard errors are reported alongside.
    """
    from src.archetypes import build_style_model

    print("Fitting archetypes and S_resid (xG Dixon-Coles residuals, 2020-2023) ...")
    tagger, s_resid = build_style_model(target="xg")
    print("S_resid (row = home archetype, col = away archetype):")
    print(s_resid.round(3).to_string(), "\n")

    runs = {
        "baseline_xg": dict(),
        "style_symmetric": dict(style_matrix=s_resid, style_split="symmetric"),
        "style_home_only": dict(style_matrix=s_resid, style_split="home"),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    preds = {}
    for name, kw in runs.items():
        print(f"Walk-forward {name} ...")
        preds[name] = walk_forward(
            matches, TEST_SEASON, verbose=False,
            model_factory=lambda kw=kw: DixonColesModel(xi=xi, target="xg", **kw),
            cluster_fn=tagger.clusters_as_of)
        preds[name].to_csv(RESULTS_DIR / f"predictions_{name}.csv", index=False)

    base = preds["baseline_xg"]
    tagged = base["home_cluster"].notna() & base["away_cluster"].notna()
    adjusted = tagged & (base["home_cluster"] != base["away_cluster"])
    print(f"\n{len(base)} matches: {tagged.sum()} with both teams tagged, "
          f"{adjusted.sum()} adjusted (different archetypes)")

    rows = {"previous baseline (reported)": {"log_loss": 0.9816, "rps": 0.2003}}
    for name, p in preds.items():
        s = summarise(p)
        rows[name] = {"log_loss": s.log_loss, "rps": s.rps, "accuracy": s.accuracy}
        if name != "baseline_xg":
            for metric in ("log_loss", "rps"):
                d = p[metric] - base[metric]
                rows[name][f"d_{metric}"] = d.mean()
                rows[name][f"d_{metric}_se"] = d.std() / np.sqrt(len(d))
    table = pd.DataFrame(rows).T
    print(f"\nOut-of-sample {TEST_SEASON}/{(TEST_SEASON + 1) % 100:02d} "
          f"(d_* = style minus baseline, negative is better):")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(table.replace({np.nan: None}).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["goals", "xg", "both"], default="both")
    ap.add_argument("--xi", type=float, default=None,
                    help="time decay per day; default: tuned value per target (OPTIMAL_XI)")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--predict", nargs=2, metavar=("HOME", "AWAY"))
    ap.add_argument("--style", action="store_true",
                    help="compare xG baseline vs archetype S_resid-adjusted model")
    args = ap.parse_args()

    matches = get_matches(args.refresh)
    if args.style:
        compare_style(matches, args.xi)
        return
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
