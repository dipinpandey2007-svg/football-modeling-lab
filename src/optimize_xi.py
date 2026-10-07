"""
Grid search for the Dixon-Coles time-decay rate xi on the training seasons.

For each candidate xi, run the matchweek walk-forward on 2020/21 - 2023/24
(2019/20 is the first season of data and has no prior data to fit on, so it
only serves as training history) and score every match out-of-sample with the
Poisson log-loss of the full-time score:

    -log Pois(home_goals; mu) - log Pois(away_goals; lambda)

The 1X2 log-loss is reported alongside for reference. Differences to the
best xi are paired by match, with standard errors, to show how sharply the
optimum is identified. The 2024/25 test season is never touched.

Usage:
    python -m src.optimize_xi                    # both targets, coarse grid
    python -m src.optimize_xi --target xg --step 0.0001
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .cache import cached_walk_forward
from .data_fetcher import load_matches

PREVIOUS_XI = 0.0019  # hand-picked default before this search

VALIDATION_SEASONS = [2020, 2021, 2022, 2023]


def evaluate_xi(matches: pd.DataFrame, xi: float, target: str) -> pd.DataFrame:
    """Per-match out-of-sample losses over the validation seasons for one xi."""
    preds = pd.concat([
        cached_walk_forward(matches, s, target=target, xi=xi) for s in VALIDATION_SEASONS])
    preds["poisson_ll"] = (-poisson.logpmf(preds["home_goals"], preds["pred_home_xg"])
                           - poisson.logpmf(preds["away_goals"], preds["pred_away_xg"]))
    return preds.set_index("match_id")[["poisson_ll", "log_loss", "rps"]]


def grid_search(matches: pd.DataFrame, grid: np.ndarray, target: str) -> pd.DataFrame:
    per_match = {}
    for xi in grid:
        per_match[xi] = evaluate_xi(matches, float(xi), target)
        print(f"  xi={xi:.4f}  poisson={per_match[xi]['poisson_ll'].mean():.5f}  "
              f"1x2={per_match[xi]['log_loss'].mean():.5f}")
    best = min(per_match, key=lambda k: per_match[k]["poisson_ll"].mean())
    rows = {}
    for xi, pm in per_match.items():
        d = pm["poisson_ll"] - per_match[best]["poisson_ll"]
        rows[round(xi, 4)] = {
            "poisson_ll": pm["poisson_ll"].mean(),
            "diff_vs_best": d.mean(),
            "diff_se": d.std() / np.sqrt(len(d)),
            "logloss_1x2": pm["log_loss"].mean(),
            "rps": pm["rps"].mean(),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "xi"
    out.attrs["n_matches"] = len(per_match[best])
    return out


def run(targets: list[str], start: float, stop: float, step: float) -> dict[str, float]:
    matches = load_matches()
    grid = np.round(np.arange(start, stop + step / 2, step), 6)
    best = {}
    for target in targets:
        print(f"\nTarget={target}: walk-forward over {VALIDATION_SEASONS}, {len(grid)} xi values")
        res = grid_search(matches, grid, target)
        best[target] = float(res["poisson_ll"].idxmin())
        print(f"\n=== target={target}: out-of-sample full-time Poisson log-loss "
              f"({res.attrs['n_matches']} matches) ===")
        with pd.option_context("display.float_format", "{:.5f}".format):
            print(res.to_string())
        print(f"Best xi = {best[target]:.4f}")
        if np.isclose(res.index, PREVIOUS_XI).any():
            ref = res.loc[res.index[np.isclose(res.index, PREVIOUS_XI)][0]]
            print(f"  previous default {PREVIOUS_XI} is {ref['diff_vs_best']:+.5f} "
                  f"(se {ref['diff_se']:.5f}) behind it")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["goals", "xg", "both"], default="both")
    ap.add_argument("--start", type=float, default=0.0005)
    ap.add_argument("--stop", type=float, default=0.0050)
    ap.add_argument("--step", type=float, default=0.0005)
    a = ap.parse_args()
    run(["goals", "xg"] if a.target == "both" else [a.target], a.start, a.stop, a.step)
