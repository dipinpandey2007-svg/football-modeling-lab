"""
Blind out-of-sample test of the frozen half-time anomaly candidates.

Reads results/ht_anomaly_candidates.json (written by explore_ht_anomalies from
2020/21-2023/24 only) and tests each candidate on 2024/25, 2025/26 and
2026/27-to-date. Nothing is re-estimated except the pre-match Dixon-Coles
ratings, which are refitted walk-forward before each matchweek as usual.

For a team row in a candidate state:
    naive : r_goals * pre-match rate              (frozen r_goals from training)
    model : naive + train_effect_goals            (frozen training effect)

Pass rule (fixed before looking at any result):
    1. the realised 2H goal residual on test data has the same sign as in training
    2. the model improves 2H Poisson log-loss over naive on the triggered rows
       by more than 2 standard errors (SE clustered by match)

Usage:
    python -m src.experiments.ht_anomaly_blind_test
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..data_fetcher import load_matches
from .ht_anomaly_search import OUT_PATH, add_dimensions, in_state
from .ht_leader_rule import poisson_log_loss, test_team_rows

TEST_SEASONS = [2024, 2025, 2026]


def cluster_mean_se(values: pd.Series, match_ids: pd.Series) -> tuple[float, float]:
    """Mean per row with SE clustered by match."""
    per = values.groupby(match_ids).sum()
    n_c = values.groupby(match_ids).size()
    mean = per.sum() / len(values)
    k = len(per)
    se = np.sqrt((((per - n_c * mean) ** 2).sum()) * k / (k - 1)) / len(values) if k > 1 else np.nan
    return float(mean), float(se)


def run() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.4f}".format)
    frozen = json.loads(OUT_PATH.read_text())
    r = frozen["r_goals"]
    print(f"Frozen from {frozen['train_seasons']}: r_goals = {r:.4f}, "
          f"{len(frozen['candidates'])} candidates")
    print("PASS RULE: (1) test residual has the training sign AND "
          "(2) 2H log-loss improvement over naive > 2 SE on triggered rows\n")

    matches = load_matches()
    print(f"Walk-forward pre-match Dixon-Coles for {TEST_SEASONS} ...")
    df = pd.concat([test_team_rows(matches, s) for s in TEST_SEASONS], ignore_index=True)
    df = df.merge(matches[["match_id", "season"]], on="match_id", suffixes=("_x", ""))
    df = add_dimensions(df, matches)
    df["naive"] = r * df["pre_rate"]
    df["resid"] = df["half_2_goals"] - df["naive"]
    df["ll_naive"] = poisson_log_loss(df["half_2_goals"].to_numpy(), df["naive"].to_numpy())
    print(f"{df['match_id'].nunique()} test matches, {len(df)} team rows "
          f"(mean 2H residual across all rows {df['resid'].mean():+.3f})\n")

    verdicts = []
    for c in frozen["candidates"]:
        eff = c["train_effect_goals"]
        mask = in_state(df, c["conditions"])
        sub = df[mask].copy()
        sub["ll_model"] = poisson_log_loss(sub["half_2_goals"].to_numpy(), (sub["naive"] + eff).to_numpy())
        sub["gain"] = sub["ll_naive"] - sub["ll_model"]       # > 0 = model better

        res_m, res_se = cluster_mean_se(sub["resid"], sub["match_id"])
        gain_m, gain_se = cluster_mean_se(sub["gain"], sub["match_id"])
        same_sign = np.sign(res_m) == np.sign(eff)
        significant = gain_m > 2 * gain_se
        passed = bool(same_sign and significant)
        verdicts.append((c["state"], passed))

        print(f"=== Candidate {c['rank']}: {c['state']} ===")
        print(f"  training: {eff:+.3f} goals/team (se {c['train_se']:.3f}, n={c['train_n_matches']})")
        print(f"  test:     {res_m:+.3f} goals/team (se {res_se:.3f}, n={sub['match_id'].nunique()} matches); "
              f"actual 2H goals {sub['half_2_goals'].mean():.3f} vs naive {sub['naive'].mean():.3f}")
        print(f"  2H log-loss per triggered row: naive {sub['ll_naive'].mean():.4f} -> model "
              f"{sub['ll_model'].mean():.4f}; gain {gain_m:+.4f} +/- {gain_se:.4f} "
              f"(z {gain_m / gain_se:+.2f})")
        by_season = sub.groupby("season").agg(
            n=("match_id", "nunique"), actual=("half_2_goals", "mean"), naive=("naive", "mean"),
            resid=("resid", "mean"), ll_gain=("gain", "mean"))
        print(by_season.to_string())
        print(f"  (1) same sign as training: {'yes' if same_sign else 'NO'}   "
              f"(2) gain > 2 SE: {'yes' if significant else 'NO'}   ->  "
              f"{'PASS' if passed else 'FAIL'}\n")

    print("=== FINAL VERDICT ===")
    for state, passed in verdicts:
        print(f"  {'PASS' if passed else 'FAIL'}  {state}")
    n_pass = sum(p for _, p in verdicts)
    print(f"  {n_pass} of {len(verdicts)} structural half-time anomalies survived the blind test.")


if __name__ == "__main__":
    run()
