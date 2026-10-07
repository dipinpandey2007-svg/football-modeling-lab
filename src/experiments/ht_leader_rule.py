"""
Half-time model backtest on 2024/25 (out-of-sample).

At half-time, each team's expected 2nd-half goals are

    naive          : r * pre-match rate                 (mu home, lambda away)
    capped_leader  : naive + delta_L  if the team leads at HT and its HT xG
                     deficit d = opp HT xG - own HT xG is in (0, CAP); else naive
    capped_both    : capped_leader, plus naive + delta_T for the opponent of
                     such a leader (it trails while ahead on xG by d)
    linear_leader  : naive + beta * d for every leading team with d > 0
                     (previous model, kept for comparison)

HT xG uses the soft (probability-weighted) split. Deficits of CAP or more get
no modifier: in training, leaders outplayed by >= 0.5 xG did not underperform
(the Leading & Dominated cell was +0.045 +/- 0.106).

Everything is fitted on 2020/21 - 2023/24 walk-forward predictions only:
  r       = sum(actual 2H goals) / sum(pre-match rate)
  delta_* = mean residual (2H goals - r * pre-match rate) of triggered teams
  beta    = least squares through the origin on leading teams with d > 0
Pre-match rates come from the xG-fitted Dixon-Coles walk-forward with the tuned
time decay (OPTIMAL_XI["xg"]), refitted before every matchweek.

Score: Poisson log-loss of each team's actual 2nd-half goals, summed over the
two teams (= log-loss of the 2H score under independent Poissons); model
differences are paired by match.

Usage:
    python -m src.experiments.ht_leader_rule
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..evaluation.cache import cached_walk_forward
from ..data_fetcher import load_half_time_stats, load_matches
from ..model.dixon_coles import DixonColesModel
from .ht_grid import second_half_share, team_rows

TEST_SEASON = 2024
CAP = 0.5          # leader xG deficits at or beyond this get no modifier
MIN_RATE = 0.02    # floor for adjusted 2nd-half expected goals


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def add_leader_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flag fraudulent leaders (and their opponents) by HT xG deficit."""
    leading, trailing = df["ht_gd"] > 0, df["ht_gd"] < 0
    df["leader_deficit"] = np.where(leading & (df["ht_xgd"] < 0), -df["ht_xgd"], 0.0)
    df["trailer_edge"] = np.where(trailing & (df["ht_xgd"] > 0), df["ht_xgd"], 0.0)
    df["leader_step"] = (df["leader_deficit"] > 0) & (df["leader_deficit"] < CAP)
    df["trailer_step"] = (df["trailer_edge"] > 0) & (df["trailer_edge"] < CAP)
    return df


def test_team_rows(matches: pd.DataFrame, season: int = TEST_SEASON,
                   xi: float | None = None) -> pd.DataFrame:
    """Test-season team rows: walk-forward pre-match rate, HT state, 2H goals."""
    preds = cached_walk_forward(matches, season, target="xg", xi=xi)
    ht = load_half_time_stats().merge(matches[["match_id", "season"]], on="match_id")
    df = ht[ht["season"] == season].merge(
        preds[["match_id", "pred_home_xg", "pred_away_xg"]], on="match_id", how="inner")
    opp = df[["match_id", "h_a", "half_1_goals", "half_1_xg_soft"]].assign(
        h_a=lambda d: d["h_a"].map({"h": "a", "a": "h"}))
    df = df.merge(opp, on=["match_id", "h_a"], suffixes=("", "_opp"))
    df["pre_rate"] = np.where(df["h_a"] == "h", df["pred_home_xg"], df["pred_away_xg"])
    df["ht_gd"] = df["half_1_goals"] - df["half_1_goals_opp"]
    df["ht_xgd"] = df["half_1_xg_soft"] - df["half_1_xg_soft_opp"]
    return add_leader_features(df)


# ---------------------------------------------------------------------------
# Fitting (training seasons only)
# ---------------------------------------------------------------------------

def _mean_se(x: np.ndarray) -> tuple[float, float, int]:
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), int(len(x))


def fit_training() -> dict:
    train = add_leader_features(team_rows("xg", xg_split="soft"))
    r = train["half_2_goals"].sum() / train["pre_rate"].sum()
    resid = (train["half_2_goals"] - r * train["pre_rate"]).to_numpy()

    d = train["leader_deficit"].to_numpy()
    on = d > 0
    beta = (d[on] @ resid[on]) / (d[on] @ d[on])
    beta_se = np.sqrt(((resid[on] - beta * d[on]) ** 2).sum() / (on.sum() - 1) / (d[on] @ d[on]))

    beyond = (d >= CAP)
    ht = load_half_time_stats().merge(load_matches()[["match_id", "season"]], on="match_id")
    return {
        "r": r,
        "share": second_half_share(ht, TEST_SEASON),
        "delta_leader": _mean_se(resid[train["leader_step"].to_numpy()]),
        "delta_trailer": _mean_se(resid[train["trailer_step"].to_numpy()]),
        "beyond_cap": _mean_se(resid[beyond]),
        "beta": (float(beta), float(beta_se), int(on.sum())),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def poisson_log_loss(goals: np.ndarray, rate: np.ndarray) -> np.ndarray:
    return -poisson.logpmf(goals, np.maximum(rate, MIN_RATE))


def compare(df: pd.DataFrame, models: dict[str, str], match_ids=None) -> pd.DataFrame:
    """Per-match Poisson log-loss (sum over both teams), paired diffs vs the first model."""
    d = df if match_ids is None else df[df["match_id"].isin(match_ids)]
    per_match = d.groupby("match_id")[[f"ll_{m}" for m in models]].sum()
    base = per_match[f"ll_{next(iter(models))}"]
    rows = {}
    for i, (m, label) in enumerate(models.items()):
        diff = per_match[f"ll_{m}"] - base
        rows[label] = {"matches": len(per_match), "log_loss": per_match[f"ll_{m}"].mean(),
                       "diff": np.nan if i == 0 else diff.mean(),
                       "diff_se": np.nan if i == 0 else diff.std() / np.sqrt(len(diff))}
    return pd.DataFrame(rows).T


def run() -> pd.DataFrame:
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.4f}".format)
    xi = DixonColesModel(target="xg").xi
    print(f"Pre-match model: xG-fitted Dixon-Coles, xi = {xi}\n")

    print("Fitting on 2020/21-2023/24 (walk-forward pre-match rates) ...")
    fit = fit_training()
    dl, dt, bc, b = fit["delta_leader"], fit["delta_trailer"], fit["beyond_cap"], fit["beta"]
    print(f"  calibration ratio r          = {fit['r']:.4f}  (raw 2H goal share {fit['share']:.4f})")
    print(f"  delta_leader  (deficit 0-{CAP}) = {dl[0]:+.4f}  (se {dl[1]:.4f}, n={dl[2]})")
    print(f"  delta_trailer (edge 0-{CAP})    = {dt[0]:+.4f}  (se {dt[1]:.4f}, n={dt[2]})")
    print(f"  leaders with deficit >= {CAP}   : mean residual {bc[0]:+.4f} (se {bc[1]:.4f}, n={bc[2]})"
          f" -> no modifier")
    print(f"  linear beta (all deficits)   = {b[0]:+.4f}  (se {b[1]:.4f}, n={b[2]})\n")

    print(f"Walk-forward Dixon-Coles for {TEST_SEASON}/{(TEST_SEASON + 1) % 100:02d} ...")
    matches = load_matches()
    df = test_team_rows(matches)
    df["exp_naive"] = fit["r"] * df["pre_rate"]
    df["exp_share"] = fit["share"] * df["pre_rate"]
    df["exp_capped_leader"] = df["exp_naive"] + dl[0] * df["leader_step"]
    df["exp_capped_both"] = df["exp_capped_leader"] + dt[0] * df["trailer_step"]
    df["exp_linear_leader"] = df["exp_naive"] + b[0] * df["leader_deficit"]
    df["exp_share_capped_leader"] = df["exp_share"] + dl[0] * df["leader_step"]
    models = {
        "naive": "Naive baseline (calibrated r)",
        "capped_leader": "Capped leader model",
        "capped_both": "Capped leader + trailer",
        "linear_leader": "Linear leader (previous)",
        "share": "Naive, raw 2H share",
        "share_capped_leader": "Capped leader on raw share",
    }
    for m in models:
        df[f"ll_{m}"] = poisson_log_loss(df["half_2_goals"].to_numpy(), df[f"exp_{m}"].to_numpy())

    print(f"Mean 2H goals per team: actual {df['half_2_goals'].mean():.3f} | "
          f"naive {df['exp_naive'].mean():.3f} | raw share {df['exp_share'].mean():.3f}\n")

    print(f"=== All {df['match_id'].nunique()} matches: 2H Poisson log-loss per match "
          f"(diff vs naive; < 0 is better) ===")
    print(compare(df, models).to_string(), "\n")

    trig = df[df["leader_step"]]
    ids = trig["match_id"].unique()
    opp = df[df["match_id"].isin(ids) & ~df["leader_step"]]
    print(f"=== Triggered matches: {len(ids)} leaders with HT xG deficit in (0, {CAP}) ===")
    print(f"  leaders:   actual 2H goals {trig['half_2_goals'].mean():.3f} | naive "
          f"{trig['exp_naive'].mean():.3f} | capped model {trig['exp_capped_leader'].mean():.3f}")
    print(f"  opponents: actual 2H goals {opp['half_2_goals'].mean():.3f} | naive "
          f"{opp['exp_naive'].mean():.3f} | capped+trailer {opp['exp_capped_both'].mean():.3f}")
    print(compare(df, models, ids).to_string())
    return df


if __name__ == "__main__":
    run()
