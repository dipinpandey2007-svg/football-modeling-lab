"""
Half-Time Discovery Grid: where does the pre-match model miss in the 2nd half?

For every team in every training match (2020/21 - 2023/24):

    expected 2H goals = share_2h * (pre-match Dixon-Coles mu or lambda)
    2H residual       = actual 2H goals - expected 2H goals

and average the residual over a grid of half-time game states:
    rows    HT goal difference (team - opponent): trailing / drawing / leading
    columns HT xG difference:  <-0.5 | -0.5..0 | 0..0.5 | >=0.5

Notes
-----
- Pre-match mu / lambda come from walk-forward fits (out-of-sample).
  2019/20 is the first season of data, so it has no prior data to fit on
  and only seeds the later fits.
- share_2h replaces the naive mu / 2: 55-56% of EPL goals come after half-time,
  so mu / 2 would add a uniform ~+0.08 bias to every cell. It is estimated for
  each test season from earlier seasons only.
- Each match contributes two rows (one per team), so the grid is roughly
  mirror-symmetric: (leading, dominating) for one team is (trailing, dominated)
  for its opponent. Standard errors treat team-rows as independent, which
  is slightly optimistic.
- HT goals are exact (football-data.co.uk half-time scores). HT xG uses
  the shot split calibrated on those scores; shots whose half is still
  not pinned down are reported in `uncertain_xg` (see data_fetcher).

Usage:
    python -m src.ht_discovery
    python -m src.ht_discovery --target goals
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .backtest import walk_forward
from .data_fetcher import load_half_time_stats, load_matches
from .dixon_coles import DixonColesModel

TEST_SEASONS = [2020, 2021, 2022, 2023]

GD_BINS = [-np.inf, -0.5, 0.5, np.inf]
GD_LABELS = ["Trailing (<=-1)", "Drawing (0)", "Leading (>=+1)"]
XGD_BINS = [-np.inf, -0.5, 0.0, 0.5, np.inf]
XGD_LABELS = ["Dominated (<-0.5)", "Slightly worse (-0.5,0)",
              "Slightly better [0,0.5)", "Dominating (>=0.5)"]

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def prematch_rates(matches: pd.DataFrame, seasons: list[int], target: str) -> pd.DataFrame:
    """Walk-forward pre-match mu (home) and lambda (away) for each match."""
    preds = pd.concat([
        walk_forward(matches, s, verbose=False,
                     model_factory=lambda: DixonColesModel(target=target))
        for s in seasons])
    return preds[["match_id", "pred_home_xg", "pred_away_xg"]].rename(
        columns={"pred_home_xg": "mu", "pred_away_xg": "lam"})


def second_half_share(ht: pd.DataFrame, before_season: int) -> float:
    """Share of goals scored after half-time, from seasons before `before_season`."""
    prior = ht[ht["season"] < before_season]
    g1, g2 = prior["half_1_goals"].sum(), prior["half_2_goals"].sum()
    return g2 / (g1 + g2)


def team_rows(target: str = "xg", xg_split: str = "hard") -> pd.DataFrame:
    """One row per team per test match: HT state, expected and actual 2H output.

    xg_split : "hard" uses each shot's most-likely half for HT xG; "soft"
        weights each shot's xG by its posterior P(1st half).
    """
    matches = load_matches()
    ht = load_half_time_stats().merge(matches[["match_id", "season"]], on="match_id")
    rates = prematch_rates(matches, TEST_SEASONS, target)

    if xg_split == "soft":
        ht = ht.assign(half_1_xg=ht["half_1_xg_soft"], half_2_xg=ht["half_2_xg_soft"])
    df = ht[ht["season"].isin(TEST_SEASONS)].merge(rates, on="match_id", how="inner")
    opp = df[["match_id", "h_a", "half_1_goals", "half_1_xg"]].assign(
        h_a=lambda d: d["h_a"].map({"h": "a", "a": "h"}))
    df = df.merge(opp, on=["match_id", "h_a"], suffixes=("", "_opp"))

    df["share_2h"] = df["season"].map({s: second_half_share(ht, s) for s in TEST_SEASONS})
    df["pre_rate"] = np.where(df["h_a"] == "h", df["mu"], df["lam"])
    df["exp_2h_goals"] = df["share_2h"] * df["pre_rate"]
    df["resid_2h"] = df["half_2_goals"] - df["exp_2h_goals"]
    df["resid_2h_naive"] = df["half_2_goals"] - df["pre_rate"] / 2

    df["ht_gd"] = df["half_1_goals"] - df["half_1_goals_opp"]
    df["ht_xgd"] = df["half_1_xg"] - df["half_1_xg_opp"]
    df["gd_state"] = pd.cut(df["ht_gd"], GD_BINS, labels=GD_LABELS)
    df["xgd_state"] = pd.cut(df["ht_xgd"], XGD_BINS, labels=XGD_LABELS, right=False)
    return df


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def discovery_grid(df: pd.DataFrame, value: str = "resid_2h"):
    """Mean, standard error and count of `value` by (HT goal state, HT xG state)."""
    g = df.groupby(["gd_state", "xgd_state"], observed=False)[value]
    shape = lambda s: s.unstack().reindex(index=GD_LABELS, columns=XGD_LABELS)
    return shape(g.mean()), shape(g.std() / np.sqrt(g.size())), shape(g.size())


def format_grid(mean: pd.DataFrame, se: pd.DataFrame, n: pd.DataFrame) -> pd.DataFrame:
    """'+0.12 (0.05) n=340' per cell; '*' marks |mean| > 2 SE."""
    out = pd.DataFrame(index=mean.index, columns=mean.columns, dtype=object)
    for i in mean.index:
        for j in mean.columns:
            if n.loc[i, j] == 0:
                out.loc[i, j] = "n=0"
                continue
            flag = "*" if abs(mean.loc[i, j]) > 2 * se.loc[i, j] else " "
            out.loc[i, j] = f"{mean.loc[i, j]:+.3f}{flag} ({se.loc[i, j]:.3f}) n={n.loc[i, j]}"
    out.index.name, out.columns.name = "HT goal diff", "HT xG diff"
    return out


def margins(df: pd.DataFrame, by: str, value: str = "resid_2h") -> pd.DataFrame:
    g = df.groupby(by, observed=False)[value]
    return pd.DataFrame({"mean": g.mean(), "se": g.std() / np.sqrt(g.size()), "n": g.size()})


def print_grid(title: str, df: pd.DataFrame, value: str = "resid_2h") -> None:
    print(f"=== {title} ===")
    print(format_grid(*discovery_grid(df, value)).to_string())
    print()


def run(target: str = "xg") -> pd.DataFrame:
    print(f"Walk-forward Dixon-Coles (target={target}) for seasons {TEST_SEASONS} ...")
    df = team_rows(target)
    n_matches = df["match_id"].nunique()
    shares = df.groupby("season")["share_2h"].first()

    pd.set_option("display.width", 250)
    print(f"{n_matches} matches, {len(df)} team rows")
    print("2H goal share used (from prior seasons): "
          + ", ".join(f"{s}: {v:.3f}" for s, v in shares.items()))
    print(f"Mean 2H residual overall: {df['resid_2h'].mean():+.4f} with share scaling, "
          f"{df['resid_2h_naive'].mean():+.4f} with naive mu/2\n")

    print("Cells: mean 2H goal residual per team (SE) n=team-rows;  * = |mean| > 2 SE\n")
    print_grid("2nd-half GOAL residual (actual - expected); HT xG = most-likely half per shot", df)

    soft = team_rows(target, xg_split="soft")
    moved = (soft["xgd_state"].to_numpy() != df["xgd_state"].to_numpy()).sum()
    print_grid(f"Same, HT xG = soft (probability-weighted) split  "
               f"[{moved} of {len(df)} team-rows change xG bucket]", soft)

    print("=== Row margins (HT goal state), all matches ===")
    print(margins(df, "gd_state").round(3).to_string(), "\n")
    print("=== Column margins (HT xG state), all matches ===")
    print(margins(df, "xgd_state").round(3).to_string(), "\n")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["goals", "xg"], default="xg",
                    help="Dixon-Coles variant providing pre-match mu/lambda")
    run(ap.parse_args().target)
