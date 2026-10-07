"""
Does the pre-match Dixon-Coles model find value against bookmaker closing odds?

Pipeline
--------
1. Walk-forward 1X2 probabilities from the xG-fitted Dixon-Coles model
   (xi = OPTIMAL_XI["xg"]), refitted before every matchweek, 2020/21-2025/26.
2. football-data.co.uk closing odds joined by (season, home, away):
     B365C*  Bet365 closing    -- the price we bet at
     MaxC*   best closing price across bookmakers -- optimistic line shopping
     AvgC*   market-average closing, PSC* Pinnacle closing -- benchmarks
3. Fair odds = 1 / p_model. EV of a 1-unit bet = p_model * odds - 1.
   Per match, bet the single outcome with the highest EV if EV > threshold.
4. Flat staking (1 unit) and quarter-Kelly (stake = 0.25 * f* * bankroll,
   f* = EV / (odds - 1), capped at 5% of bankroll), bets settled in date order.

Before any ROI, the model's 1X2 log-loss is compared with the margin-free
market probabilities (odds normalised to sum to 1): a model that is less
accurate than the market will mostly "find value" in its own errors.

xi was tuned on 2020/21-2023/24, so those seasons are in-sample for that
one hyperparameter; 2024/25 and 2025/26 are fully held out.

Usage:
    python -m src.experiments.closing_odds
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.backtest import log_loss, outcome_index
from ..evaluation.cache import cached_walk_forward
from ..data_fetcher import load_matches, map_fd_team, read_fd_csv
from ..model.dixon_coles import DixonColesModel

SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
HOLDOUT = {2024, 2025}
EV_THRESHOLDS = [0.0, 0.05, 0.10]
KELLY_FRACTION = 0.25
KELLY_CAP = 0.05
BOOKS = {"B365C": "Bet365 closing", "MaxC": "Best closing (max across books)",
         "AvgC": "Market average closing", "PSC": "Pinnacle closing"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_closing_odds(matches: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Closing odds per match_id from the cached football-data CSVs."""
    known = set(matches["home_team"]) | set(matches["away_team"])
    frames = []
    for season in seasons:
        fd = read_fd_csv(season)
        cols = {f"{b}{o}": f"{b}_{o}" for b in BOOKS for o in "HDA" if f"{b}{o}" in fd}
        out = fd[list(cols)].rename(columns=cols)
        out["season"] = season
        out["home_team"] = [map_fd_team(t, known) for t in fd["HomeTeam"]]
        out["away_team"] = [map_fd_team(t, known) for t in fd["AwayTeam"]]
        frames.append(out)
    odds = pd.concat(frames, ignore_index=True)
    return matches[["match_id", "season", "home_team", "away_team"]].merge(
        odds, on=["season", "home_team", "away_team"], how="inner", validate="one_to_one")


def model_predictions(matches: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    preds = [cached_walk_forward(matches, s, target="xg") for s in seasons]
    p = pd.concat(preds)
    p["season"] = p["match_id"].map(matches.set_index("match_id")["season"])
    return p


def implied_probs(df: pd.DataFrame, book: str) -> np.ndarray:
    """Margin-free probabilities (proportional normalisation) from a book's odds."""
    inv = 1 / df[[f"{book}_H", f"{book}_D", f"{book}_A"]].to_numpy(float)
    return inv / inv.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Betting
# ---------------------------------------------------------------------------

def select_bets(df: pd.DataFrame, book: str, threshold: float) -> pd.DataFrame:
    """One bet per match: the outcome with the highest EV, if EV > threshold."""
    p = df[["p_home", "p_draw", "p_away"]].to_numpy()
    o = df[[f"{book}_H", f"{book}_D", f"{book}_A"]].to_numpy(float)
    ev = p * o - 1
    pick = np.nanargmax(np.where(np.isnan(ev), -np.inf, ev), axis=1)
    rows = np.arange(len(df))
    bets = pd.DataFrame({
        "match_id": df["match_id"].to_numpy(),
        "date": df["date"].to_numpy(),
        "season": df["season"].to_numpy(),
        "pick": np.array(["H", "D", "A"])[pick],
        "p_model": p[rows, pick],
        "odds": o[rows, pick],
        "ev": ev[rows, pick],
        "won": pick == outcome_index(df["home_goals"], df["away_goals"]),
    })
    bets = bets[bets["ev"] > threshold].sort_values("date").reset_index(drop=True)
    bets["profit"] = np.where(bets["won"], bets["odds"] - 1, -1.0)
    return bets


def kelly_simulation(bets: pd.DataFrame, bankroll: float = 100.0) -> dict:
    """Fractional Kelly, settled sequentially in date order."""
    br, peak, max_dd, staked = bankroll, bankroll, 0.0, 0.0
    for b in bets.itertuples():
        f = min(KELLY_FRACTION * b.ev / (b.odds - 1), KELLY_CAP)
        stake = f * br
        staked += stake
        br += stake * (b.odds - 1) if b.won else -stake
        peak = max(peak, br)
        max_dd = max(max_dd, 1 - br / peak)
    return {"final_bankroll": br, "growth_%": 100 * (br / bankroll - 1),
            "kelly_roi_%": 100 * (br - bankroll) / staked if staked else np.nan,
            "max_drawdown_%": 100 * max_dd}


def summarise_bets(bets: pd.DataFrame) -> dict:
    n = len(bets)
    if n == 0:
        return {"bets": 0}
    roi_se = bets["profit"].std() / np.sqrt(n)
    return {
        "bets": n,
        "win_rate_%": 100 * bets["won"].mean(),
        "avg_odds": bets["odds"].mean(),
        "avg_ev_%": 100 * bets["ev"].mean(),
        "profit_units": bets["profit"].sum(),
        "roi_%": 100 * bets["profit"].mean(),
        "roi_se_%": 100 * roi_se,
        **kelly_simulation(bets),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.3f}".format)
    matches = load_matches()
    print(f"Walk-forward xG Dixon-Coles (xi={DixonColesModel(target='xg').xi}) for {SEASONS} ...")
    preds = model_predictions(matches, SEASONS)
    df = preds.merge(load_closing_odds(matches, SEASONS).drop(columns=["season", "home_team", "away_team"]),
                     on="match_id", how="inner")
    outcomes = outcome_index(df["home_goals"], df["away_goals"])
    print(f"{len(df)} matches with model probabilities and closing odds\n")

    # 1. Accuracy vs the market
    print("=== 1X2 log-loss: model vs margin-free closing-odds probabilities (lower = better) ===")
    acc = {}
    for label, mask in [("2020-2023 (xi tuned here)", ~df["season"].isin(HOLDOUT)),
                        ("2024-2025 (held out)", df["season"].isin(HOLDOUT)),
                        ("all 2020-2025", np.ones(len(df), bool))]:
        d, o = df[mask], outcomes[mask.to_numpy() if hasattr(mask, "to_numpy") else mask]
        row = {"model": log_loss(d[["p_home", "p_draw", "p_away"]].to_numpy(), o).mean()}
        for book in ("B365C", "AvgC", "PSC"):
            ok = d[f"{book}_H"].notna().to_numpy()
            row[book] = log_loss(implied_probs(d[ok], book), o[ok]).mean()
            row[f"model_on_{book}_rows"] = log_loss(
                d.loc[ok, ["p_home", "p_draw", "p_away"]].to_numpy(), o[ok]).mean()
        row["matches"] = len(d)
        acc[label] = row
    print(pd.DataFrame(acc).T[["matches", "model", "B365C", "AvgC", "PSC", "model_on_PSC_rows"]]
          .to_string(), "\n")
    margin = (1 / df[["B365C_H", "B365C_D", "B365C_A"]]).sum(axis=1).mean() - 1
    print(f"Average Bet365 closing margin (overround): {100 * margin:.2f}%\n")

    # 2. Value betting
    results = {}
    for book in ("B365C", "MaxC"):
        for thr in EV_THRESHOLDS:
            for label, mask in [("all", np.ones(len(df), bool)),
                                ("holdout", df["season"].isin(HOLDOUT).to_numpy())]:
                bets = select_bets(df[mask], book, thr)
                results[(BOOKS[book], f"EV>{thr:.0%}", label)] = summarise_bets(bets)
    res = pd.DataFrame(results).T
    res.index.names = ["price", "threshold", "seasons"]
    print("=== Value betting: flat 1-unit stakes and quarter-Kelly (5% cap), start bankroll 100 ===")
    print(res[["bets", "win_rate_%", "avg_odds", "avg_ev_%", "profit_units", "roi_%", "roi_se_%",
               "growth_%", "max_drawdown_%"]].to_string(), "\n")

    # 3. By season, Bet365 closing, EV > 0
    bets = select_bets(df, "B365C", 0.0)
    by_season = pd.DataFrame({s: summarise_bets(b) for s, b in bets.groupby("season")}).T
    print("=== By season: Bet365 closing, EV > 0, flat stakes ===")
    print(by_season[["bets", "win_rate_%", "avg_odds", "roi_%", "roi_se_%"]].to_string(), "\n")

    by_pick = pd.DataFrame({k: summarise_bets(b) for k, b in bets.groupby("pick")}).T
    print("=== By outcome picked: Bet365 closing, EV > 0, flat stakes ===")
    print(by_pick[["bets", "win_rate_%", "avg_odds", "avg_ev_%", "roi_%", "roi_se_%"]].to_string())
    return res


if __name__ == "__main__":
    run()
