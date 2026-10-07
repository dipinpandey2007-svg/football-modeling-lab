"""
Direction A: does the Dixon-Coles model carry information beyond the market?

1. Blend test (conditional logit over the three 1X2 outcomes)

       score_k = a_k + b_mkt * log q_open_k + b_model * log p_model_k
       P(k)    = softmax_k(score_k),  a_A = 0

   q_open  : margin-free opening probabilities (football-data's early snapshot:
             Friday for weekend games, Tuesday for midweek)
   p_model : walk-forward xG Dixon-Coles probabilities (xi = 0.0046), fitted
             before each matchweek, i.e. roughly when the snapshot is taken.
   Fitted by maximum likelihood on 2020/21-2023/24, frozen, then scored on
   2024/25, 2025/26 and 2026/27-to-date. A market-only recalibration
   (b_model = 0, same intercepts) is the fair control: any gain of the blend
   over it is information the model adds.

2. Line-movement test (one row per match)

       move     = closing home-vs-away log-odds  - opening home-vs-away log-odds
       disagree = model   home-vs-away log-odds  - opening home-vs-away log-odds

   OLS slope of move on disagree: > 0 means the market later moves toward the
   model. Plus closing-line value of betting the model's picks at opening
   odds: CLV = q_close * odds_open - 1, the EV if the margin-free closing line
   is the truth.

Usage:
    python -m src.market_blend
    python -m src.market_blend --book Avg
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .backtest import log_loss, outcome_index
from .betting import model_predictions
from .data_fetcher import load_matches, map_fd_team, read_fd_csv

TRAIN = [2020, 2021, 2022, 2023]
TEST = [2024, 2025, 2026]
OUT = ["H", "D", "A"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_odds(matches: pd.DataFrame, seasons: list[int], book: str) -> pd.DataFrame:
    """Opening ({book}H/D/A) and closing ({book}CH/CD/CA) odds per match_id."""
    known = set(matches["home_team"]) | set(matches["away_team"])
    frames = []
    for s in seasons:
        fd = read_fd_csv(s)
        out = pd.DataFrame({
            "season": s,
            "home_team": [map_fd_team(t, known) for t in fd["HomeTeam"]],
            "away_team": [map_fd_team(t, known) for t in fd["AwayTeam"]],
        })
        for o in OUT:
            out[f"open_{o}"] = fd[f"{book}{o}"].to_numpy(float)
            out[f"close_{o}"] = fd[f"{book}C{o}"].to_numpy(float)
        frames.append(out)
    odds = pd.concat(frames, ignore_index=True)
    return matches[["match_id", "season", "home_team", "away_team"]].merge(
        odds, on=["season", "home_team", "away_team"], how="inner", validate="one_to_one")


def fair_probs(df: pd.DataFrame, prefix: str) -> np.ndarray:
    inv = 1 / df[[f"{prefix}_{o}" for o in OUT]].to_numpy(float)
    return inv / inv.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Conditional logit blend
# ---------------------------------------------------------------------------

def design(q_open: np.ndarray, p_model: np.ndarray | None) -> np.ndarray:
    """Features X[i, k, j]: outcome dummies for H and D, then log-prob regressors."""
    n = len(q_open)
    feats = [np.broadcast_to(np.array([1.0, 0, 0]), (n, 3)),   # a_H
             np.broadcast_to(np.array([0, 1.0, 0]), (n, 3)),   # a_D
             np.log(q_open)]
    if p_model is not None:
        feats.append(np.log(p_model))
    return np.stack(feats, axis=2)


def _probs(theta, X):
    s = X @ theta
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def fit_clogit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MLE and standard errors (inverse observed information)."""
    n, _, k = X.shape
    Y = np.eye(3)[y]

    def nll(theta):
        P = _probs(theta, X)
        ll = np.log(P[np.arange(n), y]).sum()
        grad = np.einsum("nk,nkj->j", Y - P, X)
        return -ll, -grad

    theta0 = np.zeros(k)
    theta0[2] = 1.0
    res = minimize(nll, theta0, jac=True, method="BFGS")
    P = _probs(res.x, X)
    xbar = np.einsum("nk,nkj->nj", P, X)
    dev = X - xbar[:, None, :]
    info = np.einsum("nk,nkj,nkl->jl", P, dev, dev)
    return res.x, np.sqrt(np.diag(np.linalg.inv(info)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def paired(a: np.ndarray, b: np.ndarray) -> str:
    d = b - a
    return f"{d.mean():+.4f} +/- {d.std() / np.sqrt(len(d)):.4f}"


def ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Slope, its SE and R^2 for y = a + b x."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (len(y) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return float(coef[1]), float(se), float(1 - resid.var() / y.var())


def run(book: str = "B365") -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    matches = load_matches()
    seasons = TRAIN + TEST
    print(f"Walk-forward xG Dixon-Coles for {seasons} ...")
    preds = model_predictions(matches, seasons)
    df = preds.merge(load_odds(matches, seasons, book).drop(
        columns=["season", "home_team", "away_team"]), on="match_id", how="inner")
    df = df.dropna(subset=[f"{p}_{o}" for p in ("open", "close") for o in OUT]).reset_index(drop=True)
    y = outcome_index(df["home_goals"], df["away_goals"])
    q_open, q_close = fair_probs(df, "open"), fair_probs(df, "close")
    p_model = np.clip(df[["p_home", "p_draw", "p_away"]].to_numpy(), 1e-6, 1)
    train = df["season"].isin(TRAIN).to_numpy()
    print(f"{book} odds: {train.sum()} training matches, {(~train).sum()} test matches\n")

    # ---- 1. Blend -----------------------------------------------------------
    theta_b, se_b = fit_clogit(design(q_open[train], p_model[train]), y[train])
    theta_m, se_m = fit_clogit(design(q_open[train], None), y[train])
    names = ["a_H", "a_D", "b_market_open", "b_model"]
    print("=== Blend fitted on 2020/21-2023/24 ===")
    print(pd.DataFrame({"coef": theta_b, "se": se_b, "z": theta_b / se_b}, index=names).to_string())
    print(f"(market-only control: b_market_open = {theta_m[2]:.3f} +/- {se_m[2]:.3f})\n")

    ll = {
        "market_open": log_loss(q_open, y),
        "market_open_recal": log_loss(_probs(theta_m, design(q_open, None)), y),
        "blend": log_loss(_probs(theta_b, design(q_open, p_model)), y),
        "model_alone": log_loss(p_model, y),
        "market_close": log_loss(q_close, y),
    }
    rows = {}
    for label, mask in [*[(str(s), (df["season"] == s).to_numpy()) for s in TEST],
                        ("2024-2026 pooled", ~train), ("train 2020-2023", train)]:
        r = {k: v[mask].mean() for k, v in ll.items()}
        r["matches"] = mask.sum()
        r["blend - recal (+/- se)"] = paired(ll["market_open_recal"][mask], ll["blend"][mask])
        r["blend - open (+/- se)"] = paired(ll["market_open"][mask], ll["blend"][mask])
        rows[label] = r
    print("=== 1X2 log-loss (lower = better); differences paired by match ===")
    print(pd.DataFrame(rows).T[["matches", "market_open", "market_open_recal", "blend",
                                "market_close", "model_alone",
                                "blend - recal (+/- se)", "blend - open (+/- se)"]].to_string(), "\n")

    # ---- 2. Line movement ---------------------------------------------------
    lo = lambda q: np.log(q[:, 0]) - np.log(q[:, 2])
    move = lo(q_close) - lo(q_open)
    disagree = lo(p_model) - lo(q_open)
    print("=== Line movement: OLS of (close - open) home-vs-away log-odds on (model - open) ===")
    for label, mask in [("train 2020-2023", train), ("test 2024-2026", ~train), ("all", np.ones(len(df), bool))]:
        b, se, r2 = ols_slope(disagree[mask], move[mask])
        agree = np.mean(np.sign(move[mask]) == np.sign(disagree[mask]))
        print(f"  {label:16s} slope {b:+.4f} +/- {se:.4f} (z {b / se:+.1f}), R^2 {r2:.3f}, "
              f"same direction {agree:.1%}, n={mask.sum()}")

    # CLV of betting the model's disagreements at opening odds
    odds_open = df[[f"open_{o}" for o in OUT]].to_numpy(float)
    edge = p_model * odds_open - 1
    pick = edge.argmax(axis=1)
    idx = np.arange(len(df))
    clv = q_close[idx, pick] * odds_open[idx, pick] - 1
    roi = np.where(y == pick, odds_open[idx, pick] - 1, -1.0)
    print("\n=== Closing-line value of betting the model's best-EV pick at opening odds ===")
    print("  (CLV = margin-free closing prob x opening odds - 1; the market's margin makes the")
    print("   average over random bets about -margin)")
    for thr in (0.0, 0.05, 0.10):
        for label, mask in [("train", train), ("test", ~train)]:
            m = mask & (edge[idx, pick] > thr)
            c = clv[m]
            print(f"  model EV>{thr:4.0%} {label:5s}: {m.sum():4d} bets, mean CLV {100 * c.mean():+.2f}% "
                  f"+/- {100 * c.std() / np.sqrt(len(c)):.2f}%, beat close {np.mean(c > 0):.1%}, "
                  f"realised ROI {100 * roi[m].mean():+.1f}%")
    all_clv = (q_close * odds_open - 1).mean()
    print(f"  reference: mean CLV of every opening price {100 * all_clv:+.2f}%\n")

    # ---- 3. Verdict ---------------------------------------------------------
    z = theta_b[3] / se_b[3]
    d = ll["blend"][~train] - ll["market_open_recal"][~train]
    dz = d.mean() / (d.std() / np.sqrt(len(d)))
    b, se, _ = ols_slope(disagree[~train], move[~train])
    print("=== VERDICT ===")
    print(f"  Model weight in blend: {theta_b[3]:+.3f} +/- {se_b[3]:.3f} (z = {z:+.1f}) -> "
          f"{'SIGNIFICANT (> 2 SE)' if z > 2 else 'NOT significant'}")
    print(f"  Held-out blend vs market-only recalibration: {d.mean():+.4f} (z = {dz:+.1f}) -> "
          f"{'blend BETTER' if dz < -2 else 'blend worse' if dz > 2 else 'no significant difference'}")
    d_open = ll["blend"][~train] - ll["market_open"][~train]
    z_open = d_open.mean() / (d_open.std() / np.sqrt(len(d_open)))
    print(f"  Held-out blend vs raw opening line: {d_open.mean():+.4f} (z = {z_open:+.1f}) -> "
          f"{'blend BEATS the market' if z_open < -2 else 'blend does NOT beat the market'}")
    print(f"  Held-out log-loss: blend {ll['blend'][~train].mean():.4f} | opening "
          f"{ll['market_open'][~train].mean():.4f} | closing {ll['market_close'][~train].mean():.4f}")
    print(f"  Line-movement slope on held-out data: {b:+.3f} +/- {se:.3f} (z = {b / se:+.1f}) -> "
          f"{'model anticipates line moves' if b / se > 2 else 'no significant anticipation'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="B365", help="opening/closing odds prefix: B365, Avg, PS, BW")
    run(ap.parse_args().book)
