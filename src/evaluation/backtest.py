"""
Walk-forward out-of-sample backtest.

For each matchweek of the test season:
  1. fit a fresh model on every match played strictly before the matchweek's
     first day (prior seasons + test-season matches already played),
  2. predict 1X2 probabilities for every match in the matchweek,
  3. score the predictions with log-loss and RPS.

Understat does not publish matchweek numbers, so matchweeks are built from the
fixture calendar: a new matchweek starts when a match falls more than
`matchweek_span_days` days after the current matchweek's first day. With the
default of 3 days, a Fri-Mon weekend round and a Tue-Thu midweek round each
form one block. Postponed fixtures simply land in whichever block they are
played in -- the only thing that matters for leakage is that training data
always ends before the block starts.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..model.dixon_coles import DixonColesModel

OUTCOMES = ("H", "D", "A")  # ordered for RPS


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def outcome_index(home_goals, away_goals) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    hg, ag = np.asarray(home_goals), np.asarray(away_goals)
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Per-match multiclass log-loss (natural log)."""
    p = np.clip(probs[np.arange(len(outcomes)), outcomes], eps, 1.0)
    return -np.log(p)


def rps(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Per-match Ranked Probability Score over ordered outcomes H < D < A.

    RPS = 1/(r-1) * sum_{k=1}^{r-1} (CDF_pred_k - CDF_obs_k)^2, r = 3.
    """
    obs = np.zeros_like(probs)
    obs[np.arange(len(outcomes)), outcomes] = 1.0
    cum_diff = np.cumsum(probs, axis=1) - np.cumsum(obs, axis=1)
    return (cum_diff[:, :-1] ** 2).sum(axis=1) / (probs.shape[1] - 1)


# ---------------------------------------------------------------------------
# Walk-forward loop
# ---------------------------------------------------------------------------

def assign_matchweeks(fixtures: pd.DataFrame, span_days: int = 3) -> pd.Series:
    """Group fixtures into calendar blocks (see module docstring)."""
    days = fixtures["date"].dt.normalize().sort_values()
    mw, start, labels = 0, None, {}
    for i, d in days.items():
        if start is None or (d - start).days > span_days:
            mw, start = mw + 1, d
        labels[i] = mw
    return pd.Series(labels).reindex(fixtures.index)


def walk_forward(matches: pd.DataFrame, test_season: int = 2024,
                 model_factory: Callable[[], DixonColesModel] = DixonColesModel,
                 matchweek_span_days: int = 3, verbose: bool = True,
                 cluster_fn: Callable[[pd.Timestamp], dict[str, int]] | None = None) -> pd.DataFrame:
    """Run the walk-forward backtest and return one row per test match.

    model_factory : zero-arg callable returning an unfitted model, e.g.
        `lambda: DixonColesModel(xi=0.0046, target="xg")`. Any object with
        `fit(matches, ref_date=...)` and `predict_match(home, away)` works,
        which is the hook for swapping in extended models later.
    cluster_fn : optional callable ref_date -> {team: archetype}, e.g.
        `ArchetypeTagger.clusters_as_of`. Called once per matchweek with the
        matchweek's first day, so labels only use matches before it; the
        labels are passed to predict_match as home_cluster / away_cluster.
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    test = matches[matches["season"] == test_season].copy()
    test["matchweek"] = assign_matchweeks(test, matchweek_span_days)

    rows = []
    for mw, block in test.groupby("matchweek", sort=True):
        ref_date = block["date"].min().normalize()
        train = matches[matches["date"] < ref_date]
        model = model_factory().fit(train, ref_date=ref_date)
        clusters = cluster_fn(ref_date) if cluster_fn is not None else {}

        for _, m in block.iterrows():
            hc, ac = clusters.get(m["home_team"]), clusters.get(m["away_team"])
            if cluster_fn is not None:
                p = model.predict_match(m["home_team"], m["away_team"],
                                        home_cluster=hc, away_cluster=ac)
            else:
                p = model.predict_match(m["home_team"], m["away_team"])
            rows.append({
                "home_cluster": hc,
                "away_cluster": ac,
                "matchweek": mw,
                "ref_date": ref_date,
                "match_id": m["match_id"],
                "date": m["date"],
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "home_goals": m["home_goals"],
                "away_goals": m["away_goals"],
                "pred_home_xg": p.home_xg,
                "pred_away_xg": p.away_xg,
                "p_home": p.home_win,
                "p_draw": p.draw,
                "p_away": p.away_win,
                "rho": model.rho,
                "home_adv": model.home_adv,
                "n_train": len(train),
            })
        if verbose:
            print(f"  MW {mw:2d}  {ref_date.date()}  train={len(train):4d}  "
                  f"matches={len(block):2d}  rho={model.rho:+.3f}  gamma={model.home_adv:.3f}")

    preds = pd.DataFrame(rows)
    probs = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    outcomes = outcome_index(preds["home_goals"], preds["away_goals"])
    preds["outcome"] = [OUTCOMES[o] for o in outcomes]
    preds["log_loss"] = log_loss(probs, outcomes)
    preds["rps"] = rps(probs, outcomes)
    return preds


def baseline_scores(matches: pd.DataFrame, test_season: int = 2024) -> pd.DataFrame:
    """Reference scores for naive forecasters on the test season.

    - uniform: 1/3 each
    - base_rate: H/D/A frequencies from all seasons before the test season
    """
    test = matches[matches["season"] == test_season]
    train = matches[matches["season"] < test_season]
    outcomes = outcome_index(test["home_goals"], test["away_goals"])
    freq = np.bincount(outcome_index(train["home_goals"], train["away_goals"]),
                       minlength=3) / len(train)
    out = {}
    for name, p in {"uniform": np.full(3, 1 / 3), "base_rate": freq}.items():
        probs = np.tile(p, (len(test), 1))
        out[name] = {"log_loss": log_loss(probs, outcomes).mean(),
                     "rps": rps(probs, outcomes).mean()}
    return pd.DataFrame(out).T


def summarise(preds: pd.DataFrame) -> pd.Series:
    """Headline metrics for a walk-forward run."""
    return pd.Series({
        "matches": len(preds),
        "log_loss": preds["log_loss"].mean(),
        "rps": preds["rps"].mean(),
        "accuracy": (preds[["p_home", "p_draw", "p_away"]].to_numpy().argmax(axis=1)
                     == preds["outcome"].map({"H": 0, "D": 1, "A": 2}).to_numpy()).mean(),
    })
