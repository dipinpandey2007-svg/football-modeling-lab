"""Unit tests for the Dixon-Coles engine on synthetic data."""

import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

from src.model.dixon_coles import OPTIMAL_XI, DixonColesModel, score_matrix


def synthetic_matches(n_rounds: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    strength = dict(zip(teams, np.linspace(-0.4, 0.4, len(teams))))
    rows, date, mid = [], pd.Timestamp("2021-08-01"), 0
    for _ in range(n_rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                mu = np.exp(0.25 + strength[h] - strength[a])
                lam = np.exp(strength[a] - strength[h])
                rows.append({"match_id": mid, "date": date, "season": 2021, "home_team": h,
                             "away_team": a, "home_goals": rng.poisson(mu), "away_goals": rng.poisson(lam),
                             "home_xg": mu, "away_xg": lam})
                mid += 1
                date += pd.Timedelta(hours=6)
    return pd.DataFrame(rows)


def test_score_matrix_full_grid_is_normalised():
    assert np.isclose(score_matrix(1.6, 1.2, -0.1, max_goals=30).sum(), 1.0)


def test_analytic_gradient_matches_numerical():
    df = synthetic_matches(n_rounds=1)
    model = DixonColesModel()
    model.teams = sorted(set(df.home_team) | set(df.away_team))
    idx = {t: i for i, t in enumerate(model.teams)}
    args = (df.home_team.map(idx).to_numpy(), df.away_team.map(idx).to_numpy(),
            df.home_goals.to_numpy(float), df.away_goals.to_numpy(float),
            df.home_goals.to_numpy(), df.away_goals.to_numpy(),
            np.random.default_rng(1).random(len(df)), (np.array([0, 3]), -0.2, 0.3, 3.0))
    p0 = np.random.default_rng(2).normal(0, 0.3, 2 * len(model.teams) + 2)
    num = approx_fprime(p0, lambda p: model._neg_log_likelihood(p, *args)[0], 1e-7)
    assert np.abs(num - model._neg_log_likelihood(p0, *args)[1]).max() < 1e-5


def test_fit_satisfies_constraint_and_recovers_ordering():
    model = DixonColesModel(target="goals").fit(synthetic_matches())
    assert abs(sum(model.attack.values())) < 1e-6
    params = model.parameters_frame()
    assert params.index[0] == "T7" and params.index[-1] == "T0"   # strongest / weakest
    p = model.predict_match("T7", "T0")
    assert np.isclose(p.home_win + p.draw + p.away_win, 1.0)
    assert p.home_win > p.away_win
    assert p.score_matrix.shape == (7, 7)


def test_default_xi_is_tuned_per_target():
    assert DixonColesModel(target="goals").xi == OPTIMAL_XI["goals"] == 0.0025
    assert DixonColesModel(target="xg").xi == OPTIMAL_XI["xg"] == 0.0046
    assert DixonColesModel(xi=0.001, target="xg").xi == 0.001
