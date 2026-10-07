"""The conditional-logit blend recovers known coefficients on synthetic data."""

import numpy as np

from src.experiments.market_blend import _probs, design, fit_clogit


def test_clogit_recovers_true_coefficients():
    rng = np.random.default_rng(0)
    n = 20_000
    q = rng.dirichlet([4, 3, 3], size=n)
    p = rng.dirichlet([4, 3, 3], size=n)
    theta_true = np.array([0.10, -0.15, 0.90, 0.40])   # a_H, a_D, b_market, b_model
    X = design(q, p)
    probs = _probs(theta_true, X)
    y = np.array([rng.choice(3, p=row) for row in probs])
    theta, se = fit_clogit(X, y)
    assert np.all(np.abs(theta - theta_true) < 4 * se)


def test_market_only_design_has_three_parameters():
    q = np.full((5, 3), 1 / 3)
    assert design(q, None).shape == (5, 3, 3)
