"""
Dixon-Coles (1997) goal model with exponential time-decay weighting.

Model
-----
For a match between home team h and away team a:

    log mu     = gamma + alpha_h + beta_a      (home expected goals)
    log lambda =         alpha_a + beta_h      (away expected goals)

    P(X=x, Y=y) = tau(x, y; mu, lambda, rho) * Pois(x; mu) * Pois(y; lambda)

alpha_i  attack parameter of team i (higher = scores more)
beta_i   defence parameter of team i (higher = concedes more). With the
         identifiability constraint sum(alpha_i) = 0, the betas also absorb
         the league-average scoring rate, as in the original paper.
gamma    home advantage (log-scale)
rho      low-score dependence; tau only modifies 0-0, 1-0, 0-1 and 1-1.

Each match's log-likelihood contribution is weighted by exp(-xi * t), where t
is the number of days between that match and the reference (prediction) date.

Target
------
`target="goals"` is the classic model. `target="xg"` fits the Poisson part to
match xG instead (a quasi-Poisson likelihood -- valid for non-integer targets),
while the tau/rho term is still fitted on actual scorelines since it describes
the joint distribution of real goals.

Extension points
----------------
- `score_matrix(mu, lam, rho)` is a free function: any later module (e.g.
  half-time Bayesian updates) can adjust mu / lambda and rebuild the
  scoreline distribution without touching the fitted model.
- `fit(..., extra_weights=...)` lets callers multiply the time-decay weights
  by arbitrary per-match weights (e.g. down-weighting matches with red cards).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# Time-decay rate per day, chosen per target by src/optimize_xi.py: walk-forward
# over 2020/21-2023/24, minimising out-of-sample full-time Poisson log-loss.
OPTIMAL_XI = {"goals": 0.0025, "xg": 0.0046}
DEFAULT_XI = OPTIMAL_XI["goals"]


# ---------------------------------------------------------------------------
# Scoreline distribution helpers
# ---------------------------------------------------------------------------

def tau(x, y, mu, lam, rho):
    """Dixon-Coles low-score correction factor (vectorised)."""
    x, y = np.asarray(x), np.asarray(y)
    out = np.ones(np.broadcast(x, y, mu, lam).shape)
    out = np.where((x == 0) & (y == 0), 1.0 - mu * lam * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + mu * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + lam * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
    return out


def score_matrix(mu: float, lam: float, rho: float, max_goals: int = 6) -> np.ndarray:
    """(max_goals+1) x (max_goals+1) matrix; entry [i, j] = P(home=i, away=j).

    Not renormalised: mass beyond max_goals is simply excluded, so a small
    matrix sums to slightly less than 1.
    """
    goals = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(goals, mu), poisson.pmf(goals, lam))
    m[:2, :2] *= tau(goals[:2, None], goals[None, :2], mu, lam, rho)
    return m


def outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    """(home win, draw, away win) from a scoreline matrix, renormalised."""
    total = matrix.sum()
    home = np.tril(matrix, -1).sum() / total
    draw = np.trace(matrix) / total
    away = np.triu(matrix, 1).sum() / total
    return float(home), float(draw), float(away)


@dataclass
class MatchPrediction:
    home_team: str
    away_team: str
    home_xg: float              # mu
    away_xg: float              # lambda
    score_matrix: np.ndarray    # 7x7 by default, tau-adjusted
    home_win: float
    draw: float
    away_win: float

    @property
    def probs_1x2(self) -> np.ndarray:
        return np.array([self.home_win, self.draw, self.away_win])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DixonColesModel:
    """Time-weighted Dixon-Coles model fitted by SLSQP.

    Parameters
    ----------
    xi : time-decay rate per day (weight = exp(-xi * days_before_ref_date)).
        None -> OPTIMAL_XI[target].
    target : "goals" or "xg" -- what the Poisson rates are fitted to.
    max_goals : size of the returned scoreline matrix is (max_goals+1)^2.
    n_weakest_for_new_teams : teams unseen in training (e.g. a promoted side
        with no top-flight history) get the mean parameters of the n weakest
        teams from the latest training season.
    newcomer_prior_weight : strength (roughly in matches) of a Gaussian prior
        pulling *newcomers* -- teams whose first match in the training data is
        in the latest training season -- toward the weak-team prior estimated
        from data before that season. Without it, a newcomer with one or two
        results gets extreme parameters (e.g. 0 goals -> attack at its bound).
        Set to 0 for the pure maximum-likelihood model.
    style_matrix : optional square DataFrame/array S where S[i, j] is the mean
        goal-difference residual of an archetype-i team against an archetype-j
        team (archetypes.S_resid). Used by predict_match when clusters are given.
    style_split : how a goal-difference shift d is applied to (mu, lambda):
        "symmetric" -> mu + d/2, lambda - d/2 (expected total goals unchanged)
        "home"      -> mu + d                  (all of it on the home side)
    """

    MIN_RATE = 0.05  # floor for style-adjusted expected goals

    def __init__(self, xi: float | None = None, target: str = "goals",
                 max_goals: int = 6, n_weakest_for_new_teams: int = 3,
                 newcomer_prior_weight: float = 3.0, style_matrix=None,
                 style_split: str = "symmetric"):
        if style_split not in ("symmetric", "home"):
            raise ValueError("style_split must be 'symmetric' or 'home'")
        self.style_matrix = None if style_matrix is None else np.asarray(style_matrix, dtype=float)
        self.style_split = style_split
        if target not in ("goals", "xg"):
            raise ValueError("target must be 'goals' or 'xg'")
        self.xi = OPTIMAL_XI[target] if xi is None else xi
        self.target = target
        self.max_goals = max_goals
        self.n_weakest_for_new_teams = n_weakest_for_new_teams
        self.newcomer_prior_weight = newcomer_prior_weight

        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_adv: float = np.nan
        self.rho: float = np.nan
        self.ref_date: pd.Timestamp | None = None
        self.opt_result = None
        self._new_team_params: tuple[float, float] | None = None

    # -- weighting -----------------------------------------------------------

    def time_weights(self, dates: pd.Series, ref_date: pd.Timestamp) -> np.ndarray:
        """exp(-xi * t), t = whole days from each match to ref_date."""
        t = (ref_date.normalize() - pd.to_datetime(dates).dt.normalize()).dt.days.to_numpy()
        return np.exp(-self.xi * np.clip(t, 0, None))

    # -- likelihood ----------------------------------------------------------

    def _unpack(self, params: np.ndarray):
        n = len(self.teams)
        return params[:n], params[n:2 * n], params[2 * n], params[2 * n + 1]

    def _neg_log_likelihood(self, params, hi, ai, x, y, gx, gy, w, prior=None):
        """Weighted negative log-likelihood and its analytic gradient.

        x, y   : Poisson targets (goals or xG)
        gx, gy : actual goals, used for the tau term
        prior  : optional (team_idx, alpha0, beta0, weight) Gaussian penalty
        """
        n = len(self.teams)
        alpha, beta, gamma, rho = self._unpack(params)

        log_mu = gamma + alpha[hi] + beta[ai]
        log_lam = alpha[ai] + beta[hi]
        mu, lam = np.exp(log_mu), np.exp(log_lam)

        # tau and the derivatives of log(tau) wrt rho, log(mu), log(lam)
        m00 = (gx == 0) & (gy == 0)
        m01 = (gx == 0) & (gy == 1)
        m10 = (gx == 1) & (gy == 0)
        m11 = (gx == 1) & (gy == 1)
        t = np.ones_like(mu)
        t[m00] = 1 - mu[m00] * lam[m00] * rho
        t[m01] = 1 + mu[m01] * rho
        t[m10] = 1 + lam[m10] * rho
        t[m11] = 1 - rho
        t = np.maximum(t, 1e-10)

        dlt_rho = np.zeros_like(mu)
        dlt_rho[m00] = -mu[m00] * lam[m00] / t[m00]
        dlt_rho[m01] = mu[m01] / t[m01]
        dlt_rho[m10] = lam[m10] / t[m10]
        dlt_rho[m11] = -1 / t[m11]

        dlt_mu = np.zeros_like(mu)
        dlt_lam = np.zeros_like(mu)
        dlt_mu[m00] = dlt_lam[m00] = -mu[m00] * lam[m00] * rho / t[m00]
        dlt_mu[m01] = mu[m01] * rho / t[m01]
        dlt_lam[m10] = lam[m10] * rho / t[m10]

        # log-likelihood (constant -log(x!) terms dropped)
        ll = w * (np.log(t) + x * log_mu - mu + y * log_lam - lam)

        g_mu = w * (x - mu + dlt_mu)     # d ll / d log(mu)
        g_lam = w * (y - lam + dlt_lam)  # d ll / d log(lam)
        grad = np.concatenate([
            np.bincount(hi, g_mu, n) + np.bincount(ai, g_lam, n),   # alpha
            np.bincount(ai, g_mu, n) + np.bincount(hi, g_lam, n),   # beta
            [g_mu.sum()],                                            # gamma
            [(w * dlt_rho).sum()],                                   # rho
        ])
        nll, grad = -ll.sum(), -grad
        if prior is not None:
            idx, a0, b0, k = prior
            da, db = alpha[idx] - a0, beta[idx] - b0
            nll += 0.5 * k * (da @ da + db @ db)
            grad[idx] += k * da
            grad[n + idx] += k * db

        scale = w.sum()
        return nll / scale, grad / scale

    # -- fitting -------------------------------------------------------------

    def fit(self, matches: pd.DataFrame, ref_date: pd.Timestamp | None = None,
            extra_weights: np.ndarray | None = None) -> "DixonColesModel":
        """Fit on `matches` (standard columns from data_fetcher).

        ref_date : date of the matches to be predicted; time decay is measured
            from here. Defaults to the day after the last training match.
            Matches on/after ref_date are ignored to prevent leakage.
        extra_weights : optional per-match multipliers on the decay weights.
        """
        df = matches.copy()
        if ref_date is None:
            ref_date = df["date"].max().normalize() + pd.Timedelta(days=1)
        ref_date = pd.Timestamp(ref_date)
        keep = (df["date"] < ref_date).to_numpy()
        df = df[keep]
        if extra_weights is not None:
            extra_weights = np.asarray(extra_weights)[keep]
        if df.empty:
            raise ValueError("No training matches before ref_date")
        self.ref_date = ref_date

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        hi = df["home_team"].map(idx).to_numpy()
        ai = df["away_team"].map(idx).to_numpy()
        gx = df["home_goals"].to_numpy()
        gy = df["away_goals"].to_numpy()
        if self.target == "xg":
            x, y = df["home_xg"].to_numpy(float), df["away_xg"].to_numpy(float)
        else:
            x, y = gx.astype(float), gy.astype(float)

        w = self.time_weights(df["date"], ref_date)
        if extra_weights is not None:
            w = w * extra_weights

        # Starting values: neutral attack, league-average away rate in beta.
        x0 = np.concatenate([
            np.zeros(n),
            np.full(n, np.log(max(np.average(y, weights=w), 0.1))),
            [0.25, -0.05],
        ])
        bounds = [(-4, 4)] * (2 * n) + [(-1, 2), (-0.25, 0.25)]
        constraints = [{
            "type": "eq",
            "fun": lambda p: p[:n].sum(),
            "jac": lambda p: np.concatenate([np.ones(n), np.zeros(n + 2)]),
        }]

        res = minimize(
            self._neg_log_likelihood, x0,
            args=(hi, ai, x, y, gx, gy, w, self._newcomer_prior(df, ref_date, idx)),
            jac=True, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if not res.success:
            raise RuntimeError(f"Dixon-Coles fit did not converge: {res.message}")
        self.opt_result = res

        alpha, beta, gamma, rho = self._unpack(res.x)
        self.attack = dict(zip(self.teams, alpha))
        self.defence = dict(zip(self.teams, beta))
        self.home_adv, self.rho = float(gamma), float(rho)
        self._new_team_params = self._weak_team_prior(df)
        return self

    def _newcomer_prior(self, df: pd.DataFrame, ref_date: pd.Timestamp, idx: dict):
        """Penalty spec for teams first seen in the latest training season.

        The prior centre comes from a separate fit on data before that season
        began (i.e. roughly last season's relegation-zone level).
        """
        if self.newcomer_prior_weight <= 0:
            return None
        season_start = df.loc[df["season"] == df["season"].max(), "date"].min().normalize()
        before = df[df["date"] < season_start]
        if before.empty:
            return None
        seen = set(before["home_team"]) | set(before["away_team"])
        newcomers = [t for t in self.teams if t not in seen]
        if not newcomers:
            return None
        pre = DixonColesModel(self.xi, self.target, self.max_goals,
                              self.n_weakest_for_new_teams, newcomer_prior_weight=0)
        a0, b0 = pre.fit(before, ref_date=season_start)._new_team_params
        return (np.array([idx[t] for t in newcomers]), a0, b0, self.newcomer_prior_weight)

    def _weak_team_prior(self, df: pd.DataFrame) -> tuple[float, float]:
        """Mean (alpha, beta) of the weakest teams in the latest training season."""
        latest = df[df["season"] == df["season"].max()]
        recent = sorted(set(latest["home_team"]) | set(latest["away_team"]))
        strength = sorted(recent, key=lambda t: self.attack[t] - self.defence[t])
        weakest = strength[: self.n_weakest_for_new_teams]
        return (float(np.mean([self.attack[t] for t in weakest])),
                float(np.mean([self.defence[t] for t in weakest])))

    # -- prediction ----------------------------------------------------------

    def team_params(self, team: str) -> tuple[float, float]:
        """(alpha, beta) for a team, falling back to the weak-team prior."""
        if team in self.attack:
            return self.attack[team], self.defence[team]
        if self._new_team_params is None:
            raise RuntimeError("Model is not fitted")
        return self._new_team_params

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """(mu, lambda): expected home and away goals (or xG, if target='xg')."""
        a_h, b_h = self.team_params(home_team)
        a_a, b_a = self.team_params(away_team)
        mu = np.exp(self.home_adv + a_h + b_a)
        lam = np.exp(a_a + b_h)
        return float(mu), float(lam)

    def style_adjustment(self, home_cluster: int | None, away_cluster: int | None) -> float:
        """Goal-difference shift S[home, away]; 0 if unknown, same cluster or no matrix.

        S is antisymmetric, so S[home, away] is already from the home side's view.
        """
        if (self.style_matrix is None or home_cluster is None or away_cluster is None
                or home_cluster == away_cluster):
            return 0.0
        d = self.style_matrix[int(home_cluster), int(away_cluster)]
        return 0.0 if np.isnan(d) else float(d)

    def apply_style(self, mu: float, lam: float, d: float) -> tuple[float, float]:
        """Shift the expected goal difference by d according to style_split."""
        if d == 0.0:
            return mu, lam
        if self.style_split == "symmetric":
            mu, lam = mu + d / 2, lam - d / 2
        else:
            mu = mu + d
        return max(mu, self.MIN_RATE), max(lam, self.MIN_RATE)

    def predict_match(self, home_team: str, away_team: str,
                      home_cluster: int | None = None,
                      away_cluster: int | None = None) -> MatchPrediction:
        """Expected goals, tau-adjusted scoreline matrix and 1X2 probabilities.

        If both archetype clusters are given and the model has a style_matrix,
        mu / lambda are shifted by S[home_cluster, away_cluster] before the
        scoreline matrix and 1X2 probabilities are built.

        The 1X2 probabilities are summed over a wider 0..15 grid so they are not
        biased by the truncation of the returned (max_goals+1)^2 matrix.
        """
        mu, lam = self.expected_goals(home_team, away_team)
        mu, lam = self.apply_style(mu, lam, self.style_adjustment(home_cluster, away_cluster))
        h, d, a = outcome_probs(score_matrix(mu, lam, self.rho, max_goals=15))
        return MatchPrediction(
            home_team=home_team, away_team=away_team,
            home_xg=mu, away_xg=lam,
            score_matrix=score_matrix(mu, lam, self.rho, self.max_goals),
            home_win=h, draw=d, away_win=a,
        )

    def parameters_frame(self) -> pd.DataFrame:
        """Fitted team parameters, sorted by overall strength (alpha - beta)."""
        df = pd.DataFrame({"attack": self.attack, "defence": self.defence})
        df["strength"] = df["attack"] - df["defence"]
        return df.sort_values("strength", ascending=False)
