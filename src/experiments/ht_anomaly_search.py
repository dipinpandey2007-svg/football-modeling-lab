"""
Exploratory search for multi-variable half-time anomalies (training data only).

Unit: one team in one match, 2020/21 - 2023/24 (2019/20 only seeds the
walk-forward). Each row is described by six categorical half-time dimensions:

    venue      home / away
    ht_score   trail2+ / trail1 / level / lead1 / lead2+   (exact HT score)
    ht_xg      dominated (<-0.5) / worse (-0.5,0) / better [0,0.5) / dominating (>=0.5)
               team HT xG minus opponent HT xG, soft split
    pre_match  underdog / even / favourite: pre-match expected goal difference
               (own minus opponent walk-forward Dixon-Coles rate) < -0.4 / within / > +0.4
    rest       short (<=4 days) / normal (5-7) / long (>=8) since the team's
               previous Premier League match. Cups and Europe are NOT in the
               data, so this understates congestion for teams playing midweek elsewhere.
    rest_diff  fresher (>= 2 days more rest than opponent) / similar / tired

Target: 2nd-half goal residual = actual 2H goals - r * pre-match rate, with r
calibrated on the same training rows (mean residual 0 overall). The 2H xG
residual is reported alongside as a check: an effect seen in goals but not xG
is more likely finishing noise than behaviour.

Search: every cell of every combination of 2-4 dimensions with > MIN_MATCHES
distinct matches. Standard errors are clustered by match.

Guards against data-mining false positives:
  - multiple testing: Benjamini-Hochberg FDR and Bonferroni over all tested
    cells, and the number of |z| > 2 / > 3 hits expected from noise alone
  - internal replication: effect must have the same sign in 2020-21 and
    2022-23, and in at least 3 of the 4 seasons
  - ranking by the weaker of the two half-sample z-scores, then removing
    candidates that share > 50% of their matches with a higher-ranked one

The top candidates are frozen to results/ht_anomaly_candidates.json for a blind
test on later seasons. Nothing from 2024/25 onwards is loaded here.

Usage:
    python -m src.experiments.ht_anomaly_search
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..data_fetcher import load_matches
from ..ingest.config import RESULTS_DIR
from .ht_grid import TRAIN_SEASONS, team_rows

MIN_MATCHES = 30
MAX_DIMS = 4
TOP_K = 2
OUT_PATH = RESULTS_DIR / "ht_anomaly_candidates.json"

DIMENSIONS = ["venue", "ht_score", "ht_xg", "pre_match", "rest", "rest_diff"]
SPLITS = {"A (2020-21)": [2020, 2021], "B (2022-23)": [2022, 2023]}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def rest_days(matches: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous Premier League match (NaN for the first)."""
    long = pd.concat([
        matches[["match_id", "date", "home_team"]].rename(columns={"home_team": "team"}),
        matches[["match_id", "date", "away_team"]].rename(columns={"away_team": "team"}),
    ]).sort_values(["team", "date"])
    long["rest"] = long.groupby("team")["date"].diff().dt.days
    return long[["match_id", "team", "rest"]]


def build_rows() -> pd.DataFrame:
    """Training team rows with all six state dimensions and both residuals."""
    df = add_dimensions(team_rows("xg", xg_split="soft"), load_matches())
    r_goals = df["half_2_goals"].sum() / df["pre_rate"].sum()
    r_xg = df["half_2_xg"].sum() / df["pre_rate"].sum()
    df["resid_goals"] = df["half_2_goals"] - r_goals * df["pre_rate"]
    df["resid_xg"] = df["half_2_xg"] - r_xg * df["pre_rate"]
    df.attrs.update(r_goals=r_goals, r_xg=r_xg)
    return df


def add_dimensions(df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Add the six categorical state dimensions to team rows.

    Needs per team row: match_id, team, h_a, pre_rate, ht_gd, ht_xgd (soft split).
    Shared by the training search and the blind test so states mean the same thing.
    """
    df = df.merge(rest_days(matches).rename(columns={"rest": "rest_days"}),
                  on=["match_id", "team"], how="left", validate="one_to_one")
    opp = df[["match_id", "h_a", "pre_rate", "rest_days"]].assign(
        h_a=lambda d: d["h_a"].map({"h": "a", "a": "h"}))
    df = df.merge(opp, on=["match_id", "h_a"], suffixes=("", "_opp"), validate="one_to_one")

    df["venue"] = np.where(df["h_a"] == "h", "home", "away")
    df["ht_score"] = pd.cut(df["ht_gd"], [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf],
                            labels=["trail2+", "trail1", "level", "lead1", "lead2+"]).astype(str)
    df["ht_xg"] = pd.cut(df["ht_xgd"], [-np.inf, -0.5, 0, 0.5, np.inf], right=False,
                         labels=["dominated", "worse", "better", "dominating"]).astype(str)
    sup = df["pre_rate"] - df["pre_rate_opp"]
    df["pre_match"] = np.select([sup < -0.4, sup > 0.4], ["underdog", "favourite"], "even")
    df["rest"] = pd.cut(df["rest_days"], [-np.inf, 4, 7, np.inf],
                        labels=["short", "normal", "long"]).astype(str)
    gap = df["rest_days"] - df["rest_days_opp"]
    df["rest_diff"] = np.select([gap >= 2, gap <= -2], ["fresher", "tired"], "similar")
    df.loc[gap.isna(), "rest_diff"] = "nan"
    return df


def in_state(df: pd.DataFrame, conditions: dict[str, str]) -> np.ndarray:
    mask = np.ones(len(df), bool)
    for d, v in conditions.items():
        mask &= (df[d] == v).to_numpy()
    return mask


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cluster_stats(sub: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """Mean residual per team-row with a match-clustered SE; returns (mean, se, n_matches)."""
    n_rows = len(sub)
    per_match = sub.groupby("match_id")[col].sum()
    k = len(per_match)
    mean = per_match.sum() / n_rows
    if k < 2:
        return mean, np.nan, k
    # Cluster-robust variance of the mean: sum over clusters of (sum resid - n_c * mean)^2 / N^2
    n_c = sub.groupby("match_id").size()
    u = per_match - n_c * mean
    se = np.sqrt((u ** 2).sum() * k / (k - 1)) / n_rows
    return float(mean), float(se), k


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def search(df: pd.DataFrame) -> pd.DataFrame:
    """Every cell of every 2..MAX_DIMS-dimension combination with enough matches."""
    rows = []
    for k in range(2, MAX_DIMS + 1):
        for dims in itertools.combinations(DIMENSIONS, k):
            for values, sub in df.groupby(list(dims)):
                if "nan" in values:
                    continue
                if sub["match_id"].nunique() <= MIN_MATCHES:
                    continue
                m, se, n = cluster_stats(sub, "resid_goals")
                mx, sex, _ = cluster_stats(sub, "resid_xg")
                rec = {"state": " & ".join(f"{d}={v}" for d, v in zip(dims, values)),
                       "dims": dims, "values": values, "n_matches": n, "n_rows": len(sub),
                       "effect": m, "se": se, "z": m / se,
                       "xg_effect": mx, "xg_z": mx / sex}
                for name, seasons in SPLITS.items():
                    s = sub[sub["season"].isin(seasons)]
                    a, b, c = cluster_stats(s, "resid_goals") if len(s) > 1 else (np.nan, np.nan, 0)
                    rec[f"z_{name[0]}"] = a / b if b else np.nan
                    rec[f"eff_{name[0]}"] = a
                seas = sub.groupby("season")["resid_goals"].mean()
                rec["seasons_same_sign"] = int((np.sign(seas) == np.sign(m)).sum())
                rows.append(rec)
    res = pd.DataFrame(rows)
    res["p"] = 2 * norm.sf(res["z"].abs())
    res["q_bh"] = benjamini_hochberg(res["p"].to_numpy())
    res["p_bonferroni"] = np.minimum(res["p"] * len(res), 1)
    return res


# States built only from these dimensions were effectively examined on 2024/25
# (12-cell HT grid, Fraudulent Leader / trailer tests), so they are not blind.
CONTAMINATED_DIMS = {"venue", "ht_score", "ht_xg"}


def pick_top(res: pd.DataFrame, df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Candidates for a blind test, by rule:

    1. replicate: same sign in 2020-21 and 2022-23, and in >= 3 of 4 seasons
    2. goals and xG agree: 2H xG effect has the same sign with |z| >= 1
       (excludes finishing-variance states)
    3. not contaminated: uses at least one dimension outside CONTAMINATED_DIMS
    4. ranked by the weaker half-sample |z|; skip states sharing > 50% of
       their matches with a higher-ranked pick
    """
    rep = res[(np.sign(res["z_A"]) == np.sign(res["z_B"]))
              & (np.sign(res["z_A"]) == np.sign(res["effect"]))
              & (res["seasons_same_sign"] >= 3)
              & (np.sign(res["xg_effect"]) == np.sign(res["effect"]))
              & (res["xg_z"].abs() >= 1)
              & ~res["dims"].apply(lambda d: set(d) <= CONTAMINATED_DIMS)].copy()
    rep["robust_z"] = np.minimum(rep["z_A"].abs(), rep["z_B"].abs())
    rep = rep.sort_values(["robust_z", "z"], ascending=False, key=lambda s: s.abs())
    chosen, chosen_ids = [], []
    for r in rep.itertuples():
        ids = set(df.loc[in_state(df, dict(zip(r.dims, r.values))), "match_id"])
        if any(len(ids & c) > 0.5 * min(len(ids), len(c)) for c in chosen_ids):
            continue
        chosen.append(r.Index)
        chosen_ids.append(ids)
        if len(chosen) == k:
            break
    return rep.loc[chosen]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(f"Building training rows ({TRAIN_SEASONS}) with walk-forward pre-match rates ...")
    df = build_rows()
    print(f"{df['match_id'].nunique()} matches, {len(df)} team rows; "
          f"r_goals={df.attrs['r_goals']:.4f}, r_xg={df.attrs['r_xg']:.4f}")
    print("Dimension levels: " + "; ".join(
        f"{d}: {', '.join(f'{k}({v})' for k, v in df[d].value_counts().items())}" for d in DIMENSIONS))

    res = search(df)
    n = len(res)
    print(f"\nTested {n} states (2-{MAX_DIMS} dimensions, > {MIN_MATCHES} matches each)")
    print(f"  |z| > 2: {(res['z'].abs() > 2).sum():4d} found vs ~{n * 2 * norm.sf(2):.0f} expected by chance "
          "(states overlap heavily, so these are correlated, not independent)")
    print(f"  |z| > 3: {(res['z'].abs() > 3).sum():4d} found vs ~{n * 2 * norm.sf(3):.1f} expected by chance")
    print(f"  Benjamini-Hochberg q < 0.10: {(res['q_bh'] < 0.10).sum()}   "
          f"Bonferroni p < 0.05: {(res['p_bonferroni'] < 0.05).sum()}")

    show = ["state", "n_matches", "effect", "se", "z", "xg_effect", "xg_z",
            "eff_A", "z_A", "eff_B", "z_B", "seasons_same_sign", "q_bh", "p_bonferroni"]
    print("\n=== Top 10 by raw |z| (no replication filter) ===")
    print(res.reindex(res["z"].abs().sort_values(ascending=False).index).head(10)[show].to_string(index=False))

    top = pick_top(res, df, TOP_K)
    print(f"\n=== Top {TOP_K} blind-test candidates (replicate in 2020-21 and 2022-23 and >= 3 of 4 "
          f"seasons; goals and xG agree; not contaminated; < 50% overlap) ===")
    if top.empty:
        print("  none")
    for i, r in enumerate(top.itertuples(), 1):
        print(f"\n#{i}  {r.state}")
        print(f"    n = {r.n_matches} matches ({r.n_rows} team rows)")
        print(f"    2H goal residual {r.effect:+.3f} per team (se {r.se:.3f}, z {r.z:+.2f}); "
              f"2H xG residual {r.xg_effect:+.3f} (z {r.xg_z:+.2f})")
        print(f"    2020-21: {r.eff_A:+.3f} (z {r.z_A:+.2f}) | 2022-23: {r.eff_B:+.3f} (z {r.z_B:+.2f}) | "
              f"same sign in {r.seasons_same_sign}/4 seasons")
        print(f"    multiple testing: BH q = {r.q_bh:.3f}, Bonferroni p = {r.p_bonferroni:.3f}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    frozen = [{
        "rank": i, "state": r.state, "conditions": dict(zip(r.dims, r.values)),
        "train_effect_goals": round(r.effect, 4), "train_se": round(r.se, 4),
        "train_effect_xg": round(r.xg_effect, 4), "train_n_matches": int(r.n_matches),
        "bh_q": round(r.q_bh, 4), "bonferroni_p": round(r.p_bonferroni, 4),
    } for i, r in enumerate(top.itertuples(), 1)]
    OUT_PATH.write_text(json.dumps({
        "train_seasons": TRAIN_SEASONS, "r_goals": df.attrs["r_goals"], "r_xg": df.attrs["r_xg"],
        "n_states_tested": n, "candidates": frozen}, indent=2))
    print(f"\nFrozen candidates written to {OUT_PATH}")
    return res


if __name__ == "__main__":
    run()
