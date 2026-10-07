"""
Experiment 2 -- team style archetypes and the Style Interaction Matrix S.

Result: FAILED. Adding S_resid to Dixon-Coles predictions made 2024/25
predictions worse (see RESULTS.md, E2).

Diagnostic script, training seasons only (2019/20 - 2023/24; 2024/25 is held
out for the backtest).

Features (per team, entering each match, over its previous 10 matches):
    xg_density      xG / shots                       chance quality created
    opp_xg_density  opp xG / opp shots               chance quality conceded
    deep_ratio      deep / (deep + opp deep)         territorial dominance
    directness      xG / (deep + 1e-5)               xG per deep completion

Implementation notes
--------------------
- Each ratio is computed from 10-match rolling *sums* (= ratio of rolling
  means), not as a mean of per-match ratios. Per-match ratios are unstable:
  a match with 0 deep completions gives directness = xG / 1e-5.
- Rolling windows are shifted by one match, so a team's archetype for a match
  uses only information available before kick-off. This makes the labels
  usable as predictive features later without leakage.
- The window resets after a gap of more than 150 days (i.e. a relegation and
  later return), but carries over the normal summer break.
- KMeans labels are arbitrary; clusters are relabelled 0-3 in descending order
  of their deep_ratio centroid so output is stable between runs.

Style Interaction Matrix
------------------------
S_raw[i, j]: mean goal difference (for - against) of an archetype-i team
playing an archetype-j team. Every match is counted from both sides, which
cancels home advantage and makes S antisymmetric (diagonal = 0 by design).

S_raw mostly reflects team quality, since archetypes correlate with strength.
S_resid removes that: it is the mean of (actual GD - Dixon-Coles expected GD),
using out-of-sample walk-forward Dixon-Coles predictions. That residual
is the interaction effect a style modifier would need to capture.
2019/20 is excluded from S_resid because it has no prior data to fit on.

Usage:
    python -m src.experiments.style_archetypes                    # archetypes + S matrices
    python -m src.experiments.style_archetypes --no-residual      # skip the Dixon-Coles runs
    python -m src.experiments.style_archetypes --target goals --xi 0.0019   # PROJECT_CONTEXT 7.2 (goals S_resid)
    python -m src.experiments.style_archetypes --compare --xi 0.0019        # 7.2 integration test, 2024/25
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..data_fetcher import load_matches, load_shots, load_team_matches
from ..evaluation.backtest import summarise, walk_forward
from ..ingest.config import RESULTS_DIR
from ..model.dixon_coles import DixonColesModel

TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023]
FEATURES = ["xg_density", "opp_xg_density", "deep_ratio", "directness"]
RAW_COUNTS = ["xg", "shots", "opp_xg", "opp_shots", "deep", "opp_deep"]
WINDOW = 10
N_CLUSTERS = 4
COMPARE_SEASON = 2024  # held-out season for the integration test
RANDOM_STATE = 42
MAX_GAP_DAYS = 150
EPS = 1e-5

OUT_DIR = RESULTS_DIR / "archetypes"


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def team_match_table(matches: pd.DataFrame, shots: pd.DataFrame,
                     team_matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match with raw counts for that team and opponent."""
    home = matches.rename(columns={
        "home_team": "team", "away_team": "opp", "home_xg": "xg", "away_xg": "opp_xg",
        "home_goals": "goals", "away_goals": "opp_goals"}).assign(h_a="h")
    away = matches.rename(columns={
        "away_team": "team", "home_team": "opp", "away_xg": "xg", "home_xg": "opp_xg",
        "away_goals": "goals", "home_goals": "opp_goals"}).assign(h_a="a")
    cols = ["match_id", "date", "season", "team", "opp", "h_a", "xg", "opp_xg", "goals", "opp_goals"]
    tm = pd.concat([home[cols], away[cols]], ignore_index=True)

    n_shots = shots.groupby(["match_id", "h_a"]).size().rename("shots").reset_index()
    tm = tm.merge(n_shots, on=["match_id", "h_a"], how="left")
    opp_side = tm["h_a"].map({"h": "a", "a": "h"})
    tm = tm.merge(n_shots.rename(columns={"h_a": "opp_h_a", "shots": "opp_shots"}),
                  left_on=["match_id", opp_side.rename("opp_h_a")], right_on=["match_id", "opp_h_a"],
                  how="left").drop(columns="opp_h_a")
    tm[["shots", "opp_shots"]] = tm[["shots", "opp_shots"]].fillna(0).astype(int)

    tm = tm.merge(team_matches[["match_id", "h_a", "deep", "deep_allowed"]]
                  .rename(columns={"deep_allowed": "opp_deep"}),
                  on=["match_id", "h_a"], how="left", validate="one_to_one")
    return tm.sort_values(["team", "date"]).reset_index(drop=True)


def rolling_features(tm: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Add pre-match rolling-window style features (NaN until `window` matches)."""
    tm = tm.sort_values(["team", "date"]).copy()
    gap = tm.groupby("team")["date"].diff().dt.days
    tm["spell"] = (gap.isna() | (gap > MAX_GAP_DAYS)).groupby(tm["team"]).cumsum()

    sums = (tm.groupby(["team", "spell"])[RAW_COUNTS]
              .transform(lambda s: s.shift(1).rolling(window, min_periods=window).sum()))
    tm[FEATURES] = _ratios(sums)
    return tm.drop(columns="spell")


def _ratios(sums: pd.DataFrame) -> pd.DataFrame:
    """Style features from window sums of RAW_COUNTS (one formula for all callers)."""
    return pd.DataFrame({
        "xg_density": sums["xg"] / sums["shots"],
        "opp_xg_density": sums["opp_xg"] / sums["opp_shots"],
        "deep_ratio": sums["deep"] / (sums["deep"] + sums["opp_deep"]),
        "directness": sums["xg"] / (sums["deep"] + EPS),
    }, index=sums.index)


def features_as_of(tm: pd.DataFrame, ref_date: pd.Timestamp, window: int = WINDOW) -> pd.DataFrame:
    """Each team's style features using only matches strictly before ref_date.

    Matches `rolling_features` exactly for a team's next match, but is computed
    from a hard date cut-off, so no match on/after ref_date can leak in even
    if a team plays twice inside one prediction block. Teams without `window`
    matches in their current spell (e.g. newly promoted) are omitted.
    """
    past = tm[tm["date"] < ref_date].sort_values(["team", "date"])
    sums = {}
    for team, g in past.groupby("team"):
        if (ref_date - g["date"].iloc[-1]).days > MAX_GAP_DAYS:
            continue  # returning after a long absence: a new spell starts
        breaks = np.flatnonzero(g["date"].diff().dt.days.to_numpy() > MAX_GAP_DAYS)
        spell = g.iloc[breaks[-1]:] if len(breaks) else g
        if len(spell) >= window:
            sums[team] = spell[RAW_COUNTS].tail(window).sum()
    if not sums:
        return pd.DataFrame(columns=FEATURES)
    return _ratios(pd.DataFrame(sums).T)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def fit_archetypes(features: pd.DataFrame) -> tuple[StandardScaler, KMeans, np.ndarray]:
    """Fit scaler + KMeans; return them plus a map old_label -> ordered label.

    Use `assign_archetypes` to label new rows (e.g. the 2024/25 season)
    with the clusters fitted on the training seasons.
    """
    scaler = StandardScaler().fit(features[FEATURES])
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    km.fit(scaler.transform(features[FEATURES]))
    deep_idx = FEATURES.index("deep_ratio")
    order = np.argsort(-km.cluster_centers_[:, deep_idx])
    relabel = np.empty(N_CLUSTERS, dtype=int)
    relabel[order] = np.arange(N_CLUSTERS)
    return scaler, km, relabel


def assign_archetypes(features: pd.DataFrame, scaler: StandardScaler, km: KMeans,
                      relabel: np.ndarray) -> np.ndarray:
    return relabel[km.predict(scaler.transform(features[FEATURES]))]


def centroid_table(scaler: StandardScaler, km: KMeans, relabel: np.ndarray) -> pd.DataFrame:
    """Centroids in original units and in z-scores, rows ordered by archetype."""
    z = pd.DataFrame(km.cluster_centers_, columns=FEATURES, index=relabel).sort_index()
    raw = pd.DataFrame(scaler.inverse_transform(z), columns=FEATURES, index=z.index)
    return raw, z


def describe(z_row: pd.Series, cut: float = 0.5) -> str:
    tags = [f"{'high' if v > 0 else 'low'} {f}" for f, v in z_row.items() if abs(v) >= cut]
    return ", ".join(tags) if tags else "near average"


def member_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """10th-90th percentile of each feature among each archetype's members."""
    q = df.groupby("archetype")[FEATURES].quantile([0.1, 0.9]).unstack()
    return pd.DataFrame({f: [f"{lo:.3f}-{hi:.3f}" for lo, hi in zip(q[(f, 0.1)], q[(f, 0.9)])]
                         for f in FEATURES}, index=q.index)


def dominant_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    """Most frequent archetype per team, its share, and per-season modes."""
    counts = df.groupby(["team", "archetype"]).size().unstack(fill_value=0)
    out = pd.DataFrame({
        "dominant": counts.idxmax(axis=1),
        "share": counts.max(axis=1) / counts.sum(axis=1),
        "n": counts.sum(axis=1),
    })
    by_season = (df.groupby(["team", "season"])["archetype"]
                   .agg(lambda s: s.mode().iloc[0]).unstack())
    out["by_season"] = by_season.apply(
        lambda r: " ".join(f"{str(s)[2:]}:{int(a)}" for s, a in r.dropna().items()), axis=1)
    out[[f"pct_{k}" for k in counts.columns]] = (counts.div(counts.sum(axis=1), axis=0) * 100).round(0)
    return out.sort_values(["dominant", "share"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Style Interaction Matrix
# ---------------------------------------------------------------------------

def interaction_matrix(df: pd.DataFrame, value: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mean, standard error and count of `value` by (archetype, opp_archetype)."""
    g = df.groupby(["archetype", "opp_archetype"])[value]
    idx = pd.Index(range(N_CLUSTERS), name="archetype")
    cols = pd.Index(range(N_CLUSTERS), name="vs")
    shape = lambda s: s.unstack().reindex(index=idx, columns=cols)
    mean, n = shape(g.mean()), shape(g.size())
    se = shape(g.std() / np.sqrt(g.size()))
    return mean, se, n


def dixon_coles_expected_gd(matches: pd.DataFrame, seasons: list[int],
                            target: str = "xg", xi: float | None = None) -> pd.DataFrame:
    """Walk-forward (out-of-sample) Dixon-Coles mu - lambda for each match."""
    from ..evaluation.cache import cached_walk_forward
    preds = [cached_walk_forward(matches, s, target=target, xi=xi) for s in seasons]
    p = pd.concat(preds)
    return p[["match_id"]].assign(exp_home_gd=p["pred_home_xg"] - p["pred_away_xg"])


def labelled_pairs(feat: pd.DataFrame) -> pd.DataFrame:
    """Team-match rows where both sides carry an archetype label."""
    lab = feat[["match_id", "team", "archetype"]]
    pairs = feat.merge(lab.rename(columns={"team": "opp", "archetype": "opp_archetype"}),
                       on=["match_id", "opp"], how="inner")
    pairs["gd"] = pairs["goals"] - pairs["opp_goals"]
    pairs["xgd"] = pairs["xg"] - pairs["opp_xg"]
    return pairs


def residual_matrix(pairs: pd.DataFrame, target: str = "xg", xi: float | None = None):
    """S_resid = mean(actual GD - walk-forward Dixon-Coles expected GD).

    `target` must match the Dixon-Coles variant the matrix will later adjust,
    since the residual is defined relative to that model.
    Returns (mean, se, n, n_matches).
    """
    resid_seasons = [s for s in TRAIN_SEASONS if s > min(TRAIN_SEASONS)]
    exp = dixon_coles_expected_gd(load_matches(), resid_seasons, target, xi)
    r = pairs.merge(exp, on="match_id", how="inner")
    sign = np.where(r["h_a"] == "h", 1, -1)
    r["gd_resid"] = r["gd"] - sign * r["exp_home_gd"]
    return (*interaction_matrix(r, "gd_resid"), r["match_id"].nunique())


# ---------------------------------------------------------------------------
# Point-in-time tagging for backtests
# ---------------------------------------------------------------------------

def training_features() -> pd.DataFrame:
    """Rolling pre-match features for the training seasons (labelled rows only)."""
    matches = load_matches()
    matches = matches[matches["season"].isin(TRAIN_SEASONS)]
    tm = rolling_features(team_match_table(matches, load_shots(), load_team_matches()))
    return tm


class ArchetypeTagger:
    """Labels every team's archetype as of a date, with a fitted cluster model.

    The scaler/KMeans are fitted on the training seasons only; features for
    any date are computed by `features_as_of` from matches strictly before it.
    """

    def __init__(self, scaler: StandardScaler, km: KMeans, relabel: np.ndarray):
        self.scaler, self.km, self.relabel = scaler, km, relabel
        self.tm = team_match_table(load_matches(), load_shots(), load_team_matches())

    def clusters_as_of(self, ref_date: pd.Timestamp) -> dict[str, int]:
        feats = features_as_of(self.tm, ref_date)
        if feats.empty:
            return {}
        labels = assign_archetypes(feats, self.scaler, self.km, self.relabel)
        return dict(zip(feats.index, labels.tolist()))


def build_style_model(target: str = "xg", xi: float | None = None) -> tuple[ArchetypeTagger, pd.DataFrame]:
    """Fit archetypes + S_resid on training seasons; return (tagger, S_resid)."""
    feat = training_features().dropna(subset=FEATURES).copy()
    scaler, km, relabel = fit_archetypes(feat)
    feat["archetype"] = assign_archetypes(feat, scaler, km, relabel)
    s_resid = residual_matrix(labelled_pairs(feat), target, xi)[0]
    return ArchetypeTagger(scaler, km, relabel), s_resid


def format_matrix(mean: pd.DataFrame, se: pd.DataFrame, n: pd.DataFrame) -> pd.DataFrame:
    out = mean.copy().astype(object)
    for i in mean.index:
        for j in mean.columns:
            out.loc[i, j] = "-" if i == j else f"{mean.loc[i, j]:+.2f} +/-{se.loc[i, j]:.2f} (n={n.loc[i, j]})"
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(residual: bool = True, target: str = "xg", xi: float | None = None) -> dict:
    tm = training_features()
    feat = tm.dropna(subset=FEATURES).copy()
    print(f"Team-matches: {len(tm)} total, {len(feat)} with a full {WINDOW}-match window "
          f"({len(tm) - len(feat)} dropped: first {WINDOW} matches of each spell)\n")

    scaler, km, relabel = fit_archetypes(feat)
    feat["archetype"] = assign_archetypes(feat, scaler, km, relabel)

    # 1. Centroids / thresholds
    raw, z = centroid_table(scaler, km, relabel)
    raw["size"] = feat["archetype"].value_counts().sort_index()
    raw["profile"] = [describe(z.loc[k]) for k in z.index]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    print("=== Archetype centroids (original units) ===")
    print(raw.round(3).to_string(), "\n")
    print("=== Centroids (z-scores) ===")
    print(z.round(2).to_string(), "\n")
    print("=== Member ranges (10th-90th pct) ===")
    print(member_ranges(feat).to_string(), "\n")

    # 2. Dominant archetype per team
    dom = dominant_archetypes(feat)
    print("=== Dominant archetype per team (by_season = season:mode) ===")
    print(dom.round(2).to_string(), "\n")

    # 3. Style Interaction Matrix: both teams need a label for the match
    pairs = labelled_pairs(feat)

    results = {"team_features": feat, "dominant": dom, "centroids": raw, "centroids_z": z}
    print(f"=== Style Interaction Matrix S_raw: mean goal diff, row vs column "
          f"({pairs['match_id'].nunique()} matches with both sides labelled) ===")
    s_raw = interaction_matrix(pairs, "gd")
    print(format_matrix(*s_raw).to_string(), "\n")
    print("=== Same, xG difference ===")
    s_xg = interaction_matrix(pairs, "xgd")
    print(format_matrix(*s_xg).to_string(), "\n")
    results.update(S_raw=s_raw[0], S_xg=s_xg[0])

    if residual:
        print(f"Running walk-forward Dixon-Coles (target={target}) for 2020-2023 ...")
        *s_res, n_matches = residual_matrix(pairs, target, xi)
        print(f"=== S_resid: mean (actual GD - Dixon-Coles[{target}] expected GD), "
              f"{n_matches} matches ===")
        print(format_matrix(*s_res).to_string(), "\n")
        results["S_resid"] = s_res[0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat.to_csv(OUT_DIR / "team_match_archetypes.csv", index=False)
    dom.to_csv(OUT_DIR / "dominant_archetypes.csv")
    raw.to_csv(OUT_DIR / "centroids.csv")
    for k in ("S_raw", "S_xg", "S_resid"):
        if k in results:
            results[k].to_csv(OUT_DIR / f"{k}.csv")
    print(f"Saved outputs to {OUT_DIR}")
    return results


def compare_style(matches: pd.DataFrame, xi: float) -> None:
    """Walk-forward 2024/25: xG Dixon-Coles vs the same model + S_resid style shift.

    Archetypes and S_resid are fitted on 2019/20-2023/24 only. Differences are
    paired by match, so their standard errors are reported alongside.
    """
    print("Fitting archetypes and S_resid (xG Dixon-Coles residuals, 2020-2023) ...")
    tagger, s_resid = build_style_model(target="xg", xi=xi)
    print("S_resid (row = home archetype, col = away archetype):")
    print(s_resid.round(3).to_string(), "\n")

    runs = {
        "baseline_xg": dict(),
        "style_symmetric": dict(style_matrix=s_resid, style_split="symmetric"),
        "style_home_only": dict(style_matrix=s_resid, style_split="home"),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    preds = {}
    for name, kw in runs.items():
        print(f"Walk-forward {name} ...")
        preds[name] = walk_forward(
            matches, COMPARE_SEASON, verbose=False,
            model_factory=lambda kw=kw: DixonColesModel(xi=xi, target="xg", **kw),
            cluster_fn=tagger.clusters_as_of)
        preds[name].to_csv(RESULTS_DIR / f"predictions_{name}.csv", index=False)

    base = preds["baseline_xg"]
    tagged = base["home_cluster"].notna() & base["away_cluster"].notna()
    adjusted = tagged & (base["home_cluster"] != base["away_cluster"])
    print(f"\n{len(base)} matches: {tagged.sum()} with both teams tagged, "
          f"{adjusted.sum()} adjusted (different archetypes)")

    rows = {"previous baseline (reported)": {"log_loss": 0.9816, "rps": 0.2003}}
    for name, p in preds.items():
        s = summarise(p)
        rows[name] = {"log_loss": s.log_loss, "rps": s.rps, "accuracy": s.accuracy}
        if name != "baseline_xg":
            for metric in ("log_loss", "rps"):
                d = p[metric] - base[metric]
                rows[name][f"d_{metric}"] = d.mean()
                rows[name][f"d_{metric}_se"] = d.std() / np.sqrt(len(d))
    table = pd.DataFrame(rows).T
    print(f"\nOut-of-sample {COMPARE_SEASON}/{(COMPARE_SEASON + 1) % 100:02d} "
          f"(d_* = style minus baseline, negative is better):")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(table.replace({np.nan: None}).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-residual", action="store_true",
                    help="skip the Dixon-Coles residual matrix")
    ap.add_argument("--target", choices=["goals", "xg"], default="xg",
                    help="Dixon-Coles variant the residuals are measured against")
    ap.add_argument("--xi", type=float, default=None,
                    help="Dixon-Coles time decay; default: tuned value (OPTIMAL_XI). "
                         "The originally reported results used 0.0019.")
    ap.add_argument("--compare", action="store_true",
                    help=f"run the {COMPARE_SEASON}/{(COMPARE_SEASON + 1) % 100:02d} integration test "
                         "(xG baseline vs S_resid-adjusted model)")
    args = ap.parse_args()
    if args.compare:
        compare_style(load_matches(), args.xi)
    else:
        run(residual=not args.no_residual, target=args.target, xi=args.xi)
