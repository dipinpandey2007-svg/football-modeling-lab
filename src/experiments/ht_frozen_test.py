"""
Experiment 6 -- frozen out-of-sample test of the half-time model on 2025/26.

Result: FAILED (RESULTS.md, E6). --season 2026 is an exploratory check on
2026/27 matches up to the analysis cutoff (2026-09-20), not a test.

Every parameter below was fixed before any 2025/26 data was loaded:
xi chosen by optimize_xi on 2020/21-2023/24; r, the leader/trailer
adjustments and the cap fitted on 2020/21-2023/24 by ht_backtest. Only the
pre-match Dixon-Coles team ratings are refitted (walk-forward, before each
2025/26 matchweek, on matches already played) -- that is the model operating
normally, not re-tuning. The soft xG split uses per-minute probabilities
frozen to 2019/20-2024/25 shot minutes (data_fetcher.MINUTE_PROB_SEASONS).

At half-time, each team's expected 2nd-half goals are

    naive  : R * pre-match rate
    model  : naive + DELTA_LEADER   if leading at HT with HT xG deficit in (0, CAP)
                   + DELTA_TRAILER  if trailing at HT with HT xG edge in (0, CAP)

Scored by Poisson log-loss of the actual 2nd-half score (both teams summed),
paired by match.

Usage:
    python -m src.experiments.ht_frozen_test                 # 2025/26
    python -m src.experiments.ht_frozen_test --season 2026   # season in progress (exploratory)
"""

from __future__ import annotations

import argparse

import pandas as pd

from ..data_fetcher import load_matches
from .ht_leader_rule import CAP, compare, poisson_log_loss, test_team_rows

# ---- frozen parameters (do not tune on 2025/26) ----------------------------
TEST_SEASON = 2025
XI = 0.0046            # pre-match xG-fitted Dixon-Coles time decay
R = 0.5524             # 2H goals / pre-match expected goals, 2020/21-2023/24
DELTA_LEADER = -0.205  # leader with HT xG deficit in (0, CAP)
DELTA_TRAILER = +0.112  # trailer with HT xG edge in (0, CAP)
assert CAP == 0.5
# ----------------------------------------------------------------------------


def run(season: int = TEST_SEASON) -> pd.DataFrame:
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(f"Frozen: xi={XI}, r={R}, leader {DELTA_LEADER:+.3f}, trailer {DELTA_TRAILER:+.3f}, "
          f"cap {CAP} xG")
    print(f"Walk-forward pre-match Dixon-Coles for {season}/{(season + 1) % 100:02d} ...\n")

    df = test_team_rows(load_matches(), season=season, xi=XI)
    df["exp_naive"] = R * df["pre_rate"]
    df["exp_model"] = (df["exp_naive"] + DELTA_LEADER * df["leader_step"]
                       + DELTA_TRAILER * df["trailer_step"])
    models = {"naive": "Naive baseline", "model": "Capped leader + trailer"}
    for m in models:
        df[f"ll_{m}"] = poisson_log_loss(df["half_2_goals"].to_numpy(), df[f"exp_{m}"].to_numpy())

    print(f"Mean 2H goals per team: actual {df['half_2_goals'].mean():.3f} | "
          f"naive {df['exp_naive'].mean():.3f}\n")
    print(f"=== {season}/{(season + 1) % 100:02d}, all {df['match_id'].nunique()} matches: 2H Poisson log-loss per match "
          f"(diff < 0 = model better) ===")
    print(compare(df, models).to_string(), "\n")

    lead = df[df["leader_step"]]
    ids = lead["match_id"].unique()
    opp = df[df["match_id"].isin(ids) & ~df["leader_step"]]
    print(f"=== {season}/{(season + 1) % 100:02d} triggered matches: {len(ids)} fraudulent leaders "
          f"(HT lead, HT xG deficit in (0, {CAP})) ===")
    print(f"  leaders:   actual 2H goals {lead['half_2_goals'].mean():.3f} | naive "
          f"{lead['exp_naive'].mean():.3f} | model {lead['exp_model'].mean():.3f}")
    print(f"  trailers:  actual 2H goals {opp['half_2_goals'].mean():.3f} | naive "
          f"{opp['exp_naive'].mean():.3f} | model {opp['exp_model'].mean():.3f}")
    print(compare(df, models, ids).to_string(), "\n")

    print("=== Triggered matches ===")
    print(triggered_table(df, ids).to_string(index=False))
    return df


def triggered_table(df: pd.DataFrame, ids) -> pd.DataFrame:
    """One line per fraudulent-leader match: HT state, 2H expectations and outcome."""
    from ..data_fetcher import load_matches
    m = load_matches().set_index("match_id")
    rows = []
    for mid in ids:
        lead = df[(df["match_id"] == mid) & df["leader_step"]].iloc[0]
        trail = df[(df["match_id"] == mid) & ~df["leader_step"]].iloc[0]
        g = df[df["match_id"] == mid]
        rows.append({
            "date": m.loc[mid, "date"].date(),
            "match": f"{m.loc[mid, 'home_team']} v {m.loc[mid, 'away_team']}",
            "leader": lead["team"],
            "HT": f"{m.loc[mid, 'home_goals'] - g.loc[g['h_a'] == 'h', 'half_2_goals'].iloc[0]}-"
                  f"{m.loc[mid, 'away_goals'] - g.loc[g['h_a'] == 'a', 'half_2_goals'].iloc[0]}",
            "HT_xG_deficit": round(lead["leader_deficit"], 2),
            "FT": f"{m.loc[mid, 'home_goals']}-{m.loc[mid, 'away_goals']}",
            "ldr_2H_goals": int(lead["half_2_goals"]),
            "ldr_exp_naive": round(lead["exp_naive"], 2),
            "ldr_exp_model": round(lead["exp_model"], 2),
            "trl_2H_goals": int(trail["half_2_goals"]),
            "trl_exp_naive": round(trail["exp_naive"], 2),
            "trl_exp_model": round(trail["exp_model"], 2),
            "ll_naive": round(g["ll_naive"].sum(), 3),
            "ll_model": round(g["ll_model"].sum(), 3),
        })
    return pd.DataFrame(rows).sort_values("date")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=TEST_SEASON)
    run(ap.parse_args().season)
