"""
Data pipeline orchestrator: download, join and write the processed tables.

The work is done by the modules in src/ingest/:
    config         paths, seasons, request limits
    storage        persistence helpers and table loaders
    understat_api  Understat results / shots / team histories
    football_data  football-data.co.uk results (exact HT scores, odds)
    halves         half-time boundary inference and per-half team stats

This module runs them in order and re-exports their public names, so
existing imports such as `from src.data_fetcher import load_matches` work
unchanged.

Outputs
-------
data/raw/results_{season}.json      raw Understat league results
data/raw/shots_{season}.json        raw Understat shots, {match_id: {"h": [...], "a": [...]}}
data/raw/teams_{season}.json        raw Understat per-team match history (deep, PPDA, ...)
data/raw/fd_{season}.csv            football-data.co.uk results (exact half-time scores, odds)
data/processed/matches.csv          one row per match (see MATCH_COLUMNS)
data/processed/team_matches.csv     one row per team per match: deep completions, PPDA
data/processed/ht_scores.csv        exact half-time score per match_id
data/processed/shots.csv            one row per shot, incl. minute and inferred half
data/processed/half_time_stats.csv  one row per team per match: xG/shots/goals by half

Usage:
    python -m src.data_fetcher
"""

from __future__ import annotations

import asyncio

import aiohttp
import pandas as pd
from understat import Understat

from .ingest.config import (CACHE_DIR, CURRENT_SEASON, DATA_DIR, DEFAULT_SEASONS, LEAGUE,
                            MINUTE_PROB_SEASONS, PROCESSED_DIR, RAW_DIR)
from .ingest.football_data import (FD_TEAM_ALIASES, FD_URL, fd_to_frame, fetch_season_fd,
                                   map_fd_team, read_fd_csv)
from .ingest.halves import (AMBIGUOUS_LAST_MINUTE, FIRST_HALF_END, HALF_STATS_COLUMNS,
                            MAX_FIRST_HALF_MINUTE, assign_halves, boundary_posterior,
                            drop_boundary, feasible_boundaries, first_half_minute_probs,
                            goal_side, half_split_diagnostics, half_split_metrics, shots_to_frame)
from .ingest.storage import (load_half_time_stats, load_ht_scores, load_matches,
                             load_shots, load_team_matches)
from .ingest.understat_api import (MATCH_COLUMNS, fetch_season_results, fetch_season_shots,
                                   fetch_season_teams, results_to_frame, teams_to_frame)

_goal_side = goal_side  # backwards-compatible alias

# Public API re-exported from src.ingest (kept stable for existing imports).
__all__ = [
    # config
    "CACHE_DIR", "CURRENT_SEASON", "DATA_DIR", "DEFAULT_SEASONS", "LEAGUE",
    "MINUTE_PROB_SEASONS", "PROCESSED_DIR", "RAW_DIR",
    # football-data
    "FD_TEAM_ALIASES", "FD_URL", "fd_to_frame", "map_fd_team", "read_fd_csv",
    # halves
    "AMBIGUOUS_LAST_MINUTE", "FIRST_HALF_END", "HALF_STATS_COLUMNS", "MAX_FIRST_HALF_MINUTE",
    "assign_halves", "boundary_posterior", "drop_boundary", "feasible_boundaries",
    "first_half_minute_probs", "goal_side", "half_split_diagnostics", "half_split_metrics",
    "shots_to_frame",
    # storage
    "load_half_time_stats", "load_ht_scores", "load_matches", "load_shots", "load_team_matches",
    # understat
    "MATCH_COLUMNS", "results_to_frame", "teams_to_frame",
    # orchestration
    "fetch_data",
]


async def _fetch_all(seasons: list[int], refresh: bool, with_shots: bool):
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        understat = Understat(session)
        results, shots, teams, fd = {}, {}, {}, {}
        for season in seasons:
            print(f"Season {season}/{(season + 1) % 100:02d}")
            # The season in progress is always re-downloaded; finished ones are cached.
            fresh = refresh or season == CURRENT_SEASON
            results[season] = await fetch_season_results(understat, season, fresh)
            teams[season] = await fetch_season_teams(understat, season, fresh)
            fd[season] = await fetch_season_fd(session, season, fresh)
            if with_shots:
                ids = [r["id"] for r in results[season]]
                shots[season] = await fetch_season_shots(understat, season, ids)
        return results, shots, teams, fd


def fetch_data(seasons: list[int] = DEFAULT_SEASONS, refresh: bool = False,
               with_shots: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Fetch (or load from cache) all sources and write the processed CSVs.

    refresh=True re-downloads results for every season (the in-progress season
    is always refreshed); cached shots for already-fetched matches are reused.
    """
    results, shots, teams, fd = asyncio.run(_fetch_all(list(seasons), refresh, with_shots))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    matches = results_to_frame(results)
    matches.to_csv(PROCESSED_DIR / "matches.csv", index=False)
    team_matches = teams_to_frame(teams, matches)
    if len(team_matches) != 2 * len(matches):
        print(f"  warning: {len(team_matches)} team-match rows for {len(matches)} matches")
    team_matches.to_csv(PROCESSED_DIR / "team_matches.csv", index=False)
    ht = fd_to_frame(fd, matches)
    ht.to_csv(PROCESSED_DIR / "ht_scores.csv", index=False)

    shots_df = None
    if with_shots:
        ht_scores = {int(r.match_id): (int(r.ht_home_goals), int(r.ht_away_goals))
                     for r in ht.itertuples()}
        shots_df = shots_to_frame(shots, ht_scores)
        shots_df.to_csv(PROCESSED_DIR / "shots.csv", index=False)
        half_split_metrics(shots_df, matches, ht).to_csv(PROCESSED_DIR / "half_time_stats.csv", index=False)
    return matches, shots_df


if __name__ == "__main__":
    m, s = fetch_data()
    print(m.groupby("season").size())
    print(f"{len(s)} shots\n")
    ht = load_half_time_stats()
    half_split_diagnostics(s, ht, m)
    print("=== half_time_stats.csv (first 5 rows) ===")
    print(ht.head().to_string())
