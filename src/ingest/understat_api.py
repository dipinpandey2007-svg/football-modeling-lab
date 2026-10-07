"""
Understat source: league results, per-match shots and per-team match histories.

Raw responses are cached under RAW_DIR:
    results_{season}.json   league results (one list per season)
    shots_{season}.json     {match_id: {"h": [...], "a": [...]}}
    teams_{season}.json     per-team match history (deep, PPDA, ...)
"""

from __future__ import annotations

import asyncio

import pandas as pd
from understat import Understat

from .config import LEAGUE, MAX_CONCURRENT_REQUESTS, RAW_DIR
from .storage import read_json, with_retries, write_json

MATCH_COLUMNS = [
    "match_id", "date", "days_ago", "season",
    "home_team", "away_team",
    "home_xg", "away_xg",
    "home_goals", "away_goals",
]


# ---------------------------------------------------------------------------
# Downloads (async, cached)
# ---------------------------------------------------------------------------

async def fetch_season_results(understat: Understat, season: int, refresh: bool) -> list[dict]:
    path = RAW_DIR / f"results_{season}.json"
    if path.exists() and not refresh:
        return read_json(path)
    results = await with_retries(understat.get_league_results, LEAGUE, season)
    write_json(path, results)
    print(f"  fetched {len(results)} results for {season}")
    return results


async def fetch_season_shots(understat: Understat, season: int, match_ids: list[str]) -> dict:
    """Fetch shots for every match in a season, skipping matches already cached."""
    path = RAW_DIR / f"shots_{season}.json"
    cache: dict = read_json(path) if path.exists() else {}
    missing = [m for m in match_ids if m not in cache]
    if not missing:
        return cache

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def one(mid: str):
        async with sem:
            cache[mid] = await with_retries(understat.get_match_shots, mid)

    print(f"  fetching shots for {len(missing)} matches in {season} ...")
    try:
        await asyncio.gather(*(one(m) for m in missing))
    finally:
        # Persist whatever we got, so an interrupted run can resume.
        write_json(path, cache)
    return cache


async def fetch_season_teams(understat: Understat, season: int, refresh: bool) -> list[dict]:
    """Per-team match histories; the only Understat source for deep completions."""
    path = RAW_DIR / f"teams_{season}.json"
    if path.exists() and not refresh:
        return read_json(path)
    teams = await with_retries(understat.get_teams, LEAGUE, season)
    write_json(path, teams)
    return teams


# ---------------------------------------------------------------------------
# Table builders (pure)
# ---------------------------------------------------------------------------

def results_to_frame(results_by_season: dict[int, list[dict]]) -> pd.DataFrame:
    """Flatten raw Understat league results into the standard match table.

    `days_ago` is measured relative to the most recent match in the dataset.
    It is informational only: the model recomputes elapsed time relative to
    each prediction date, which is what the time-decay weighting needs.
    """
    rows = []
    for season, results in results_by_season.items():
        for r in results:
            if not r.get("isResult"):
                continue
            rows.append({
                "match_id": int(r["id"]),
                "date": pd.Timestamp(r["datetime"]),
                "season": int(season),
                "home_team": r["h"]["title"],
                "away_team": r["a"]["title"],
                "home_xg": float(r["xG"]["h"]),
                "away_xg": float(r["xG"]["a"]),
                "home_goals": int(r["goals"]["h"]),
                "away_goals": int(r["goals"]["a"]),
            })
    df = pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    df["days_ago"] = (df["date"].max().normalize() - df["date"].dt.normalize()).dt.days
    return df[MATCH_COLUMNS]


def teams_to_frame(teams_by_season: dict[int, list[dict]], matches: pd.DataFrame) -> pd.DataFrame:
    """Flatten per-team match histories into one row per team per match.

    Understat's team history has no match id, so rows are joined to `matches`
    on (kickoff datetime, team, venue).
    deep        passes completed within ~20 yards of the opponent's goal
    ppda        opponent passes allowed per defensive action (pressing intensity)
    """
    rows = []
    for season, teams in teams_by_season.items():
        for t in teams:
            for h in t["history"]:
                rows.append({
                    "date": pd.Timestamp(h["date"]),
                    "season": int(season),
                    "team": t["title"],
                    "h_a": h["h_a"],
                    "deep": int(h["deep"]),
                    "deep_allowed": int(h["deep_allowed"]),
                    "ppda_att": int(h["ppda"]["att"]),
                    "ppda_def": int(h["ppda"]["def"]),
                    "ppda_allowed_att": int(h["ppda_allowed"]["att"]),
                    "ppda_allowed_def": int(h["ppda_allowed"]["def"]),
                    "xpts": float(h["xpts"]),
                })
    df = pd.DataFrame(rows)
    keys = pd.concat([
        matches[["match_id", "date", "home_team"]].rename(columns={"home_team": "team"}).assign(h_a="h"),
        matches[["match_id", "date", "away_team"]].rename(columns={"away_team": "team"}).assign(h_a="a"),
    ])
    df = df.merge(keys, on=["date", "team", "h_a"], how="inner", validate="one_to_one")
    cols = ["match_id", "date", "season", "team", "h_a", "deep", "deep_allowed",
            "ppda_att", "ppda_def", "ppda_allowed_att", "ppda_allowed_def", "xpts"]
    return df[cols].sort_values(["date", "match_id", "h_a"], ascending=[True, True, False]).reset_index(drop=True)
