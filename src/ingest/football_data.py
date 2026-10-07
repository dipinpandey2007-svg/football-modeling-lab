"""
football-data.co.uk source: results CSVs with exact half-time scores and odds.

Raw CSVs are cached as RAW_DIR / fd_{season}.csv (latin-1).
"""

from __future__ import annotations

from io import StringIO

import aiohttp
import pandas as pd

from .config import CURRENT_SEASON, FD_LEAGUE, RAW_DIR
from .storage import with_retries

FD_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/{league}.csv"

# football-data.co.uk name -> Understat name. Names not listed here must match
# Understat exactly; map_fd_team raises on anything it cannot map.
FD_TEAM_ALIASES = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield Weds": "Sheffield Wednesday",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
    "Spurs": "Tottenham",
    "QPR": "Queens Park Rangers",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Ipswich Town": "Ipswich",
    "Luton Town": "Luton",
}


def season_code(season: int) -> str:
    """2024 -> '2425' (football-data's URL segment for 2024/25)."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


async def fetch_season_fd(session: aiohttp.ClientSession, season: int, refresh: bool,
                          league: str = FD_LEAGUE) -> str:
    """Download (or read cached) the results CSV for one season."""
    path = RAW_DIR / f"fd_{season}.csv"
    if path.exists() and not refresh:
        return path.read_text(encoding="latin-1")
    url = FD_URL.format(season_code=season_code(season), league=league)

    async def get():
        async with session.get(url) as resp:  # aiohttp follows the www -> bare-domain redirect
            resp.raise_for_status()
            return (await resp.read()).decode("latin-1")

    text = await with_retries(get)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="latin-1")
    return text


def read_fd_csv(season: int) -> pd.DataFrame:
    """Cached raw football-data CSV for a season (all columns, incl. odds)."""
    text = (RAW_DIR / f"fd_{season}.csv").read_text(encoding="latin-1")
    return pd.read_csv(StringIO(text)).dropna(subset=["HomeTeam"])


def map_fd_team(name: str, known: set[str]) -> str:
    """football-data.co.uk team name -> Understat team name."""
    mapped = FD_TEAM_ALIASES.get(name.strip(), name.strip())
    if mapped not in known:
        raise KeyError(f"football-data team {name!r} has no Understat match; add it to FD_TEAM_ALIASES")
    return mapped


def fd_to_frame(fd_by_season: dict[int, str], matches: pd.DataFrame) -> pd.DataFrame:
    """Parse football-data CSVs and join them to Understat matches.

    Joined on (season, home team, away team), which is unique within a
    season. Raises if any fixture fails to join one-to-one (except in the
    in-progress CURRENT_SEASON, where it is skipped) or if full-time scores
    disagree, since either would mean the half-time scores can't be trusted.
    Returns match_id, fd_date, ht_home_goals, ht_away_goals.
    """
    known = set(matches["home_team"]) | set(matches["away_team"])
    frames = []
    for season, text in fd_by_season.items():
        fd = pd.read_csv(StringIO(text)).dropna(subset=["HomeTeam"])
        frames.append(pd.DataFrame({
            "season": season,
            "fd_date": pd.to_datetime(fd["Date"], dayfirst=True),
            "home_team": [map_fd_team(t, known) for t in fd["HomeTeam"]],
            "away_team": [map_fd_team(t, known) for t in fd["AwayTeam"]],
            "ht_home_goals": fd["HTHG"].astype(int),
            "ht_away_goals": fd["HTAG"].astype(int),
            "fd_home_goals": fd["FTHG"].astype(int),
            "fd_away_goals": fd["FTAG"].astype(int),
        }))
    fd = pd.concat(frames, ignore_index=True)

    m = matches[matches["season"].isin(fd_by_season)]
    joined = m.merge(fd, on=["season", "home_team", "away_team"], how="outer",
                     validate="one_to_one", indicator=True)
    unmatched = joined[joined["_merge"] != "both"]
    # In the season being played, one source can be a few matches ahead of the
    # other; those fixtures are simply skipped until both have them.
    lagging = unmatched["season"] == CURRENT_SEASON
    if lagging.any():
        print(f"  warning: {lagging.sum()} {CURRENT_SEASON} fixtures in only one source "
              f"(skipped until both have them)")
    if (~lagging).any():
        raise ValueError(f"{(~lagging).sum()} fixtures failed to join:\n"
                         f"{unmatched.loc[~lagging, ['season', 'home_team', 'away_team', '_merge']]}")
    joined = joined[joined["_merge"] == "both"]
    bad_ft = joined[(joined["home_goals"] != joined["fd_home_goals"])
                    | (joined["away_goals"] != joined["fd_away_goals"])]
    if len(bad_ft):
        raise ValueError(f"{len(bad_ft)} full-time scores differ between Understat and football-data")
    bad_ht = joined[(joined["ht_home_goals"] > joined["home_goals"])
                    | (joined["ht_away_goals"] > joined["away_goals"])]
    if len(bad_ht):
        raise ValueError(f"{len(bad_ht)} half-time scores exceed the full-time score")
    day_gap = (joined["date"].dt.normalize() - joined["fd_date"]).dt.days.abs()
    if (day_gap > 1).any():
        print(f"  warning: {(day_gap > 1).sum()} fixtures dated >1 day apart in the two sources")
    return joined[["match_id", "fd_date", "ht_home_goals", "ht_away_goals"]].astype({"match_id": int})
