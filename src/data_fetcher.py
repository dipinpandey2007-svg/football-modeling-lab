"""
Data ingestion from Understat.

Pulls EPL match results (goals + xG) and per-shot data for a list of seasons,
caching the raw API responses under data/raw/ and writing tidy tables under
data/processed/. Re-running only hits the API for anything not already cached.

Understat labels a season by its starting year: 2019 -> 2019/20, 2024 -> 2024/25.

Outputs
-------
data/raw/results_{season}.json      raw league results (one list per season)
data/raw/shots_{season}.json        raw shots, {match_id: {"h": [...], "a": [...]}}
data/raw/teams_{season}.json        raw per-team match history (deep, PPDA, ...)
data/raw/fd_{season}.csv            football-data.co.uk results (exact half-time scores)
data/processed/ht_scores.csv        exact half-time score per match_id
data/processed/matches.csv          one row per match (see MATCH_COLUMNS)
data/processed/shots.csv            one row per shot, incl. minute and half
data/processed/half_time_stats.csv  one row per team per match: xG/shots/goals by half
data/processed/team_matches.csv     one row per team per match: deep completions
                                    and PPDA, keyed by match_id
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
from understat import Understat

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
# Season in progress: always re-downloaded, and fixtures present in only one
# source are skipped rather than treated as a join error.
CURRENT_SEASON = 2026
# Seasons whose shot minutes define the per-minute 1st-half probabilities
# used for the soft split; frozen before 2025/26 was added.
MINUTE_PROB_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)
LEAGUE = "epl"

# Polite concurrency limit for the per-match shot endpoint (~380 calls / season).
MAX_CONCURRENT_REQUESTS = 8
MAX_RETRIES = 4

FD_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"

# football-data.co.uk name -> Understat name. Names not listed here must match
# Understat exactly; fd_to_frame raises on anything it cannot map.
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

MATCH_COLUMNS = [
    "match_id", "date", "days_ago", "season",
    "home_team", "away_team",
    "home_xg", "away_xg",
    "home_goals", "away_goals",
]


# ---------------------------------------------------------------------------
# Raw fetching (async, cached)
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    tmp.replace(path)


async def _with_retries(coro_fn, *args):
    """Call an async Understat method with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_fn(*args)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(2 ** attempt)


async def _fetch_season_results(understat: Understat, season: int, refresh: bool) -> list[dict]:
    path = RAW_DIR / f"results_{season}.json"
    if path.exists() and not refresh:
        return _read_json(path)
    results = await _with_retries(understat.get_league_results, LEAGUE, season)
    _write_json(path, results)
    print(f"  fetched {len(results)} results for {season}")
    return results


async def _fetch_season_shots(understat: Understat, season: int, match_ids: list[str]) -> dict:
    """Fetch shots for every match in a season, skipping matches already cached."""
    path = RAW_DIR / f"shots_{season}.json"
    cache: dict = _read_json(path) if path.exists() else {}
    missing = [m for m in match_ids if m not in cache]
    if not missing:
        return cache

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def one(mid: str):
        async with sem:
            cache[mid] = await _with_retries(understat.get_match_shots, mid)

    print(f"  fetching shots for {len(missing)} matches in {season} ...")
    try:
        await asyncio.gather(*(one(m) for m in missing))
    finally:
        # Persist whatever we got, so an interrupted run can resume.
        _write_json(path, cache)
    return cache


async def _fetch_season_teams(understat: Understat, season: int, refresh: bool) -> list[dict]:
    """Per-team match histories; the only Understat source for deep completions."""
    path = RAW_DIR / f"teams_{season}.json"
    if path.exists() and not refresh:
        return _read_json(path)
    teams = await _with_retries(understat.get_teams, LEAGUE, season)
    _write_json(path, teams)
    return teams


async def _fetch_season_fd(session: aiohttp.ClientSession, season: int, refresh: bool) -> str:
    """football-data.co.uk results CSV (source of exact half-time scores)."""
    path = RAW_DIR / f"fd_{season}.csv"
    if path.exists() and not refresh:
        return path.read_text(encoding="latin-1")
    url = FD_URL.format(f"{season % 100:02d}{(season + 1) % 100:02d}")

    async def get():
        async with session.get(url) as resp:  # aiohttp follows the www -> bare-domain redirect
            resp.raise_for_status()
            return (await resp.read()).decode("latin-1")

    text = await _with_retries(get)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="latin-1")
    return text


async def _fetch_all(seasons: list[int], refresh: bool, with_shots: bool):
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        understat = Understat(session)
        results, shots, teams, fd = {}, {}, {}, {}
        for season in seasons:
            print(f"Season {season}/{(season + 1) % 100:02d}")
            # The season in progress is always re-downloaded; finished ones are cached.
            fresh = refresh or season == CURRENT_SEASON
            results[season] = await _fetch_season_results(understat, season, fresh)
            teams[season] = await _fetch_season_teams(understat, season, fresh)
            fd[season] = await _fetch_season_fd(session, season, fresh)
            if with_shots:
                ids = [r["id"] for r in results[season]]
                shots[season] = await _fetch_season_shots(understat, season, ids)
        return results, shots, teams, fd


# ---------------------------------------------------------------------------
# Cleaning / structuring
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


# Half assignment. Understat has no period field and minutes count from 0, so
# 1st-half stoppage time is logged as 45, 46, 47, ... and the 2nd half also
# starts at minute 45. The only direct evidence of the half-time boundary is
# the minute counter dropping back (e.g. 48 -> 46) in chronological order.
FIRST_HALF_END = 45
# Latest minute seen before a half-time drop in 2019-2025 data is 56; drops from
# later minutes are data-ordering glitches (one known case: 70 -> 46).
MAX_FIRST_HALF_MINUTE = 60
# Last minute that can plausibly be 1st-half stoppage time (max seen before a drop).
AMBIGUOUS_LAST_MINUTE = 56


def drop_boundary(minutes: list[int]) -> int | None:
    """Index of the first 2nd-half shot, from a minute drop; None if no drop.

    Walks the shots with current_half = 1, prev_minute = -1 and switches to
    half 2 on the first drop in minute that is plausibly half-time (from a
    minute in [45, 60] to a minute >= 45). It switches at most once, so later
    drops (2nd-half stoppage, extra time) are ignored. Only ~14% of matches
    have such a drop: it needs a 1st-half stoppage shot followed by an early
    2nd-half shot with a lower minute.
    """
    current_half, prev_minute = 1, -1
    for i, minute in enumerate(minutes):
        if (current_half == 1 and minute < prev_minute
                and FIRST_HALF_END <= prev_minute <= MAX_FIRST_HALF_MINUTE
                and minute >= FIRST_HALF_END):
            return i
        prev_minute = minute
    return None


def feasible_boundaries(minutes: list[int], scorers: list[str | None],
                        ht_score: tuple[int, int]) -> tuple[int, int] | None:
    """Range [lo, hi] of boundary indices k (shots[:k] = 1st half) consistent with
    the exact HT score and the minute constraints; None if no k is consistent.

    Constraints: 1st-half shots have minute <= MAX_FIRST_HALF_MINUTE, 2nd-half
    shots have minute >= FIRST_HALF_END, and the goals among shots[:k] equal
    the HT score. Each constraint holds on a contiguous range of k, so the
    feasible set is an interval.
    """
    n = len(minutes)
    cum_h = np.concatenate([[0], np.cumsum([s == "h" for s in scorers])])
    cum_a = np.concatenate([[0], np.cumsum([s == "a" for s in scorers])])
    m = np.asarray(minutes)
    prefix_max = np.concatenate([[-1], np.maximum.accumulate(m)]) if n else np.array([-1])
    suffix_min = np.concatenate([np.minimum.accumulate(m[::-1])[::-1], [10 ** 9]]) if n else np.array([10 ** 9])
    ok = ((cum_h == ht_score[0]) & (cum_a == ht_score[1])
          & (prefix_max <= MAX_FIRST_HALF_MINUTE) & (suffix_min >= FIRST_HALF_END))
    ks = np.flatnonzero(ok)
    return (int(ks[0]), int(ks[-1])) if len(ks) else None


def first_half_minute_probs(minute_counts: pd.Series) -> dict[int, float]:
    """P(a shot logged at minute m was in the 1st half), for m in 45..56.

    Minutes count from 0, so both 1st-half stoppage time and the 2nd half
    produce minutes 45+. The 2nd-half count at minute 45+j is estimated from
    1st-half minute j (same post-kickoff lull), scaled by the 2H/1H shot-rate
    ratio at minutes 50-54 vs 5-9. Whatever is left over is 1st-half stoppage
    time. Minutes <= 44 are certainly 1st half, >= 57 certainly 2nd half.
    """
    c = minute_counts.reindex(range(0, 120), fill_value=0)
    ratio = c.loc[50:54].sum() / c.loc[5:9].sum()
    probs = {}
    for m in range(FIRST_HALF_END, AMBIGUOUS_LAST_MINUTE + 1):
        est_2h = ratio * c.loc[m - FIRST_HALF_END]
        probs[m] = float(np.clip(1 - est_2h / max(c.loc[m], 1), 0.01, 0.99))
    return probs


def _p_first(minute: int, probs: dict[int, float]) -> float:
    if minute < FIRST_HALF_END:
        return 1.0
    return probs.get(minute, 0.0)


def boundary_posterior(minutes: list[int], lo: int, hi: int,
                       probs: dict[int, float]) -> np.ndarray:
    """Posterior over boundary k in [lo, hi] (shots[:k] = 1st half).

    log P(k) = sum_{i<k} log p1(m_i) + sum_{i>=k} log(1 - p1(m_i)), using the
    per-minute 1st-half probabilities (shots outside the window contribute
    the same constant to every k and are skipped).
    """
    p1 = np.array([_p_first(m, probs) for m in minutes[lo:hi]])
    with np.errstate(divide="ignore"):
        l1, l2 = np.log(p1), np.log(1 - p1)
    # k = lo + j: shots lo..lo+j-1 in half 1, lo+j..hi-1 in half 2
    score = np.array([l1[:j].sum() + l2[j:].sum() for j in range(hi - lo + 1)])
    w = np.exp(score - score.max())
    return w / w.sum()


def assign_halves(minutes: list[int], scorers: list[str | None] | None = None,
                  ht_score: tuple[int, int] | None = None,
                  probs: dict[int, float] | None = None):
    """Assign 1/2 to a match's shots, given in chronological order.

    scorers  : per shot, the side a goal counts for ("h"/"a"), None if no goal
               (an own goal counts for the shooter's opponent)
    ht_score : exact (home, away) half-time score, e.g. from football-data
    probs    : per-minute 1st-half probabilities (first_half_minute_probs);
               without them, a window is split at minute <= 45

    Returns (halves, source, uncertain, p_first):
      "drop"        minute drop consistent with the HT score -- exact
      "ht_score"    HT score + minute limits pin the boundary -- exact
      "ht_window"   boundary somewhere in a window of shots; the most likely
                    boundary under `probs` is used
      "ht_conflict" no boundary fits the HT score; falls back to the minute window
      "window"      no HT score and no drop; most likely boundary in the minute window
    uncertain[i] : shot i lies in the window (its half is not proven)
    p_first[i]   : posterior probability that shot i was in the 1st half
    """
    n = len(minutes)
    drop_k = drop_boundary(minutes)
    window = feasible_boundaries(minutes, scorers, ht_score) if ht_score is not None else None
    if window is not None and drop_k is not None and window[0] <= drop_k <= window[1]:
        window, source = (drop_k, drop_k), "drop"
    elif window is not None:
        source = "ht_score" if window[0] == window[1] else "ht_window"
    else:
        source = "ht_conflict" if ht_score is not None else "window"
        if drop_k is not None and ht_score is None:
            window, source = (drop_k, drop_k), "drop"
        else:
            window = feasible_boundaries(minutes, [None] * n, (0, 0))
    if window is None:  # minutes too disordered for any boundary: split at minute 45
        rule_k = next((i for i, m in enumerate(minutes) if m > FIRST_HALF_END), n)
        window = (rule_k, rule_k)

    lo, hi = window
    if probs is not None:
        post = boundary_posterior(minutes, lo, hi, probs)
        k = lo + int(post.argmax())
        # P(shot i in half 1) = P(k > i) for shots inside the window
        p_in = 1 - np.cumsum(post)[:-1] if hi > lo else np.array([])
    else:
        rule_k = next((i for i, m in enumerate(minutes) if m > FIRST_HALF_END), n)
        k = min(max(rule_k, lo), hi)
        p_in = np.array([float(i < k) for i in range(lo, hi)])
    p_first = [1.0 if i < lo else 0.0 if i >= hi else float(p_in[i - lo]) for i in range(n)]
    uncertain = [lo <= i < hi for i in range(n)]
    return [1 if i < k else 2 for i in range(n)], source, uncertain, p_first


def _goal_side(shot: dict) -> str | None:
    if shot["result"] == "Goal":
        return shot["h_a"]
    if shot["result"] == "OwnGoal":
        return "a" if shot["h_a"] == "h" else "h"
    return None


def shots_to_frame(shots_by_season: dict[int, dict],
                   ht_scores: dict[int, tuple[int, int]] | None = None) -> pd.DataFrame:
    """Flatten raw shot data into one row per shot, with the half it was in.

    Understat stores home and away shots in separate arrays; shot `id` is
    chronological across both, so each match's shots are merged and ordered
    by id before `assign_halves` runs. `ht_scores` maps match_id -> exact
    (home, away) half-time score and is used to calibrate the split.

    half_source     see assign_halves
    half_uncertain  True if the shot's half is still not pinned down
    p_first_half    posterior probability the shot was in the 1st half
                    (1/0 for pinned-down shots)
    """
    ht_scores = ht_scores or {}
    # Frozen to MINUTE_PROB_SEASONS so that adding a season never changes how
    # earlier seasons (or the new one) are split.
    minute_counts = pd.Series([int(s["minute"]) for season, by_match in shots_by_season.items()
                               if season in MINUTE_PROB_SEASONS
                               for sides in by_match.values() for side in ("h", "a")
                               for s in sides.get(side, [])]).value_counts()
    probs = first_half_minute_probs(minute_counts)
    rows = []
    for season, by_match in shots_by_season.items():
        for mid, sides in by_match.items():
            match_shots = sorted(sides.get("h", []) + sides.get("a", []), key=lambda s: int(s["id"]))
            halves, source, uncertain, p_first = assign_halves(
                [int(s["minute"]) for s in match_shots],
                [_goal_side(s) for s in match_shots],
                ht_scores.get(int(mid)), probs)
            for s, half, unc, p1 in zip(match_shots, halves, uncertain, p_first):
                minute = int(s["minute"])
                rows.append({
                    "shot_id": int(s["id"]),
                    "match_id": int(mid),
                    "season": int(season),
                    "date": pd.Timestamp(s["date"]),
                    "minute": minute,
                    "half": half,
                    "half_source": source,
                    "half_uncertain": unc,
                    "p_first_half": round(p1, 4),
                    "h_a": s["h_a"],
                    "team": s["h_team"] if s["h_a"] == "h" else s["a_team"],
                    "player": s["player"],
                    "player_id": int(s["player_id"]),
                    "xg": float(s["xG"]),
                    "result": s["result"],
                    "situation": s["situation"],
                    "shot_type": s["shotType"],
                    "last_action": s.get("lastAction"),
                    "x": float(s["X"]),
                    "y": float(s["Y"]),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Chronological (id) order within each match, so the half boundary is visible.
    return df.sort_values(["date", "match_id", "shot_id"]).reset_index(drop=True)


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
    from io import StringIO
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


HALF_STATS_COLUMNS = [
    "match_id", "team", "h_a",
    "half_1_xg", "half_1_shots", "half_1_goals",
    "half_2_xg", "half_2_shots", "half_2_goals",
    "half_1_xg_soft", "half_2_xg_soft",
    "split_exact", "uncertain_xg", "expected_error_xg",
]


def half_split_metrics(shots: pd.DataFrame, matches: pd.DataFrame,
                       ht: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-match, per-team xG / shots / goals for each half.

    One row per team per match (built from `matches`, so a team with no shots
    in a half gets zeros rather than a missing row).

    Goals: with `ht` (football-data half-time scores) half_1_goals is the exact
    HT score and half_2_goals = full time - half time. Without it, goals come
    from the shot split. Understat logs an own goal as a shot row (xG 0,
    result "OwnGoal") under the *conceding* team; it is excluded from that
    team's shots and, in the shot-based count, credited to the opponent.

    half_*_xg_soft     xG split by each shot's posterior P(1st half) instead of
                       its most-likely half; unbiased in aggregate (hard labels
                       put every minute-46 shot in half 2, though ~43% are half 1)
    split_exact        every shot in the match has a pinned-down half
    uncertain_xg       upper bound: this team's xG on shots whose half is not
                       pinned down (all of it could be in the wrong half)
    expected_error_xg  expected xG in the wrong half under the posterior:
                       sum of xG * P(shot is in the other half from its label)
    """
    s = shots.copy()
    own_goal = s["result"] == "OwnGoal"
    s["scoring_side"] = s["h_a"].where(~own_goal, s["h_a"].map({"h": "a", "a": "h"}))
    s["is_goal"] = s["result"].isin(["Goal", "OwnGoal"]).astype(int)

    attempts = (s[~own_goal].groupby(["match_id", "h_a", "half"])
                  .agg(xg=("xg", "sum"), shots=("xg", "size")))
    goals = s.groupby(["match_id", "scoring_side", "half"])["is_goal"].sum().rename("goals")
    goals.index = goals.index.set_names("h_a", level="scoring_side")
    agg = attempts.join(goals, how="outer").fillna(0).unstack("half", fill_value=0)
    agg.columns = [f"half_{half}_{metric}" for metric, half in agg.columns]

    keys = pd.concat([
        matches[["match_id", "home_team"]].rename(columns={"home_team": "team"}).assign(h_a="h"),
        matches[["match_id", "away_team"]].rename(columns={"away_team": "team"}).assign(h_a="a"),
    ])
    out = keys.merge(agg.reset_index(), on=["match_id", "h_a"], how="left")
    for col in HALF_STATS_COLUMNS[3:9]:
        out[col] = out.get(col, 0)
    out = out.fillna({c: 0 for c in HALF_STATS_COLUMNS[3:9]})
    if ht is not None:
        ft = pd.concat([
            matches[["match_id", "home_goals"]].rename(columns={"home_goals": "ft"}).assign(h_a="h"),
            matches[["match_id", "away_goals"]].rename(columns={"away_goals": "ft"}).assign(h_a="a"),
        ])
        exact = pd.concat([
            ht[["match_id", "ht_home_goals"]].rename(columns={"ht_home_goals": "ht"}).assign(h_a="h"),
            ht[["match_id", "ht_away_goals"]].rename(columns={"ht_away_goals": "ht"}).assign(h_a="a"),
        ]).merge(ft, on=["match_id", "h_a"])
        out = out.merge(exact, on=["match_id", "h_a"], how="left")
        has = out["ht"].notna()
        out.loc[has, "half_1_goals"] = out.loc[has, "ht"]
        out.loc[has, "half_2_goals"] = out.loc[has, "ft"] - out.loc[has, "ht"]

    for col in ("half_1_shots", "half_1_goals", "half_2_shots", "half_2_goals"):
        out[col] = out[col].astype(int)

    inexact = s.groupby("match_id")["half_uncertain"].any()
    out["split_exact"] = ~out["match_id"].map(inexact).fillna(False).astype(bool)
    unc = s[s["half_uncertain"] & ~own_goal].groupby(["match_id", "h_a"])["xg"].sum().rename("uncertain_xg")
    out = out.merge(unc.reset_index(), on=["match_id", "h_a"], how="left").fillna({"uncertain_xg": 0.0})
    soft = (s[~own_goal].assign(x1=lambda d: d["xg"] * d["p_first_half"],
                                x2=lambda d: d["xg"] * (1 - d["p_first_half"]))
              .groupby(["match_id", "h_a"])[["x1", "x2"]].sum()
              .rename(columns={"x1": "half_1_xg_soft", "x2": "half_2_xg_soft"}))
    out = out.merge(soft.reset_index(), on=["match_id", "h_a"], how="left").fillna(
        {"half_1_xg_soft": 0.0, "half_2_xg_soft": 0.0})
    p_wrong = np.where(s["half"] == 1, 1 - s["p_first_half"], s["p_first_half"])
    err = (s.assign(err=s["xg"] * p_wrong)[~own_goal]
             .groupby(["match_id", "h_a"])["err"].sum().rename("expected_error_xg"))
    out = out.merge(err.reset_index(), on=["match_id", "h_a"], how="left").fillna({"expected_error_xg": 0.0})
    return (out[HALF_STATS_COLUMNS]
            .sort_values(["match_id", "h_a"], ascending=[True, False]).reset_index(drop=True))


def half_split_diagnostics(shots: pd.DataFrame, half_stats: pd.DataFrame,
                           matches: pd.DataFrame) -> None:
    """Print checks on the half assignment and the half-time table."""
    old_half = (shots["minute"] > FIRST_HALF_END).astype(int) + 1
    mins = list(range(45, 50))
    table = pd.DataFrame({
        "all_shots": shots["minute"].value_counts().reindex(mins, fill_value=0),
        "half1_old_rule": shots.loc[old_half == 1, "minute"].value_counts().reindex(mins, fill_value=0),
        "half1_new": shots.loc[shots["half"] == 1, "minute"].value_counts().reindex(mins, fill_value=0),
        "half1_from_drop": shots.loc[(shots["half"] == 1) & (shots["half_source"] == "drop"), "minute"]
                                .value_counts().reindex(mins, fill_value=0),
    }).rename_axis("minute")
    print("=== Shots with half == 1, by minute (old rule: minute <= 45) ===")
    print(table.to_string(), "\n")

    by_match = shots.groupby("match_id").agg(source=("half_source", "first"),
                                             unc=("half_uncertain", "any"))
    print("Half-split source per match:")
    print(by_match["source"].value_counts().to_string())
    print(f"Matches with every shot's half pinned down: {(~by_match['unc']).sum()} / {len(by_match)}")
    probs = first_half_minute_probs(shots["minute"].value_counts())
    print("P(1st half | minute):  " + "  ".join(f"{m}:{p:.2f}" for m, p in probs.items()))
    n_matches = shots["match_id"].nunique()
    unc = shots[shots["half_uncertain"]]
    p_wrong = np.where(shots["half"] == 1, 1 - shots["p_first_half"], shots["p_first_half"])
    print(f"Shots in an unresolved window: {len(unc)} / {len(shots)}")
    print(f"uncertain_xg (upper bound):    {unc['xg'].sum():7.1f} xG total, "
          f"{unc['xg'].sum() / n_matches:.3f} per match")
    print(f"expected_error_xg (posterior): {(shots['xg'] * p_wrong).sum():7.1f} xG total, "
          f"{(shots['xg'] * p_wrong).sum() / n_matches:.3f} per match, "
          f"{p_wrong.sum():.0f} shots expected in the wrong half\n")

    # Estimate the true number of 1st-half shots per minute around half-time.
    # 2nd-half minute 45+j behaves like 1st-half minute j (kickoff lull), scaled
    # by the 2H/1H shot-rate ratio measured at minutes 50-54 vs 5-9.
    counts = shots["minute"].value_counts()
    ratio = counts.reindex(range(50, 55)).sum() / counts.reindex(range(5, 10)).sum()
    est = pd.DataFrame({"minute": range(45, 51)})
    est["all_shots"] = est["minute"].map(counts)
    est["est_2nd_half"] = (ratio * est["minute"].sub(45).map(counts)).round()
    est["est_1st_half"] = (est["all_shots"] - est["est_2nd_half"]).clip(lower=0)
    est["labelled_1st_half"] = est["minute"].map(shots.loc[shots["half"] == 1, "minute"].value_counts()).fillna(0).astype(int)
    print(f"=== Estimated vs labelled 1st-half shots (2H minute 45+j ~ {ratio:.2f} x minute j) ===")
    print(est.set_index("minute").to_string())
    gap = (est["est_1st_half"] - est["labelled_1st_half"]).sum()
    print(f"Net 1st-half shots still labelled 2nd half at minutes 45-50: ~{gap:.0f} "
          f"(~{gap / len(shots):.1%} of all shots, ~{gap / shots['match_id'].nunique():.2f} per match)\n")

    tot = half_stats.assign(g=half_stats["half_1_goals"] + half_stats["half_2_goals"])
    tot = tot.pivot(index="match_id", columns="h_a", values="g").reindex(matches["match_id"])
    ok = ((tot["h"].to_numpy() == matches["home_goals"].to_numpy())
          & (tot["a"].to_numpy() == matches["away_goals"].to_numpy()))
    print(f"half_1_goals + half_2_goals == final score: {ok.sum()} / {len(ok)} matches\n")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def fetch_data(seasons: list[int] = DEFAULT_SEASONS, refresh: bool = False,
               with_shots: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Fetch (or load from cache) matches and shots, write processed CSVs.

    refresh=True re-downloads season results (useful for an in-progress season);
    cached shots for already-fetched matches are always reused.
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


def load_matches() -> pd.DataFrame:
    """Load the processed match table written by fetch_data()."""
    return pd.read_csv(PROCESSED_DIR / "matches.csv", parse_dates=["date"])


def load_team_matches() -> pd.DataFrame:
    """Load the per-team match table (deep completions, PPDA) from fetch_data()."""
    return pd.read_csv(PROCESSED_DIR / "team_matches.csv", parse_dates=["date"])


def load_shots() -> pd.DataFrame:
    """Load the processed shot table written by fetch_data()."""
    return pd.read_csv(PROCESSED_DIR / "shots.csv", parse_dates=["date"])


def load_half_time_stats() -> pd.DataFrame:
    """Load the per-team half-time table written by fetch_data()."""
    return pd.read_csv(PROCESSED_DIR / "half_time_stats.csv")


if __name__ == "__main__":
    m, s = fetch_data()
    print(m.groupby("season").size())
    print(f"{len(s)} shots\n")
    ht = load_half_time_stats()
    half_split_diagnostics(s, ht, m)
    print("=== half_time_stats.csv (first 5 rows) ===")
    print(ht.head().to_string())
