"""Persistence helpers, a retry wrapper for async downloads, and table loaders."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import aiohttp
import pandas as pd

from .config import ANALYSIS_CUTOFF, MAX_RETRIES, PROCESSED_DIR


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    """Write atomically (temp file + rename) so an interrupted run never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    tmp.replace(path)


async def with_retries(coro_fn, *args):
    """Await coro_fn(*args) with exponential backoff on network / decode errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_fn(*args)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Loaders for the processed tables written by data_fetcher.fetch_data()
#
# Every loader applies the analysis cutoff (config.ANALYSIS_CUTOFF): matches
# played after it are dropped, and the per-match tables keep only the
# remaining match_ids. The processed CSVs on disk are never modified.
# ---------------------------------------------------------------------------

def analysis_cutoff() -> pd.Timestamp | None:
    """Last date (inclusive) of matches visible to analyses; None = no cutoff."""
    value = os.environ.get("EPL_ANALYSIS_CUTOFF", ANALYSIS_CUTOFF).strip()
    if value.lower() in ("", "none"):
        return None
    return pd.Timestamp(value).normalize()


def _within_cutoff(dates: pd.Series) -> pd.Series:
    cutoff = analysis_cutoff()
    if cutoff is None:
        return pd.Series(True, index=dates.index)
    return dates.dt.normalize() <= cutoff


def load_matches() -> pd.DataFrame:
    """One row per match: ids, date, season, teams, xG, goals."""
    df = pd.read_csv(PROCESSED_DIR / "matches.csv", parse_dates=["date"])
    return df[_within_cutoff(df["date"])].reset_index(drop=True)


def _visible_match_ids() -> set[int]:
    return set(load_matches()["match_id"])


def _keep_visible(df: pd.DataFrame) -> pd.DataFrame:
    if analysis_cutoff() is None:
        return df
    return df[df["match_id"].isin(_visible_match_ids())].reset_index(drop=True)


def load_team_matches() -> pd.DataFrame:
    """One row per team per match: deep completions, PPDA, xpts."""
    return _keep_visible(pd.read_csv(PROCESSED_DIR / "team_matches.csv", parse_dates=["date"]))


def load_shots() -> pd.DataFrame:
    """One row per shot, with its inferred half."""
    return _keep_visible(pd.read_csv(PROCESSED_DIR / "shots.csv", parse_dates=["date"]))


def load_half_time_stats() -> pd.DataFrame:
    """One row per team per match: xG / shots / goals by half."""
    return _keep_visible(pd.read_csv(PROCESSED_DIR / "half_time_stats.csv"))


def load_ht_scores() -> pd.DataFrame:
    """Exact half-time score per match_id (football-data.co.uk)."""
    return _keep_visible(pd.read_csv(PROCESSED_DIR / "ht_scores.csv", parse_dates=["fd_date"]))
