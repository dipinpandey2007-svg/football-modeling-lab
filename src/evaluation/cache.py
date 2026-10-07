"""
Disk cache for walk-forward predictions.

A full-season walk-forward refits Dixon-Coles ~40 times (several seconds per
season), and most scripts rerun the same default fits. `cached_walk_forward`
stores each season's predictions under data/cache/walk_forward/, keyed by:

    - the season and model settings (target, resolved xi, newcomer prior weight,
      matchweek span)
    - a hash of every match up to the end of that season (so new or corrected
      data invalidates the entry, including the in-progress season)
    - a hash of the source of model/dixon_coles.py and evaluation/backtest.py (so a code change
      to the model or the walk-forward loop invalidates every entry)

Only plain model configurations are cached; runs with a custom model factory
or a cluster_fn should call backtest.walk_forward directly.

Set EPL_NO_CACHE=1 to bypass the cache; `clear_cache()` deletes it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pandas as pd

from .backtest import walk_forward
from ..model.dixon_coles import DixonColesModel
from ..ingest.config import CACHE_DIR

WALK_FORWARD_DIR = CACHE_DIR / "walk_forward"
_SRC = Path(__file__).resolve().parent.parent
_CODE_FILES = [_SRC / "model" / "dixon_coles.py", _SRC / "evaluation" / "backtest.py"]


def _code_hash() -> str:
    h = hashlib.sha256()
    for f in _CODE_FILES:
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _data_hash(matches: pd.DataFrame, season: int) -> str:
    """Hash of every match the walk-forward for `season` can see."""
    end = matches.loc[matches["season"] == season, "date"].max()
    visible = matches[matches["date"] <= end].sort_values(["date", "match_id"])
    row_hashes = pd.util.hash_pandas_object(visible, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:16]


def cache_key(matches: pd.DataFrame, season: int, target: str = "xg", xi: float | None = None,
              newcomer_prior_weight: float = 3.0, matchweek_span_days: int = 3) -> str:
    resolved_xi = DixonColesModel(xi=xi, target=target).xi
    settings = f"s{season}_{target}_xi{resolved_xi}_npw{newcomer_prior_weight}_span{matchweek_span_days}"
    return f"{settings}_{_data_hash(matches, season)}_{_code_hash()}"


def cached_walk_forward(matches: pd.DataFrame, season: int, target: str = "xg",
                        xi: float | None = None, newcomer_prior_weight: float = 3.0,
                        matchweek_span_days: int = 3, verbose: bool = False) -> pd.DataFrame:
    """backtest.walk_forward for a plain DixonColesModel, memoised on disk."""
    def compute() -> pd.DataFrame:
        return walk_forward(
            matches, season, verbose=verbose, matchweek_span_days=matchweek_span_days,
            model_factory=lambda: DixonColesModel(
                xi=xi, target=target, newcomer_prior_weight=newcomer_prior_weight))

    if os.environ.get("EPL_NO_CACHE") == "1":
        return compute()
    path = WALK_FORWARD_DIR / f"{cache_key(matches, season, target, xi, newcomer_prior_weight, matchweek_span_days)}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    preds = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    preds.to_pickle(tmp)
    tmp.replace(path)
    return preds


def clear_cache() -> None:
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect or clear the prediction cache")
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()
    if args.clear:
        clear_cache()
        print(f"Cleared {CACHE_DIR}")
    else:
        files = sorted(WALK_FORWARD_DIR.glob("*.pkl")) if WALK_FORWARD_DIR.exists() else []
        size = sum(f.stat().st_size for f in files) / 1e6
        print(f"{len(files)} cached walk-forward runs in {WALK_FORWARD_DIR} ({size:.1f} MB)")
        for f in files:
            print(f"  {f.name}")
