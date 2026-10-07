"""Paths, season constants and request limits shared by the ingestion modules."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Understat labels a season by its starting year: 2019 -> 2019/20.
DEFAULT_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
# Season in progress: always re-downloaded, and fixtures present in only one
# source are skipped rather than treated as a join error.
CURRENT_SEASON = 2026
# Seasons whose shot minutes define the per-minute 1st-half probabilities
# used for the soft split; frozen before 2025/26 was added.
MINUTE_PROB_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)

LEAGUE = "epl"          # Understat league code
FD_LEAGUE = "E0"        # football-data.co.uk division code

# Polite concurrency limit for the per-match shot endpoint (~380 calls / season).
MAX_CONCURRENT_REQUESTS = 8
MAX_RETRIES = 4
