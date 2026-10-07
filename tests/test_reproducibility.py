"""
Guards that keep the reported results reproducible:
analysis cutoff, immutable frozen candidates, raw-data checksum manifest.
"""

import hashlib
import json

import pandas as pd
import pytest

from src.ingest import manifest
from src.ingest.config import ANALYSIS_CUTOFF, PROCESSED_DIR, RAW_DIR

HAS_DATA = (PROCESSED_DIR / "matches.csv").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="data not built; run python -m src.data_fetcher")

# SHA-256 (\r removed) of results/frozen/ht_anomaly_candidates.json as frozen
# before the blind test. It must never change.
FROZEN_SHA256 = "f762f9d9b468a432859a7f2fa5685d5dd3a77cbddea4856b03e7b8f0466829c7"


# --- analysis cutoff ----------------------------------------------------------

def test_cutoff_is_the_reported_date():
    assert ANALYSIS_CUTOFF == "2026-09-20"


@needs_data
def test_default_cutoff_hides_later_matches(monkeypatch):
    from src.ingest.storage import load_matches
    monkeypatch.delenv("EPL_ANALYSIS_CUTOFF", raising=False)
    assert load_matches()["date"].max().normalize() <= pd.Timestamp(ANALYSIS_CUTOFF)


@needs_data
def test_cutoff_applies_to_every_table(monkeypatch):
    from src.ingest import storage
    monkeypatch.setenv("EPL_ANALYSIS_CUTOFF", "2026-08-31")
    ids = set(storage.load_matches()["match_id"])
    assert storage.load_matches()["date"].max() < pd.Timestamp("2026-09-01")
    for loader in (storage.load_half_time_stats, storage.load_shots,
                   storage.load_team_matches, storage.load_ht_scores):
        assert set(loader()["match_id"]) <= ids
    monkeypatch.setenv("EPL_ANALYSIS_CUTOFF", "none")
    assert len(storage.load_matches()) == len(pd.read_csv(PROCESSED_DIR / "matches.csv"))


# --- frozen anomaly candidates ------------------------------------------------

def test_frozen_candidates_are_unchanged():
    from src.experiments.ht_anomaly_search import FROZEN_PATH
    digest = hashlib.sha256(FROZEN_PATH.read_bytes().replace(b"\r", b"")).hexdigest()
    assert digest == FROZEN_SHA256
    frozen = json.loads(FROZEN_PATH.read_text())
    assert [c["state"] for c in frozen["candidates"]] == [
        "ht_xg=dominating & pre_match=underdog & rest=normal",
        "ht_score=level & pre_match=underdog & rest=normal & rest_diff=similar",
    ]


def test_search_never_writes_to_the_frozen_location():
    from src.experiments import ht_anomaly_blind_test
    from src.experiments.ht_anomaly_search import FROZEN_PATH, OUT_PATH
    assert OUT_PATH.resolve().parent != FROZEN_PATH.resolve().parent
    assert ht_anomaly_blind_test.FROZEN_PATH == FROZEN_PATH   # the blind test reads the frozen record


# --- raw-data manifest --------------------------------------------------------

def test_manifest_lists_every_season_and_source():
    names = set(manifest.read_manifest())
    expected = {f"{kind}_{s}.{ext}" for s in range(2019, 2027)
                for kind, ext in (("results", "json"), ("shots", "json"), ("teams", "json"), ("fd", "csv"))}
    assert names == expected


@pytest.mark.skipif(not RAW_DIR.exists(), reason="raw data not downloaded")
def test_completed_season_raw_data_matches_manifest():
    status = manifest.verify()
    completed = {k: v for k, v in status.items() if "_2026." not in k}
    assert all(v == "ok" for v in completed.values()), {k: v for k, v in completed.items() if v != "ok"}
