"""
Regression tests against the real processed data (skipped if data/ is absent).

These pin results that earlier analysis depended on, so a refactor or a
data rebuild that silently changes them fails loudly.
"""

import numpy as np
import pandas as pd
import pytest

from src.ingest.config import PROCESSED_DIR

pytestmark = pytest.mark.skipif(not (PROCESSED_DIR / "matches.csv").exists(),
                                reason="processed data not built; run python -m src.data_fetcher")


def test_completed_seasons_have_380_matches():
    from src.data_fetcher import load_matches
    counts = load_matches().groupby("season").size()
    for season in range(2019, 2026):
        assert counts[season] == 380


def test_half_time_goals_add_up_to_full_time():
    from src.data_fetcher import load_half_time_stats, load_matches
    m = load_matches().set_index("match_id")
    ht = load_half_time_stats()
    tot = ht.assign(g=ht.half_1_goals + ht.half_2_goals).pivot(index="match_id", columns="h_a", values="g")
    tot = tot.reindex(m.index)
    assert (tot["h"] == m["home_goals"]).all() and (tot["a"] == m["away_goals"]).all()


def test_half_time_goals_match_football_data_exactly():
    from src.data_fetcher import load_half_time_stats
    from src.ingest.storage import load_ht_scores
    ht = load_half_time_stats().pivot(index="match_id", columns="h_a", values="half_1_goals")
    fd = load_ht_scores().set_index("match_id")
    joined = fd.join(ht, how="inner")
    assert len(joined) == len(fd)
    assert (joined["h"] == joined["ht_home_goals"]).all() and (joined["a"] == joined["ht_away_goals"]).all()


def test_soft_split_preserves_shot_xg_totals():
    from src.data_fetcher import load_half_time_stats
    ht = load_half_time_stats()
    hard = ht.half_1_xg + ht.half_2_xg
    soft = ht.half_1_xg_soft + ht.half_2_xg_soft
    assert np.allclose(hard, soft, atol=1e-9)


def test_xg_model_2024_backtest_log_loss_is_unchanged():
    """Headline held-out result of the xi tuning (reported as 0.9658)."""
    from src.evaluation.backtest import summarise
    from src.evaluation.cache import cached_walk_forward
    from src.data_fetcher import load_matches
    preds = cached_walk_forward(load_matches(), 2024, target="xg")
    assert round(summarise(preds)["log_loss"], 4) == 0.9658


def test_cache_returns_identical_predictions(monkeypatch):
    from src.evaluation.cache import cached_walk_forward
    from src.data_fetcher import load_matches
    matches = load_matches()
    cached = cached_walk_forward(matches, 2026, target="xg")
    monkeypatch.setenv("EPL_NO_CACHE", "1")
    fresh = cached_walk_forward(matches, 2026, target="xg")
    pd.testing.assert_frame_equal(cached.reset_index(drop=True), fresh.reset_index(drop=True))
