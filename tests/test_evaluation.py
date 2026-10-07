"""Metrics and walk-forward mechanics on synthetic data."""

import numpy as np
import pandas as pd

from src.evaluation.backtest import assign_matchweeks, log_loss, rps, walk_forward
from src.model.dixon_coles import DixonColesModel
from tests.test_dixon_coles import synthetic_matches


def test_log_loss_known_values():
    probs = np.array([[1 / 3, 1 / 3, 1 / 3], [0.5, 0.3, 0.2]])
    out = log_loss(probs, np.array([0, 2]))
    assert np.allclose(out, [np.log(3), -np.log(0.2)])


def test_rps_known_values():
    probs = np.array([[0.5, 0.3, 0.2], [1.0, 0.0, 0.0]])
    out = rps(probs, np.array([0, 0]))
    # cumulative (0.5, 0.8) vs (1, 1): ((0.5)^2 + (0.2)^2) / 2 = 0.145; perfect forecast = 0
    assert np.allclose(out, [0.145, 0.0])


def test_matchweek_blocks_split_on_gaps():
    dates = pd.to_datetime(["2024-08-16", "2024-08-17", "2024-08-19", "2024-08-24", "2024-08-27"])
    blocks = assign_matchweeks(pd.DataFrame({"date": dates}))
    assert blocks.tolist() == [1, 1, 1, 2, 2]   # Fri-Mon is one block; Sat starts the next


class _RecordingModel(DixonColesModel):
    calls: list = []

    def fit(self, matches, ref_date=None, extra_weights=None):
        _RecordingModel.calls.append((matches["date"].max(), pd.Timestamp(ref_date)))
        return super().fit(matches, ref_date=ref_date, extra_weights=extra_weights)


def test_walk_forward_never_trains_on_the_block_it_predicts():
    train = synthetic_matches(n_rounds=4, seed=1)
    test = synthetic_matches(n_rounds=2, seed=2)
    test["date"] = test["date"] + pd.Timedelta(days=60)
    test["season"] = 2022
    test["match_id"] += 10_000
    _RecordingModel.calls = []
    preds = walk_forward(pd.concat([train, test]), 2022, model_factory=_RecordingModel, verbose=False)
    assert len(preds) == len(test)
    for last_train_date, ref_date in _RecordingModel.calls:
        assert last_train_date < ref_date
    assert (preds["date"].dt.normalize() >= preds["ref_date"]).all()
