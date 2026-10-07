"""Unit tests for half-time boundary inference (synthetic shot sequences)."""

import numpy as np
import pytest

from src.ingest.halves import (assign_halves, boundary_posterior, drop_boundary,
                               feasible_boundaries, goal_side)


def test_drop_boundary_detects_stoppage_time_drop():
    # 1st-half stoppage shot at 47, then 2nd half restarts at 46
    assert drop_boundary([10, 30, 44, 47, 46, 60, 88]) == 4


def test_drop_boundary_ignores_late_and_extra_time_drops():
    assert drop_boundary([10, 50, 70, 46]) is None          # from 70: data glitch, not half-time
    assert drop_boundary([10, 30, 60, 95, 91]) is None       # 2nd-half stoppage / extra time
    assert drop_boundary([5, 20, 50, 60]) is None            # no drop at all


def test_drop_boundary_switches_only_once():
    assert drop_boundary([10, 47, 46, 92, 91]) == 2


def test_feasible_boundaries_respects_ht_score_and_minutes():
    minutes = [10, 30, 46, 50, 70]
    scorers = ["h", None, None, "a", None]
    # HT 1-0: boundary after the 10' goal and before the 50' away goal, but
    # the 30' shot must be 1st half -> k in [2, 3]
    assert feasible_boundaries(minutes, scorers, (1, 0)) == (2, 3)
    # HT 1-1 needs the 50' goal in the 1st half: allowed (50 <= 60) -> k in [4, 4]
    assert feasible_boundaries(minutes, scorers, (1, 1)) == (4, 4)
    # HT 2-0 is impossible
    assert feasible_boundaries(minutes, scorers, (2, 0)) is None


def test_assign_halves_uses_drop_when_consistent_with_score():
    halves, source, uncertain, p_first = assign_halves(
        [10, 47, 46, 80], ["h", None, None, None], (1, 0), probs={m: 0.5 for m in range(45, 57)})
    assert halves == [1, 1, 2, 2]
    assert source == "drop"
    assert not any(uncertain)
    assert p_first == [1.0, 1.0, 0.0, 0.0]


def test_assign_halves_reports_conflict_with_impossible_score():
    _, source, _, _ = assign_halves([10, 80], ["h", None], (0, 1), probs=None)
    assert source == "ht_conflict"


def test_assign_halves_window_uses_probabilities():
    probs = {m: (0.9 if m == 46 else 0.01) for m in range(45, 57)}
    halves, source, uncertain, p_first = assign_halves([20, 46, 52, 80], [None] * 4, (0, 0), probs)
    assert source == "ht_window"
    assert halves == [1, 1, 2, 2]               # minute 46 most likely 1st half under these probs
    assert uncertain == [False, True, True, False]
    assert p_first[1] > 0.5 > p_first[2]


def test_boundary_posterior_sums_to_one():
    post = boundary_posterior([20, 45, 46, 47, 60], 1, 4, {m: 0.4 for m in range(45, 57)})
    assert post.shape == (4,)
    assert np.isclose(post.sum(), 1.0)


@pytest.mark.parametrize("result,h_a,expected", [
    ("Goal", "h", "h"), ("Goal", "a", "a"),
    ("OwnGoal", "h", "a"), ("OwnGoal", "a", "h"),   # own goals count for the opponent
    ("SavedShot", "h", None),
])
def test_goal_side(result, h_a, expected):
    assert goal_side({"result": result, "h_a": h_a}) == expected
