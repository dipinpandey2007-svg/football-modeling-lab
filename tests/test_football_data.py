"""Validation of the football-data.co.uk join (synthetic CSVs)."""

import pandas as pd
import pytest

from src.ingest.config import CURRENT_SEASON
from src.ingest.football_data import fd_to_frame, map_fd_team


def understat_matches(season: int) -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": [1, 2],
        "date": pd.to_datetime(["2021-08-14 14:00", "2021-08-15 16:30"]),
        "season": season,
        "home_team": ["Manchester City", "Wolverhampton Wanderers"],
        "away_team": ["Arsenal", "Newcastle United"],
        "home_goals": [2, 0],
        "away_goals": [1, 0],
    })


def fd_csv(rows: list[str]) -> str:
    return "Date,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG\n" + "\n".join(rows) + "\n"


GOOD = ["14/08/2021,Man City,Arsenal,2,1,1,0", "15/08/2021,Wolves,Newcastle,0,0,0,0"]


def test_join_maps_names_and_returns_ht_scores():
    out = fd_to_frame({2021: fd_csv(GOOD)}, understat_matches(2021))
    assert out.set_index("match_id")[["ht_home_goals", "ht_away_goals"]].to_dict("index") == {
        1: {"ht_home_goals": 1, "ht_away_goals": 0}, 2: {"ht_home_goals": 0, "ht_away_goals": 0}}


def test_join_raises_on_full_time_mismatch():
    bad = [GOOD[0].replace(",2,1,1,0", ",3,1,1,0"), GOOD[1]]
    with pytest.raises(ValueError, match="full-time scores differ"):
        fd_to_frame({2021: fd_csv(bad)}, understat_matches(2021))


def test_join_raises_on_unmatched_fixture_in_completed_season():
    with pytest.raises(ValueError, match="failed to join"):
        fd_to_frame({2021: fd_csv(GOOD[:1])}, understat_matches(2021))


def test_join_skips_unmatched_fixture_in_current_season():
    out = fd_to_frame({CURRENT_SEASON: fd_csv(GOOD[:1])}, understat_matches(CURRENT_SEASON))
    assert out["match_id"].tolist() == [1]


def test_unknown_team_name_raises():
    with pytest.raises(KeyError):
        map_fd_team("Sporting Nowhere", {"Arsenal"})
