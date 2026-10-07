"""
End-to-end check that each experiment still produces the numbers reported in
docs/PROJECT_CONTEXT.md and RESULTS.md.

Runs every experiment as a subprocess (as documented in the README) and checks
headline values in its output. Needs the processed data; takes a few minutes
on a cold cache. Skip with:  python -m pytest -m "not reported"
"""

import json
import subprocess
import sys

import pytest

from src.ingest.config import PROCESSED_DIR, RESULTS_DIR

pytestmark = [
    pytest.mark.reported,
    pytest.mark.skipif(not (PROCESSED_DIR / "matches.csv").exists(),
                       reason="processed data not built; run python -m src.data_fetcher"),
]


def run(module: str, *args: str) -> str:
    proc = subprocess.run([sys.executable, "-m", f"src.experiments.{module}", *args],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout


def line(output: str, startswith: str) -> list[str]:
    """Whitespace-split tokens of the first output line starting with `startswith`."""
    for ln in output.splitlines():
        if ln.strip().startswith(startswith):
            return ln.split()
    raise AssertionError(f"no line starting with {startswith!r}")


def test_e1_baseline_tuned_xi():
    out = run("baseline")
    assert line(out, "dixon_coles_goals")[1:4] == ["0.9812", "0.2016", "0.5342"]
    assert line(out, "dixon_coles_xg")[1:4] == ["0.9658", "0.1951", "0.5447"]
    assert line(out, "uniform")[1:3] == ["1.0986", "0.2370"]


def test_e1_baseline_original_xi():
    out = run("baseline", "--xi", "0.0019")
    assert line(out, "dixon_coles_goals")[1:4] == ["0.9868", "0.2033", "0.5316"]
    assert line(out, "dixon_coles_xg")[1:4] == ["0.9816", "0.2003", "0.5421"]


def test_e2_xi_optimum_values():
    assert line(run("xi_optimisation", "--target", "xg", "--start", "0.0046", "--stop", "0.0046"),
                "0.00460")[1] == "2.99086"
    assert line(run("xi_optimisation", "--target", "goals", "--start", "0.0025", "--stop", "0.0025"),
                "0.00250")[1] == "3.01154"


def test_e3_style_archetypes_integration_test():
    out = run("style_archetypes", "--compare", "--xi", "0.0019")
    assert line(out, "style_symmetric")[1:7] == ["0.9858", "0.2018", "0.5289", "0.0042", "0.0027", "0.0015"]
    assert line(out, "style_home_only")[1:6] == ["0.9851", "0.2017", "0.5289", "0.0036", "0.0027"]


def test_e3_goals_residual_matrix():
    out = run("style_archetypes", "--target", "goals", "--xi", "0.0019")
    s_resid = out[out.index("S_resid"):]
    assert line(s_resid, "0 ")[2:9:3] == ["-0.06", "+0.12", "+0.02"]
    assert line(s_resid, "1 ")[5:9:3] == ["+0.11", "-0.18"]


def test_e4_half_time_grid_as_reported():
    out = run("ht_grid", "--xi", "0.0019")
    soft = out[out.index("soft (probability-weighted)"):]
    assert "+0.173*" in line(soft, "Drawing (0)")
    assert "-0.169*" in line(soft, "Leading (>=+1)") and "+0.104*" in line(soft, "Leading (>=+1)")


def test_e5_capped_leader_rule():
    out = run("ht_leader_rule")
    assert line(out, "delta_leader")[4:8] == ["-0.2052", "(se", "0.0760,", "n=125)"]
    assert line(out, "Capped leader model")[4:6] == ["2.3038", "-0.0067"]


def test_e6_frozen_test_2025_fails():
    out = run("ht_frozen_test")
    assert line(out, "Naive baseline")[2:4] == ["380.0000", "2.3202"]
    assert line(out, "Capped leader + trailer")[5:8] == ["2.3233", "0.0030", "0.0044"]


def test_e6_exploratory_2026_check_uses_cutoff():
    out = run("ht_frozen_test", "--season", "2026")
    assert "all 50 matches" in out
    assert line(out, "Naive baseline")[2:4] == ["50.0000", "2.2557"]


def test_e7_anomaly_search_reselects_the_frozen_candidates():
    out = run("ht_anomaly_search")
    assert line(out, "Tested")[1] == "1188"
    generated = json.loads((RESULTS_DIR / "generated" / "ht_anomaly_candidates.json").read_text())
    frozen = json.loads((RESULTS_DIR / "frozen" / "ht_anomaly_candidates.json").read_text())
    assert generated["candidates"] == frozen["candidates"]


def test_e8_blind_test_fails_both():
    out = run("ht_anomaly_blind_test")
    assert "0 of 2 structural half-time anomalies survived the blind test." in out
    tests = [ln for ln in out.splitlines() if ln.strip().startswith("test:")]
    assert [t.split()[1] for t in tests] == ["+0.005", "+0.024"]


def test_e9_market_more_accurate_and_value_bets_lose():
    out = run("closing_odds")
    assert line(out, "all 2020-2025")[2:7] == ["2280.000", "0.977", "0.963", "0.962", "0.956"]
    assert line(out, "Bet365 closing")[4:6] == ["1999.000", "31.516"]
    assert line(out, "Bet365 closing")[9] == "-3.595"


def test_e10_market_blend_verdict():
    out = run("market_blend")
    assert "Model weight in blend: +0.085 +/- 0.163 (z = +0.5) -> NOT significant" in out
    assert "blend does NOT beat the market" in out
    assert "Line-movement slope on held-out data: +0.059 +/- 0.014 (z = +4.1)" in out
