# Half-Time Signals in Premier League Football — an Out-of-Sample Study

An exploratory quantitative study of English Premier League matches (2019/20–2026/27):
a walk-forward Dixon-Coles goal model built on Understat xG, a search for half-time game
states in which the second half deviates from pre-match expectations, and a benchmark of
the model against bookmaker odds.

**Outcome in one paragraph.** Tuning the model's time-decay rate on training seasons gave
a clear held-out improvement (2024/25 1X2 log-loss 0.9816 → 0.9658). Every half-time
pattern that looked significant in training — a 12-cell game-state grid, a "leader
outplayed on xG" rule, and two candidates from a systematic 1,188-state search — failed
frozen or blind out-of-sample tests. The model is less accurate than bookmaker closing
odds and adds no statistically significant information to the opening line. The project
stops there: no robust half-time predictive signal was established, and no betting edge
is claimed.

---

## Research question

> After the first half of a Premier League match, does the half-time state (score, first-half
> xG, pre-match strength, rest) predict second-half goals beyond what a good pre-match model
> already expects?

Secondary questions: how good can a team-level xG Dixon-Coles model be made, and does it carry
information that bookmaker prices do not?

## Why it is interesting

Half-time is a natural update point: 45 minutes of evidence (shots, xG, score) arrive that a
pre-match rating cannot know about. If teams that are "leading but outplayed" or "dominating
but drawing" systematically under- or over-perform after the break, that is a measurable
behavioural or tactical effect. It is also a setting where false positives are easy to
manufacture: many game states, small samples per state, and noisy goal outcomes. The
project is as much about testing such patterns properly as about finding them.

## Data

| Source | Used for | Coverage |
|---|---|---|
| [Understat](https://understat.com) (via the `understat` library) | match results and xG, every shot (minute, xG, outcome), team match histories (deep completions, PPDA) | 2019/20–2025/26 complete (380 matches each); 2026/27 first 50 matches |
| [football-data.co.uk](https://www.football-data.co.uk) | exact half-time scores; opening and closing 1X2 odds (Bet365, Pinnacle, market average/max) | same seasons |

Sources are joined on (season, home team, away team) with a validated name map; all 2,710
fixtures joined one-to-one and every full-time score agreed between sources.

**Half-time split.** Understat shots have no period field and minutes count from 0, so
first-half stoppage-time shots (logged as 45', 46', 47' …) are indistinguishable from early
second-half shots by minute alone. The pipeline orders shots chronologically by shot id,
detects minute "drops" at half-time, constrains the boundary with the exact football-data
half-time score, and splits remaining ambiguous shots by a per-minute probability of being
first-half ("soft split"). Half-time goals are exact; the expected misassigned xG is about
0.027 per match. Details: [docs/DATA.md](docs/DATA.md).

Data is **not** included in this repository (see [Reproduction](#reproduction)); a SHA-256
manifest of the raw files used is in `data/MANIFEST.sha256`.

## Methodology

- **Model:** Dixon-Coles (1997) with exponential time decay, fitted by SLSQP with an analytic
  gradient. Two variants: fitted to goals, or to match xG (quasi-Poisson; the low-score
  correction still uses goals). A weak-team prior stabilises newly promoted sides.
- **Walk-forward evaluation:** before every matchweek the model is refitted on matches strictly
  before that matchweek's first day; nothing from the predicted block is ever used.
- **Metrics:** 1X2 log-loss and ranked probability score; for second-half models, Poisson
  log-loss of second-half goals. Differences between models are paired by match with standard
  errors (clustered by match where both teams contribute rows).
- **Discipline:** training seasons 2020/21–2023/24 (2019/20 only seeds the first fit);
  parameters frozen before testing; pass rules written before results; a separate pristine
  season (2025/26) for the frozen half-time test; multiple-testing control and internal
  replication for the systematic search; a blind test on unseen seasons; market benchmarks.

## Experimental progression

| # | Experiment | Status |
|---|---|---|
| E1 | Dixon-Coles baseline (goals vs xG target) | Baseline established |
| E2 | Time-decay ξ optimisation | **Confirmed** held-out improvement |
| E3 | Tactical style archetypes (KMeans) as an interaction adjustment | Failed |
| E4 | Half-time discovery grid (score × first-half xG) | Exploratory |
| E5 | "Leader outplayed on xG" rules (grid, linear, capped step) | Superseded — not independent evidence |
| E6 | Frozen test of the capped leader rule on 2025/26 | **Failed** |
| E7 | Systematic multi-variable half-time state search (1,188 states) | Exploratory |
| E8 | Blind test of the two frozen search candidates | **Both failed** |
| E9 | Model vs bookmaker closing odds; value-betting simulation | No edge |
| E10 | Market blend: does the model add information to the opening line? | No significant information |

Full per-experiment record (hypothesis, data, metric, result, interpretation):
**[RESULTS.md](RESULTS.md)**. Complete chronological log: [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md).

## Key results

### What survived testing — time-decay optimisation (E2)
ξ was chosen by walk-forward log-loss on 2020/21–2023/24 only (goals model 0.0025, xG model
0.0046, from 0.0019), then applied once to 2024/25:

| 2024/25 held out | Log-loss | RPS | Accuracy |
|---|---|---|---|
| xG model, ξ = 0.0019 | 0.9816 | 0.2003 | 54.2% |
| **xG model, ξ = 0.0046** | **0.9658** (−0.0158 ± 0.0054) | **0.1951** | 54.5% |
| Goals model, ξ = 0.0025 | 0.9812 (−0.0055 ± 0.0017 vs 0.0019) | 0.2016 | 53.4% |
| Uniform / historical base rate | 1.0986 / 1.0799 | 0.2370 / 0.2348 | – |

This is the strongest result. It is an improvement of the model, not a claim about the market
(see E9–E10).

### What failed — half-time hypotheses (E4–E8)
- **Discovery grid (E4, training data):** teams ≥0.5 xG ahead at half-time scored +0.12 ± 0.035
  second-half goals more than pre-match expectation; "leading but slightly outplayed" teams
  −0.17 ± 0.08. Exploratory, not a test.
- **Leader rule (E5→E6):** a capped rule (leader with a 0–0.5 xG deficit: −0.205 goals; its
  opponent +0.112) looked strong on 2024/25 — but 2024/25 had already been examined for this
  effect. Frozen and tested once on 2025/26: second-half log-loss 2.3202 (naive) vs **2.3233**
  (rule), +0.0030 ± 0.0044 — **worse**. The leader effect weakened (−0.21 training, −0.14 in
  2025/26) and the trailer effect reversed.
- **Systematic search (E7→E8):** 1,188 states; 87 with |z| > 2 vs ~54 expected by chance;
  only 3 pass Benjamini-Hochberg q < 0.10. Two candidates selected by pre-written rules
  (replication in both halves of training, goals/xG agreement, no contamination) were frozen
  and blind-tested on 2024/25–2026/27: training effects of +0.375 and +0.203 goals became
  **+0.005 ± 0.108** and **+0.024 ± 0.077**. Both failed.

The repeated pattern — effects at z ≈ 3 in four training seasons shrinking to ~0 on unseen
seasons — is the main empirical lesson of the half-time work.

### Market comparison (E9–E10)
- **Accuracy:** 1X2 log-loss over 2020/21–2025/26 — model 0.977, Bet365 closing (margin
  removed) 0.963, market average closing 0.962, Pinnacle closing 0.956 (model 0.971 on the same
  matches). The model is less accurate than the market in every period.
- **Value betting (one bet per match when model EV > 0, Bet365 closing):** 1,999 bets, 31.5%
  won, ROI −3.6% ± 3.9%; quarter-Kelly staking lost ~95% of bankroll. The model's average
  claimed edge (+19.8%) reflects model error, not value.
- **Market blend:** in a conditional-logit blend fitted on 2020/21–2023/24, the model's weight
  beyond the opening line is +0.085 ± 0.163 (not significant), and the frozen blend does not
  beat the opening line on 2024/25–2026/27. Model–market disagreement weakly predicts the
  opening-to-closing line move (slope z ≈ 4, R² = 0.02), but closing-line value of model picks
  remains negative (−3.8%), i.e. not exploitable after the bookmaker margin.

## What this project did NOT establish
- No robust half-time predictive signal.
- No evidence that betting markets — pre-match or in-play — are inefficient. In-play prices
  were never available; the half-time work only shows that these heuristics do not beat the
  project's own pre-match baseline.
- No causal claims: all half-time effects are associations between game states and outcomes.
- The pre-match model does not beat the market.

## Limitations
- One league, one xG provider (Understat); team-level features only (no lineups or injuries).
- The pre-match xG model over-predicts recent scoring levels (2024/25 ≈3.2 predicted vs 2.93
  actual goals per match); training-calibrated ratios cannot correct a test-season shift.
- Rest/congestion uses Premier League fixtures only (cup and European matches are absent).
- Half-time xG split carries residual uncertainty (~0.027 xG per match expected error);
  half-time goals are exact.
- 2024/25 was examined repeatedly for half-time effects, so only 2025/26 (and the blind test)
  provide clean half-time evidence. ξ was tuned without 2024/25.
- Summed shot xG differs from Understat's match xG by >0.05 in ~11% of team-matches
  (unexplained provider-side difference).

## Reproduction

Requirements: Python 3.11+ (developed on 3.13, Windows 11). Internet access for the first
data build.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps understat==0.1.14   # its pinned aiohttp 3.8.3 does not build on 3.13

python -m src.data_fetcher                # download + build data/processed (~2,700 shot requests)
python -m src.ingest.manifest             # compare raw data with the manifest used for the results
python -m pytest -m "not reported"        # fast unit/regression tests
python -m pytest -m reported              # re-run every experiment and check reported numbers
```

Experiments (each prints its results; outputs go to the git-ignored `results/`):

| Command | Reproduces |
|---|---|
| `python -m src.experiments.baseline [--xi 0.0019]` | E1, E2 headline (tuned ξ by default) |
| `python -m src.experiments.xi_optimisation --target xg --start 0.004 --stop 0.0055 --step 0.0001` | E2 |
| `python -m src.experiments.style_archetypes --target goals --xi 0.0019` / `--compare --xi 0.0019` | E3 |
| `python -m src.experiments.ht_grid --xi 0.0019` | E4 |
| `python -m src.experiments.ht_leader_rule` | E5 (final variant) |
| `python -m src.experiments.ht_frozen_test` / `--season 2026` | E6 (+ exploratory 2026/27 check) |
| `python -m src.experiments.ht_anomaly_search` | E7 (writes `results/generated/`, never the frozen record) |
| `python -m src.experiments.ht_anomaly_blind_test` | E8 (reads `results/frozen/`) |
| `python -m src.experiments.closing_odds` | E9 |
| `python -m src.experiments.market_blend [--book Avg]` | E10 |

Reproducibility notes:
- **Analysis cutoff 2026-09-20.** 2026/27 is in progress and is re-downloaded on every data
  build; analyses only see matches up to the cutoff (`src/ingest/config.py`,
  override with `EPL_ANALYSIS_CUTOFF`).
- **Frozen record.** `results/frozen/ht_anomaly_candidates.json` is the pre-registered blind-test
  input; it is tracked, checksum-tested, and never written by code.
- **Raw-data manifest.** Upstream providers can revise data. `python -m src.ingest.manifest`
  reports any raw file that differs from the data used here (2026/27 files are expected to differ
  once more matches are played).
- **Cache.** Walk-forward predictions are cached in `data/cache/` keyed by settings, data hash and
  model-code hash; `EPL_NO_CACHE=1` disables it, `python -m src.evaluation.cache --clear` clears it.
- **Superseded experiments.** Two intermediate half-time variants (the grid backtest and a linear
  leader slope; PROJECT_CONTEXT §7.4–7.5) were replaced by the final capped rule before version
  control and are not retained; their numbers are recorded as historical in RESULTS.md.

## Repository structure

```
README.md, RESULTS.md, LICENSE, requirements.txt, pytest.ini
docs/DATA.md                 data sources, joins, half-time split, limitations
docs/PROJECT_CONTEXT.md      full chronological research log with all numbers
data/MANIFEST.sha256         checksums of the raw data used (data itself is git-ignored)
results/frozen/              pre-registered blind-test candidates (immutable)
src/
  data_fetcher.py            builds data/processed from the sources
  ingest/                    config, storage/loaders, understat_api, football_data, halves, manifest
  model/dixon_coles.py       Dixon-Coles model
  evaluation/                walk-forward backtest, metrics, prediction cache
  experiments/               E1–E10, one module each
tests/                       unit, protection and end-to-end result tests
```

## Conclusion

The project built a careful data pipeline (including an exact half-time split that the raw data
does not provide), a walk-forward Dixon-Coles model whose time-decay tuning delivered a real
held-out improvement, and a sequence of half-time hypotheses that were formed on training data
and then rejected by frozen and blind tests. Benchmarked against bookmakers, the model is less
accurate than closing odds and adds no significant information to opening prices. The
evidence stopped supporting further claims, so the research ends here; possible extensions
that were considered but not pursued are listed at the end of
[docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md).

## License

Code: MIT (see [LICENSE](LICENSE)). Data from Understat and football-data.co.uk is not
redistributed here and remains subject to each provider's terms of use.
