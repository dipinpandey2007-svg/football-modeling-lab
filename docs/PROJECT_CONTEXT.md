# EPL Research Hypothesis — Full Project Context

Complete record of a Premier League football modelling research project: goals, data,
methodology, every experiment and its exact results, what failed and why, the current
codebase, and open next steps. Written 2026-10-07 (project "today" date). All numbers
below are as produced by the code; "±" values are standard errors unless stated.

---

## 0. TL;DR

- Built a clean Python pipeline: Understat xG/shots + football-data.co.uk exact half-time
  scores and bookmaker odds, for EPL seasons 2019/20–2026/27 (2026/27 = first 50 matches).
- Built a Dixon-Coles (1997) goal model with time decay, fitted by SLSQP with analytic
  gradients, with a "goals" target and an "xG" target (quasi-Poisson on xG).
- Strict walk-forward evaluation everywhere (refit before every matchweek, train only on
  matches before the matchweek's first day).
- **What worked:** tuning the time-decay `xi` per target (goals 0.0025, xG 0.0046) improved
  2024/25 out-of-sample 1X2 log-loss from 0.9816 → **0.9658** for the xG model
  (−0.0158 ± 0.0054). Data engineering (half-time split calibration) is solid.
- **What failed (all tested out-of-sample, honestly):**
  - Tactical "style archetype" interaction matrix (KMeans clusters) → made predictions worse.
  - Half-time state adjustments (12-cell grid, continuous "Fraudulent Leader" slope, capped
    step rule) → looked good on training and on the much-examined 2024/25 season, **failed a
    frozen pristine test on 2025/26**.
  - Systematic multi-variable half-time anomaly mining (1,188 states, multiple-testing
    controls, internal replication) → top 2 frozen candidates **failed a blind test** on
    2024/25–2026/27 (effects shrank to ~0).
  - Betting vs bookmaker closing odds → model is **less accurate than the market**
    (log-loss 0.977 vs 0.956–0.963); value betting loses (Bet365 closing ROI −3.6% ± 3.9%,
    quarter-Kelly −95% bankroll).
  - Market blend test: model adds **no significant information beyond the opening line**
    (model weight +0.085 ± 0.163). Model disagreement weakly predicts open→close line moves
    (slope z ≈ 4) but closing-line value is still negative after the margin.
- **Current state:** repo under git, data layer refactored into `src/ingest/`, disk cache
  for walk-forward predictions, 23 passing tests. Recommended next milestone (not started):
  a pre-registered lower-league efficiency scan (Championship, League One/Two, National
  League, Scottish leagues) using the existing `market_blend` test.

---

## 1. Environment & tooling

- OS: Windows 11; shells: Git Bash and PowerShell.
- Default `python` on the machine is MSYS2 Python 3.14 (bad wheel support for scipy), so the
  project uses a venv built from **Python 3.13** (`py -3.13 -m venv .venv`). Run everything
  with `.venv\Scripts\python`.
- Key packages (installed versions): numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, aiohttp 3.14.4,
  beautifulsoup4 4.15.0, scikit-learn 1.9.1, understat 0.1.14, pytest 9.1.1, pyflakes.
- `understat` 0.1.14 pins aiohttp==3.8.3 (does not build on 3.13) plus pytest packages, so it
  is installed with `pip install --no-deps understat==0.1.14`. It uses Understat's JSON
  endpoints (`getLeagueData`, `getMatchData`, `getTeamData`) and works fine with aiohttp 3.14.
- Git: two commits so far — `ddb009e` "Snapshot research state before refactor",
  `194cf90` "Split data_fetcher into src/ingest, add prediction cache and tests".
  `data/` and `results/` are git-ignored except the frozen pre-registration file
  `results/ht_anomaly_candidates.json`.

---

## 2. Data sources

### 2.1 Understat (via `understat` Python library)
- League: `epl`. Season labelled by start year (2019 = 2019/20).
- **League results** (`get_league_results`): per match `id`, home/away team `title`, goals,
  xG, `datetime`, and a `forecast` (Understat's own post-match xG-based forecast — **not used**
  because it is computed after the match).
- **Shots** (`get_match_shots`, one call per match, ~380/season, concurrency 8, exponential
  backoff retries): per shot `id`, `minute`, `result` (MissedShots / BlockedShot / SavedShot /
  Goal / ShotOnPost / OwnGoal), `X`, `Y`, `xG`, `player`, `player_id`, `h_a`, `situation`,
  `shotType`, `lastAction`, `player_assisted`, team names, date.
- **Team match histories** (`get_teams`, one call per season): per team per match `deep`,
  `deep_allowed` (deep completions = passes completed within ~20 yards of goal), `ppda`
  and `ppda_allowed` (att/def counts; pressing intensity), `xpts`, etc. This is the only
  Understat source of deep completions. Joined to matches on (datetime, team, venue).
- Coverage: 2019/20–2025/26 = 380 matches each (2,660 total); 2026/27 = 50 matches
  (21 Aug – 20 Sep 2026). 58,483 shots for 2019/20–2024/25.
- Understat has **no xG for lower leagues** (covers EPL, La Liga, Bundesliga, Serie A,
  Ligue 1, RFPL only).

### 2.2 football-data.co.uk
- URL pattern `https://www.football-data.co.uk/mmz4281/{yy}{yy+1}/E0.csv` (e.g. `2425`).
  The server 302-redirects `www.` → bare domain (curl needs `-L`; aiohttp follows it).
- Used columns: `Date` (dd/mm/yyyy), `HomeTeam`, `AwayTeam`, `HTHG`, `HTAG` (exact half-time
  goals), `FTHG`, `FTAG`, and odds:
  - Opening-ish snapshot (collected ~Friday for weekend games, ~Tuesday for midweek):
    `B365H/D/A`, `PSH` (Pinnacle), `AvgH`, `MaxH`, `BWH`, `WHH`.
  - Closing: `B365CH/CD/CA`, `PSCH…`, `AvgCH…`, `MaxCH…`, `BWCH…`, `WHCH…`.
  - Asian handicap and over/under 2.5 columns exist (`B365AHH`, `B365C>2.5`, etc.) — unused.
  - `PSC*` missing for 170 matches in 2025/26 and absent in 2026/27; `WH` absent from 2025/26.
- Team-name mapping football-data → Understat: `Man City`→`Manchester City`,
  `Man United`→`Manchester United`, `Newcastle`→`Newcastle United`,
  `Nott'm Forest`→`Nottingham Forest`, `West Brom`→`West Bromwich Albion`,
  `Wolves`→`Wolverhampton Wanderers` (+ spare aliases for Sheffield Weds, Spurs, QPR, etc.).
  The mapper raises on any unmapped name.
- Join on (season, home team, away team): all 2,280 → 2,660 → 2,710 fixtures joined one-to-one
  and **every full-time score agreed** between sources. For the in-progress season, fixtures
  present in only one source are skipped with a warning (none so far).
- Understat datetimes are ~1 hour earlier than football-data times (e.g. 19:00 vs 20:00),
  consistent with UTC vs UK local time; dates checked within ±1 day.
- Lower leagues available (checked for 2024/25): E1 Championship (552 rows, Bet365 closing
  margin 6.3%), E2 League One (7.2%), E3 League Two (7.3%), EC National League (8.3%, no shot
  columns), SC0 Scottish Premiership (228 rows, 6.4%), SC1 Scottish Championship (180, 9.1%).
  EPL E0 margin 5.6%. All have HT scores, opening + closing odds incl. Pinnacle closing, and
  (except EC) shots/shots-on-target (`HS`, `AS`, `HST`, `AST`).

---

## 3. Processed data files (`data/processed/`)

- `matches.csv`: `match_id, date, days_ago, season, home_team, away_team, home_xg, away_xg,
  home_goals, away_goals`. `days_ago` is relative to the latest match (informational only).
- `shots.csv`: `shot_id, match_id, season, date, minute, half, half_source, half_uncertain,
  p_first_half, h_a, team, player, player_id, xg, result, situation, shot_type, last_action,
  x, y` (rows in chronological shot-id order within match).
- `team_matches.csv`: per team per match `deep, deep_allowed, ppda_att, ppda_def,
  ppda_allowed_att, ppda_allowed_def, xpts` (4,560 rows = 2 × 2,280 for 2019–2024).
- `ht_scores.csv`: `match_id, fd_date, ht_home_goals, ht_away_goals` (football-data).
- `half_time_stats.csv`: per team per match `match_id, team, h_a, half_1_xg, half_1_shots,
  half_1_goals, half_2_xg, half_2_shots, half_2_goals, half_1_xg_soft, half_2_xg_soft,
  split_exact, uncertain_xg, expected_error_xg`.

Raw caches in `data/raw/`: `results_{season}.json`, `shots_{season}.json`,
`teams_{season}.json`, `fd_{season}.csv`. Finished seasons are cached; the current season
(`CURRENT_SEASON = 2026`) is always re-downloaded.

---

## 4. Data-quality findings and the half-time split problem (important)

Understat shots have **no period (half) field**, and minutes count from 0:
- First-half stoppage time is logged as minutes 45, 46, 47, …; the second half also starts at
  minute 45. Shot counts by minute (2019–2024): 40:605, 41:600, 42:638, 43:575, 44:550,
  **45:921, 46:1045, 47:902, 48:865**, 49:747, 50:741, 51:698 … minute 0:275, 1:455.
- So "minute ≤ 45 = first half" mislabels first-half stoppage shots as second half.

Steps taken:
1. **Shot ordering:** Understat stores home and away shots in separate arrays. Shot `id` is
   chronological across both teams — merging by id gives at most one minute drop per match.
2. **Minute-drop rule:** a drop (e.g. 48 → 46) marks the half-time boundary. Valid only if
   the previous minute is in [45, 60] and the new minute ≥ 45; switches at most once (so
   2nd-half stoppage/extra-time drops are ignored). Found in only **325/2,280 matches (14%)**.
   One data glitch: Brentford v Wolves 2021/22 (match 16597) logs shots at 63' and 70'
   before a 46' shot (ignored by the window rule).
3. **Own goals:** Understat logs an own goal as a shot row (`result = "OwnGoal"`, xG 0)
   under the *conceding* team (235 rows). Crediting them to the opponent reproduces final
   scores in **100%** of matches vs 90.4% if counted as listed. Own goals are excluded from
   shot counts and xG.
4. **Shot xG vs match xG:** summed shot xG differs from Understat's match-level xG by >0.05 in
   497/4,560 team-matches (162 by >0.2; max ≈0.9), usually higher. Unexplained (possibly a
   per-possession adjustment); half-level xG therefore sums to shot xG, not match xG.
5. **Exact HT scores from football-data:** the shot-based HT score was wrong in **90/2,280
   matches (3.9%)**, with 91 goals in the wrong half. With exact HT scores, the feasible
   boundary positions form an interval (constraints: goals before boundary = HT score;
   first-half shots have minute ≤ 60; second-half shots minute ≥ 45). A detected drop inside
   the interval is taken as exact. Result: sources per match `ht_window` 1,813, `drop` 325,
   `ht_score` 142; **zero conflicts** with the HT scores; every drop was consistent.
6. **Per-minute probability of being 1st half** (kickoff analogy: 2nd-half minute 45+j behaves
   like 1st-half minute j, scaled by the 2H/1H shot-rate ratio ≈ 1.32 measured at minutes
   50–54 vs 5–9): P(1H | minute) = 45: 0.61, 46: 0.43, 47: 0.26, 48: 0.25, 49: 0.05, 50: 0.08,
   51: 0.01, 52: 0.01, 53: 0.06, 54–56: 0.01; ≤44 → 1; ≥57 → 0.
   Estimated vs labelled 1st-half shots: min 45 est 559 / labelled 744; 46: 446 / 181;
   47: 238 / 147; 48: 219 / 114; 49: 39 / 87; 50: 57 / 45.
7. **MAP boundary** (most likely boundary under these probabilities) changed **0 labels** — the
   probability crosses 0.5 exactly between minutes 45 and 46, same as the minute-45 rule.
   (An earlier claim that MAP would fix most error was wrong.)
8. **Soft split** added: `half_*_xg_soft` weights each shot's xG by its posterior P(1st half);
   unbiased in aggregate. Remaining error: `uncertain_xg` upper bound 653.9 xG total
   (0.287/match); **expected misassigned xG 61.0 total (0.027/match)**, ~713 shots in the
   wrong half in expectation.
9. **Freezing:** per-minute probabilities are computed only from 2019–2024 shot minutes
   (`MINUTE_PROB_SEASONS`) so adding seasons never changes earlier splits (verified
   byte-identical after adding 2025/26 and 2026/27).
10. 2nd-half share of goals: ~0.546–0.556 depending on window/seasons (e.g. 0.549 from
    2019/20–2023/24; per-prior-season values 0.541, 0.534, 0.539, 0.539). Using μ/2 instead
    biases second-half expectations upward (mean residual +0.083 vs +0.028 with the share).

Other data notes: 2026/27 split sources ht_window 41, drop 9; 2025/26: ht_window 284,
drop 72, ht_score 24.

---

## 5. Pre-match model: Dixon-Coles (`src/dixon_coles.py`)

Model for home team h vs away team a:
- log μ = γ + α_h + β_a (home expected goals); log λ = α_a + β_h (away).
- α = attack (higher scores more); β = defence parameter (**higher concedes more**);
  constraint Σα = 0 (SLSQP equality constraint); β absorbs the league scoring level.
- P(x, y) = τ(x, y; μ, λ, ρ) · Pois(x; μ) · Pois(y; λ), τ modifies 0-0, 1-0, 0-1, 1-1.
- Weighted log-likelihood, weight = exp(−ξ · t), t = days from match to prediction date.
  Objective normalised by Σ weights. Analytic gradient (checked vs numerical, error ~6e-8).
- Bounds: α, β ∈ [−4, 4], γ ∈ [−1, 2], ρ ∈ [−0.25, 0.25]; τ floored at 1e-10.
- `target="goals"` (classic) or `target="xg"`: Poisson part fitted to match xG (quasi-Poisson,
  valid for non-integer targets), τ/ρ still fitted to actual goals.
- **Newcomer prior** (`newcomer_prior_weight=3.0`): teams first seen in the latest training
  season are pulled by a Gaussian penalty (≈3 matches' weight) toward the mean (α, β) of the
  3 weakest teams of a fit on data before that season. Added because, without it, Ipswich
  (no prior EPL data) after one 0-goal match had attack at the lower bound and Man City v
  Ipswich got P(away win) = 0.00. Teams never seen get the weak-team prior.
  Sensitivity on 2024/25 (k = 0/1/3/10): goals LL 0.9862/0.9865/0.9868/0.9869;
  xG LL 0.9822/0.9818/0.9816/0.9814; min probability rose from 0.001 to ~0.02.
- `predict_match(home, away, home_cluster=None, away_cluster=None)` returns expected goals
  (μ, λ), a 7×7 τ-adjusted score matrix, and 1X2 probabilities summed over a 16×16 grid.
  `score_matrix(mu, lam, rho)` is a free function (hook for in-play updates). Optional
  `style_matrix` shifts μ/λ by an archetype interaction value (see §7).
- **Time decay tuned per target:** `OPTIMAL_XI = {"goals": 0.0025, "xg": 0.0046}`
  (`xi=None` picks it; originally 0.0019 for both).
- Fitted ρ ≈ 0 in recent EPL (−0.06 to +0.002 across 2024/25 refits); home advantage
  γ ≈ 0.18–0.22. A fit takes ~0.05 s.

---

## 6. Evaluation framework

- **Walk-forward** (`src/backtest.py::walk_forward`): test season split into "matchweek"
  blocks (new block when a match is >3 days after the block's first day; 2024/25 has 42
  blocks). Before each block, refit on all matches with date < block start; predict block.
  Pluggable `model_factory` and optional `cluster_fn` hook.
- Metrics: 1X2 log-loss (natural log), Ranked Probability Score (ordered H < D < A), accuracy;
  second-half models use Poisson log-loss of second-half goals per team, summed per match.
- Paired differences by match with standard errors; clustered SEs by match where both teams
  are rows.
- Discipline used: training seasons 2020/21–2023/24 (2019/20 only seeds history),
  frozen parameters before testing, pass rules written before results, separate held-out
  seasons, multiple-testing correction for searches.

---

## 7. Experiments, in chronological order, with results

### 7.1 Baseline Dixon-Coles, 2024/25 out-of-sample (ξ = 0.0019, newcomer prior k=3)
| Model | Log-loss | RPS | Accuracy |
|---|---|---|---|
| DC goals | 0.9868 | 0.2033 | 53.2% |
| DC xG | **0.9816** | **0.2003** | 54.2% |
| Uniform 1/3 | 1.0986 | 0.2370 | – |
| Historical H/D/A base rate | 1.0799 | 0.2348 | – |
(Without the newcomer prior: goals 0.9862/0.2034, xG 0.9822/0.2006.)

### 7.2 Tactical style archetypes (`src/archetypes.py`) — computed with ξ = 0.0019
- Features per team (pre-match, previous 10 matches, ratios of rolling sums, window reset
  after gaps >150 days): `xg_density = xG/shots`, `opp_xg_density = opp xG/opp shots`,
  `deep_ratio = deep/(deep + opp deep)`, `directness = xG/(deep + 1e-5)`.
- StandardScaler + KMeans(4, random_state=42, n_init=10) on 2019–2023 (3,480 of 3,800
  team-matches labelled). Clusters ordered by deep_ratio:
  | # | xG/shot | opp xG/shot | deep_ratio | directness | n | Profile / teams |
  |---|---|---|---|---|---|---|
  | 0 | 0.127 | 0.098 | 0.632 | 0.202 | 784 | Dominant & controlled: Man City, Arsenal, Chelsea, Man Utd |
  | 1 | 0.122 | 0.138 | 0.614 | 0.192 | 679 | Dominant but open: Liverpool (City some seasons) |
  | 2 | 0.097 | 0.116 | 0.421 | 0.195 | 1278 | Out-possessed, low quality: Palace, Burnley, Wolves, Norwich |
  | 3 | 0.125 | 0.111 | 0.380 | 0.307 | 739 | Direct: West Ham, Everton, Brentford, Luton |
- Raw interaction matrix S_raw (mean goal diff row vs column, 1,687 matches):
  0v1 +0.19, 0v2 +0.98, 0v3 +0.76, 1v2 +0.88, 1v3 +0.50, 2v3 −0.11 (antisymmetric) — this is
  mostly team quality.
- Residual matrix S_resid (actual GD − walk-forward DC expected GD), goals model, 1,407
  matches: 0v1 −0.06, 0v2 +0.12, 0v3 +0.02, 1v2 +0.11, 1v3 −0.18, 2v3 +0.13; all within ~1.2 SE.
- Recomputed vs xG model: 0v1 −0.095, 0v2 +0.174, 0v3 +0.039, 1v2 +0.190, 1v3 −0.106,
  2v3 +0.109.
- **Integration test on 2024/25** (shift GD by S, split μ+S/2, λ−S/2, or home-only;
  clusters computed point-in-time, verified leak-free): baseline 0.9816 / 0.2003 / 54.2%;
  symmetric 0.9858 / 0.2018 / 52.9% (+0.0042 ± 0.0027); home-only 0.9851 / 0.2017
  (+0.0036 ± 0.0027). 353 matches tagged, 281 adjusted. **Failed — worse than baseline.**

### 7.3 Half-time discovery grid (`src/ht_discovery.py`)
- Rows: one per team per match, 2020/21–2023/24 (1,520 matches, 3,040 rows); expected
  2nd-half goals = (2H goal share from prior seasons) × pre-match walk-forward rate (xG DC).
- Residual = actual 2H goals − expected. Grid: HT goal diff (Trailing ≤−1 / Drawing /
  Leading ≥+1) × HT xG diff (Dominated <−0.5 / Slightly worse [−0.5,0) / Slightly better
  [0,0.5) / Dominating ≥0.5).
- Final version (exact HT scores, soft xG split; computed with ξ = 0.0019):
  | HT goals \ HT xG | Dominated | Slightly worse | Slightly better | Dominating |
  |---|---|---|---|---|
  | Trailing | −0.043 (.034) n=522 | +0.074 (.061) 212 | +0.138 (.078) 125 | +0.049 (.114) 58 |
  | Drawing | −0.011 (.052) 256 | −0.011 (.048) 347 | +0.048 (.048) 347 | **+0.173** (.062) 256 |
  | Leading | +0.045 (.106) 58 | **−0.169** (.076) 125 | −0.091 (.056) 212 | **+0.104** (.045) 522 |
- Column margins: Dominated −0.030, Slightly worse −0.007, Slightly better +0.022,
  **Dominating +0.119 ± 0.035**. Row margins: +0.015, +0.045, +0.018 (no HT-score effect).
- Grid is mirror-symmetric (each match gives 2 rows): "Leading & Slightly worse" and
  "Trailing & Slightly better" are the same matches.
- History of corrections: the first grid used shot-based HT scores (Drawing&Dominating
  +0.195); a "robustness" filter that dropped 200 matches with goals at 45–48' made the
  effects look smaller but was itself biased (removed early-2H goals); exact HT scores
  restored the original picture.

### 7.4 Half-time grid backtest on 2024/25 (ξ = 0.0019)
- Naive 2H expectation = 54.9% × pre-match rate. All 380 matches (2H Poisson LL per match):
  naive 2.3065; grid 2.3071 (+0.0006 ± 0.0064); empirical-Bayes shrunk grid 2.3064
  (τ = 0.062).
- Buckets: "Unlucky Dominator" (Drawing & Dominating, 61 matches) realised −0.047 vs
  predicted +0.173 → grid worse (+0.019 ± 0.018). "Fraudulent Leader" (Leading & Slightly
  worse, 32 matches) realised **−0.403**; mirror "Unlucky Trailer" +0.270; grid improved
  **−0.092 ± 0.033**. "Leading & Dominating" (133) realised −0.176 vs +0.104 → worse.
- Calibration problem found: 2024/25 actual 2H goals 0.786/team vs naive 0.878.

### 7.5 Continuous "Fraudulent Leader" slope (ξ = 0.0019)
- Calibration ratio r = Σ actual 2H goals / Σ pre-match rate on training = 0.5575
  (home 0.5541, away 0.5614); raw share 0.5492.
- Linear slope through origin on leaders behind on HT xG: β_L = −0.0897 ± 0.1105 (n = 183);
  trailer slope +0.1002 ± 0.1129 (not significant — effect is non-linear: small deficits
  strongly negative, deficits ≥0.5 not).
- 2024/25 (380 matches): naive (r) 2.3095; raw-share naive 2.3065; leader 2.3075
  (−0.0020 ± 0.0007); leader+trailer 2.3073; walk-forward-updated ratio 2.3105 (r_t drifted
  0.585 → 0.533; actual ≈0.49); leader on raw share 2.3046 (−0.0049 ± 0.0012).
  47 triggered matches: naive 2.1717 → leader 2.1558 (−0.0160 ± 0.0056). Leaders scored
  0.447 vs naive 0.845 vs model 0.811 (model under-reacts); opponents 1.064 vs 0.917.

### 7.6 Time-decay ξ optimisation (`src/optimize_xi.py`)
- Walk-forward over 2020/21–2023/24 (1,520 matches), out-of-sample full-time Poisson log-loss
  −log Pois(hg; μ) − log Pois(ag; λ).
- Goals target: best **ξ = 0.0025** (3.01154); old 0.0019 +0.00044 ± 0.00097 (noise).
  Coarse grid 0.0005…0.0050: 3.02282, 3.01736, 3.01380, 3.01199, 3.01154, 3.01225, 3.01377,
  3.01604, 3.01890, 3.02227.
- xG target: optimum at the edge of the requested 0.0005–0.0050 range; extended to 0.0100;
  best **ξ = 0.0046** (2.99086; 0.0047 tied); flat within 0.0003 for 0.0040–0.0055;
  old 0.0019 ≈ +0.008 ± 0.004 worse. Half-life: goals ~277 days, xG ~151 days.
- **2024/25 held-out effect:** goals LL 0.9868 → 0.9812 (−0.0055 ± 0.0017), RPS 0.2033 →
  0.2016, acc 53.4%; **xG LL 0.9816 → 0.9658 (−0.0158 ± 0.0054), RPS 0.2003 → 0.1951
  (−0.0052 ± 0.0018), acc 54.5%.** This is the project's best confirmed improvement.
- Results computed earlier with ξ = 0.0019 (archetypes, grids, §7.4–7.5) are stale relative
  to the new default.

### 7.7 Capped step-function leader rule (ξ = 0.0046, `src/ht_backtest.py`)
- Training (2020–2023, ξ = 0.0046): r = 0.5524; leader penalty δ_L = **−0.2052 ± 0.0760**
  (n = 125, HT xG deficit in (0, 0.5)); trailer bonus δ_T = **+0.1123 ± 0.0781** (n = 125);
  leaders with deficit ≥0.5: +0.0270 ± 0.1067 (n = 58, no modifier); linear β −0.0840 ± 0.1107.
- 2024/25 (380 matches): naive 2.3105; capped leader 2.3038 (−0.0067 ± 0.0030); capped
  leader+trailer 2.3026 (−0.0080 ± 0.0033); linear 2.3086; raw-share naive 2.3091; capped on
  raw share 2.3024. 32 triggered matches: naive 2.2276 → capped 2.1476 (−0.080 ± 0.033),
  +trailer 2.1331 (−0.0945 ± 0.0361). Leaders 0.469 vs naive 0.883 vs model 0.678; opponents
  1.156 vs 0.922 vs 1.034. Actual 2H goals 0.786 vs naive 0.913.
- **Caveat:** same 32 matches as the grid cell already seen on 2024/25 → not independent
  evidence.

### 7.8 Frozen pristine test on 2025/26 (`src/ht_final_test.py`) — FAILED
- Frozen: ξ = 0.0046, r = 0.5524, δ_L = −0.205, δ_T = +0.112, cap 0.5 xG; only pre-match
  ratings refit walk-forward.
- All 380 matches: naive **2.3202** vs model **2.3233** (+0.0030 ± 0.0044).
- 37 triggered matches: naive 2.2053 vs model 2.2364 (+0.031 ± 0.045).
- Leaders: actual 0.703 vs naive 0.843 vs model 0.638 (residual −0.140 ± 0.134 — right sign,
  weaker). Trailers: 0.730 vs 0.851 vs 0.963 (residual −0.121 ± 0.132 — reversed).
- Post-hoc leader-only: 2.3211 (+0.0009) overall; +0.0088 ± 0.0425 on triggered.
- Leader effect across samples: training −0.21 (n=125), 2024/25 −0.41 (32), 2025/26 −0.14 (37).
  Trailer: +0.11, +0.23, −0.12 (noise). Mean 2H goals 2025/26: actual 0.780 vs naive 0.837.

### 7.9 Live exploratory check, 2026/27 to date (50 matches, 21 Aug–20 Sep 2026)
- Naive 2.2557 vs model 2.2564 (+0.0007 ± 0.0081). Actual 2H goals 0.780 vs naive 0.861.
- 2 triggers: Man City v Bournemouth (23 Aug; HT 0-1, Bournemouth deficit 0.17, FT 2-1;
  per-match LL 2.241 → 1.975, model better); Brighton v Arsenal (19 Sep; HT 2-0, Brighton
  deficit 0.14, FT 3-0; 2.065 → 2.367, model worse). Promoted: Coventry, Hull, Ipswich.

### 7.10 Value betting vs closing odds (`src/betting.py`), seasons 2020/21–2025/26
- Model: walk-forward xG DC, ξ = 0.0046 (ξ tuned on 2020–23 → those seasons in-sample for ξ;
  2024–25 held out). 2,280 matches.
- **Accuracy vs market (1X2 log-loss, margin removed by normalisation):**
  | Seasons | Model | Bet365 close | Avg close | Pinnacle close |
  |---|---|---|---|---|
  | 2020–2023 | 0.967 | 0.949 | 0.949 | 0.949 |
  | 2024–2025 | 0.998 | 0.990 | 0.989 | 0.974 (model 0.981 on same rows) |
  | All | 0.977 | 0.963 | 0.962 | 0.956 (model 0.971) |
- Bet365 average closing overround 5.49%.
- Value betting (one bet per match on max-EV outcome, EV = p·odds − 1):
  - Bet365 closing, EV > 0, all seasons: **1,999 bets, win rate 31.5%, avg odds 4.27, model
    avg EV +19.8%, profit −71.9 units, ROI −3.6% ± 3.9%, quarter-Kelly (5% cap) bankroll
    −95.3%, max drawdown 97.9%.** EV > 5%: 1,534 bets, ROI −5.9%; EV > 10%: 1,194, −5.1%.
  - Held out 2024–25, EV > 0: 670 bets, 35.2% wins, ROI +3.6% ± 6.4%, Kelly −29.7%.
  - Best price across books (MaxC, unrealistic): all seasons 2,275 bets ROI +2.1% ± 4.1%.
  - ROI by season (B365C, EV>0): 2020 +9.5%, 2021 −18.7%, 2022 −5.0%, 2023 −13.6%,
    2024 +16.4%, 2025 −8.9%. By pick: away 1,005 bets −1.0%; draw 111 +22.9% ± 20.3%;
    home 883 −9.9% ± 4.9%.
- Interpretation: the claimed ~20% EV vs −4% realised means disagreement = model error;
  bets cluster on longshots; Kelly overbets.

### 7.11 Market blend test — "Direction A" (`src/market_blend.py`)
- Conditional logit over H/D/A: score_k = a_k + b_mkt·log q_open,k + b_model·log p_model,k,
  fit on 2020–2023 (1,520 matches), frozen, scored on 2024/25, 2025/26, 2026/27 (810).
  Market-only recalibration as a control; raw opening line as the fair benchmark.
- Bet365: a_H −0.034, a_D −0.077, b_market_open 0.945 ± 0.155, **b_model +0.085 ± 0.163
  (z 0.5)**; market-only b = 1.019 ± 0.064.
- Held-out log-loss: opening 0.9982; recalibrated market 1.0004; blend 0.9994; closing 0.9943;
  model alone 1.0016. By season (open / recal / blend / close / model): 2024 0.9708 /
  0.9714 / 0.9698 / 0.9678 / 0.9658; 2025 1.0185 / 1.0218 / 1.0215 / 1.0131 / 1.0303;
  2026 1.0523 / 1.0586 / 1.0570 / 1.0531 / 1.0549. Blend − opening = +0.0012 ± 0.0014
  (does not beat market). Blend − recal −0.0010 (z −2.1) but recal is worse than raw opening.
- **Line movement:** OLS of (close − open home-vs-away log-odds) on (model − open):
  train slope +0.0213 ± 0.0109 (z 1.9, R² 0.002); **test +0.0588 ± 0.0144 (z 4.1, R² 0.020,
  same direction 52.8%)**; all +0.034 ± 0.0087. Could partly be mean reversion of noisy
  opening snapshots.
- **Closing-line value** of betting model picks at opening odds (CLV = q_close·odds_open − 1):
  EV>0 train 1,329 bets −3.85% ± 0.29%; test 693 bets −3.76% ± 0.33% (realised ROI +5.3%,
  luck); EV>10% test 397 bets −3.06% ± 0.48%; reference all opening prices −4.70%.
- Average-market odds robustness: b_model +0.024 ± 0.166; blend − open +0.0021 ± 0.0014;
  slope +0.0505 ± 0.0134 (z 3.8); CLV EV>0 test −4.21% vs reference −4.05% (no better than random).
- **Verdict:** model weight not significant; blend doesn't beat the market; tiny real
  line-movement signal not exploitable after margin. By the pre-agreed decision rule, don't
  use the model as a 1X2 betting engine in the EPL.

### 7.12 Systematic half-time anomaly mining (`src/explore_ht_anomalies.py`)
- Rows: team-match, 2020/21–2023/24 (3,040). Six dimensions: venue (home/away); ht_score
  (trail2+ 278 / trail1 639 / level 1206 / lead1 639 / lead2+ 278); ht_xg (dominated 836 /
  worse 684 / better 684 / dominating 836; soft split); pre_match (underdog / even /
  favourite by pre-match expected goal diff < −0.4 / > +0.4: 1004/1032/1004); rest days since
  previous EPL match (short ≤4: 895 / normal 5–7: 1325 / long ≥8: 814; cups/Europe not in
  data); rest_diff (fresher/similar/tired by ±2 days: 445/2138/445).
- Target: 2H goal residual vs r_goals = 0.5524 × pre-match rate; 2H xG residual
  (r_xg = 0.5650) as a check. Match-clustered SEs.
- Searched every cell of every 2–4-dimension combination with >30 matches: **1,188 states**.
  |z|>2: 87 found vs ~54 expected by chance; |z|>3: 10 vs ~3.2; Benjamini-Hochberg q<0.10: 3;
  Bonferroni p<0.05: 1 (lead2+ & better & similar rest, n=35, effect −0.403, but xG effect
  ≈0 and weak replication → finishing noise).
- Selection rules for blind candidates: same sign in 2020–21 and 2022–23 and ≥3/4 seasons;
  goals and xG effects agree (|xG z| ≥ 1); not contaminated (must use a dimension outside
  {venue, ht_score, ht_xg}, since those were examined on 2024/25); <50% match overlap;
  ranked by weaker half-sample |z|.
- Dropped earlier candidates: "home & trail1 & dominated & even" (52 matches, −0.290, xG
  +0.035 → finishing variance) and "lead1 & worse" (108, −0.228 → Fraudulent Leader family,
  contaminated).
- **Frozen candidates** (`results/ht_anomaly_candidates.json`):
  1. `ht_xg=dominating & pre_match=underdog & rest=normal` — 55 matches, +0.375 ± 0.112
     (z 3.34), xG +0.236 (z 3.36), 2020–21 +0.456 / 2022–23 +0.321, 3/4 seasons, BH q 0.141.
  2. `ht_score=level & pre_match=underdog & rest=normal & rest_diff=similar` — 150 matches,
     +0.203 ± 0.077 (z 2.65), xG +0.143 (z 3.18), +0.231 / +0.184, 4/4, BH q 0.415.

### 7.13 Blind test of anomaly candidates (`src/ht_anomaly_blind_test.py`) — both FAILED
- Pass rule (printed before results): same sign as training AND 2H log-loss gain over naive
  > 2 SE on triggered rows. Test: 2024/25, 2025/26, 2026/27 (810 matches; overall mean
  residual −0.091, i.e. naive over-predicts).
- #1: test +0.005 ± 0.108 (27 matches); LL gain −0.0768 ± 0.0505 → FAIL.
  By season residual: 2024 −0.147 (13), 2025 +0.126 (9), 2026 +0.181 (5).
- #2: test +0.024 ± 0.077 (71); gain −0.0143 ± 0.0223 → FAIL.
  By season: 2024 −0.113 (36), 2025 +0.160 (29), 2026 +0.194 (6).
- Post-hoc only: relative to the overall −0.091 level both states ran ~+0.10 above average
  (a third to half of training size; not significant).

---

## 8. Recurring lessons / known issues

1. **Training-sample effects at z≈3 shrink to ~0 out-of-sample** — every half-time idea.
2. **2024/25 was examined many times** (grid, leader rules) → not a clean test for those
   ideas; 2025/26 was the first pristine test.
3. **Baseline level bias:** pre-match xG DC over-predicts recent goal levels (2024/25 ≈3.2
   predicted vs 2.93 actual goals/match; 2H naive 0.913 vs 0.786 per team; 2025/26 0.837 vs
   0.780; 2026/27 0.861 vs 0.780). Training-calibrated ratios cannot fix a test-season shift.
   A proposed (not implemented) fix: an adaptive overall scoring-rate parameter.
4. **Realised ROI is noise**; CLV and log-loss vs market are the meaningful metrics.
5. **We never had in-play odds**, so "in-play markets are efficient" is NOT shown — only that
   half-time heuristics don't beat our own baseline.
6. Rest/congestion uses EPL fixtures only (no cup/Europe matches).
7. Shot-xG vs match-xG discrepancy unexplained.

---

## 9. Codebase (current, after refactor)

```
main.py                     Entry: 2024/25 backtest (goals + xG), --predict HOME AWAY,
                            --style (archetype comparison), --xi, --refresh
requirements.txt            numpy, pandas, scipy, scikit-learn, aiohttp, bs4, pytest, pyflakes
                            (+ understat installed with --no-deps)
pytest.ini                  pythonpath=., testpaths=tests
src/
  data_fetcher.py           Orchestrator: downloads all sources, writes processed CSVs,
                            prints half-split diagnostics; re-exports ingest API (__all__)
  ingest/config.py          Paths, DEFAULT_SEASONS 2019–2026, CURRENT_SEASON=2026,
                            MINUTE_PROB_SEASONS 2019–2024, LEAGUE="epl", FD_LEAGUE="E0"
  ingest/storage.py         Atomic JSON writes, retry helper, load_matches/shots/
                            team_matches/half_time_stats/ht_scores
  ingest/understat_api.py   Understat fetchers, results_to_frame, teams_to_frame
  ingest/football_data.py   FD fetcher (league param), aliases, map_fd_team, fd_to_frame
                            (validated join), read_fd_csv
  ingest/halves.py          drop_boundary, feasible_boundaries, first_half_minute_probs,
                            boundary_posterior, assign_halves, goal_side, shots_to_frame,
                            half_split_metrics, half_split_diagnostics
  dixon_coles.py            DixonColesModel, score_matrix, tau, OPTIMAL_XI
  backtest.py               walk_forward, log_loss, rps, baseline_scores, summarise
  cache.py                  cached_walk_forward (disk cache keyed by season, settings,
                            data hash, model/backtest source hash); EPL_NO_CACHE=1;
                            python -m src.cache [--clear]
  optimize_xi.py            ξ grid search
  archetypes.py             Style features, KMeans archetypes, S matrices, ArchetypeTagger
  ht_discovery.py           12-cell half-time grid (hard/soft xG split)
  ht_backtest.py            Half-time backtest: naive vs capped leader/trailer (+ linear)
  ht_final_test.py          Frozen test (--season 2025 default; 2026 exploratory)
  betting.py                Value betting vs closing odds, flat + quarter-Kelly
  market_blend.py           Conditional-logit market blend, line movement, CLV (--book B365|Avg)
  explore_ht_anomalies.py   Multi-dimensional half-time state search + candidate freezing
  ht_anomaly_blind_test.py  Blind test of frozen candidates
tests/
  test_halves.py            Synthetic tests of half-time inference, own goals
  test_dixon_coles.py       Gradient check, constraint, ordering recovery, ξ defaults
  test_pipeline_regression.py  380 matches/season, HT goals = football-data, HT+2H = FT,
                            soft split preserves xG, 2024/25 xG LL = 0.9658, cache = fresh
results/ht_anomaly_candidates.json   Frozen pre-registration (tracked in git)
```
- Refactor verification: rebuilt processed CSVs byte-identical; stdout of all 11 analysis
  scripts identical (cold and warm cache); 23 tests pass (~2 s); pyflakes clean.
- Cache speedup: per script 18–26 s → 1.5 s warm; full 11-script suite 52 s cold / 28 s warm.

Commands:
```
.venv\Scripts\python -m src.data_fetcher          # rebuild data (+ diagnostics)
.venv\Scripts\python main.py                      # 2024/25 backtest
.venv\Scripts\python -m src.optimize_xi --target xg --start 0.004 --stop 0.0055 --step 0.0001
.venv\Scripts\python -m src.market_blend [--book Avg]
.venv\Scripts\python -m src.ht_final_test --season 2026
.venv\Scripts\python -m pytest
```

---

## 10. Strategy discussion so far

Critique: pipeline and research discipline are strong; the pre-match model is improved but
still below market accuracy; half-time work showed no signal vs our own baseline (not vs
in-play prices); level bias unresolved; team-level features only; one league, one xG provider.

Three directions considered:
- **A. Market-anchored model / line movement** — done (§7.11): no usable edge in EPL.
- **B. Player-level / availability features** (Understat rosters: minutes, xG, xA, xGChain;
  rest/congestion; manager changes; injuries need a new source) — risk: market prices
  lineups within minutes; reconstructing pre-announcement information without leakage is hard;
  costly to build.
- **C. Totals / Asian handicap markets** (lower margins; Poisson model's natural output) —
  risk: requires fixing the goal-level bias first; sharp markets.
- Also discussed: pivot to inefficient secondary leagues vs player micro-features vs
  financial time series. Comparison: lower leagues reuse ~80% of the code and can be tested
  in days but have higher margins (6.3–9.1%), no free xG (use goals + shots-on-target proxy),
  lower stake limits; player features are weeks–months and tested in the most efficient
  league; finance reuses methodology only.

**Recommended next milestone (not started): lower-league efficiency scan.**
1. Generalise ingestion by league code (E1, E2, E3, EC, SC0, SC1).
2. Dixon-Coles target blending goals with shots-on-target as an xG proxy.
3. Tune ξ per league on 2019/20–2022/23 with `optimize_xi`.
4. Pre-registered `market_blend` test per league: fit 2019/20–2022/23, score 2023/24–2025/26.
   Proceed only if model weight > 2 SE, blend beats raw opening line by > 2 SE, and model
   picks have positive CLV vs Pinnacle close; require ~2.6 SE (Bonferroni-style) given six
   leagues. If a league passes → player-level features there + stake-limit realism. If all
   fail → team-level ratings don't beat football 1X2 within reach; pivot, carrying the
   methodology.
