# Research record

One entry per experiment. Numbers are copied from the experiment output (and
[docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md), section given as §). "±" is a
standard error; model comparisons are paired by match.

**Status key.** *Confirmed* — improvement on held-out data under a rule fixed beforehand.
*Failed* — a frozen or blind test that did not pass. *Exploratory* — training-data
observation, not a test. *Superseded* — intermediate step whose evidence is not independent
or whose conclusion was replaced by a later test. *Post-hoc* observations are labelled as such
and never change a verdict.

Common setup: walk-forward Dixon-Coles refitted before every matchweek (training data strictly
before the matchweek's first day); training seasons 2020/21–2023/24 (2019/20 seeds the first
fit); 2026/27 data limited to matches up to 2026-09-20.

---

### E1 — Dixon-Coles baseline · §7.1 · `src.experiments.baseline --xi 0.0019`
- **Question:** how accurate is a walk-forward Dixon-Coles model, fitted to goals or to xG?
- **Train / test:** all matches before each 2024/25 matchweek / 2024/25 (380 matches).
- **Metric:** 1X2 log-loss, RPS, accuracy.
- **Result (ξ = 0.0019):** goals 0.9868 / 0.2033 / 53.2%; **xG 0.9816 / 0.2003 / 54.2%**;
  uniform 1.0986 / 0.2370; historical base rate 1.0799 / 0.2348.
- **Status:** baseline established. The xG-fitted model is the better variant.

### E2 — Time-decay ξ optimisation · §7.6 · `src.experiments.xi_optimisation`
- **Hypothesis:** the default decay (ξ = 0.0019/day) is not optimal; tuning it improves forecasts.
- **Train / test:** grid search on 2020/21–2023/24 walk-forward (1,520 matches) / 2024/25.
- **Metric:** full-time Poisson log-loss (search); 1X2 log-loss, RPS (held-out).
- **Result:** optimum goals ξ = 0.0025 (3.01154; 0.0019 only +0.00044 ± 0.00097 worse),
  xG ξ = 0.0046 (2.99086; flat within 0.0003 over 0.0040–0.0055). Held-out 2024/25:
  xG log-loss 0.9816 → **0.9658 (−0.0158 ± 0.0054)**, RPS 0.2003 → 0.1951 (−0.0052 ± 0.0018);
  goals 0.9868 → 0.9812 (−0.0055 ± 0.0017).
- **Status:** **Confirmed.** Tuned values are the model defaults (`OPTIMAL_XI`).
- **Interpretation:** xG is a less noisy signal than goals, so the xG model benefits from a
  shorter memory (half-life ~151 vs ~277 days).

### E3 — Tactical style archetypes · §7.2 · `src.experiments.style_archetypes --xi 0.0019`
- **Hypothesis:** stylistic match-ups (beyond team strength) shift goal difference.
- **Method:** rolling 10-match features (xG/shot for and against, deep-completion share,
  xG per deep completion), KMeans (k = 4) on 2019/20–2023/24; interaction matrix of goal-difference
  residuals vs Dixon-Coles (S_resid); S_resid added to μ/λ.
- **Train / test:** 2019/20–2023/24 / 2024/25.
- **Result:** S_resid cells all within ~1.2 SE of zero (goals model: −0.18 … +0.13). Adding it:
  log-loss 0.9816 → 0.9858 (+0.0042 ± 0.0027) symmetric, 0.9851 home-only.
- **Status:** **Failed.** Archetypes mostly re-encode team strength.

### E4 — Half-time discovery grid · §7.3 · `src.experiments.ht_grid --xi 0.0019`
- **Question:** in which half-time states does the pre-match model miss second-half goals?
- **Data:** 2020/21–2023/24, 3,040 team-rows; residual = actual 2H goals − 2H share × pre-match rate.
- **Result:** HT score alone shows nothing (row means +0.015, +0.045, +0.018). By first-half xG:
  dominating teams +0.119 ± 0.035, dominated −0.030. Starred cells: drawing & dominating +0.173,
  leading & slightly worse −0.169, leading & dominating +0.104.
- **Status:** **Exploratory** (training data; 12 cells, mirror-symmetric).
- **Note:** with the later tuned ξ the grid shifts (e.g. drawing & dominating +0.152); the
  reported table uses ξ = 0.0019 as at the time.

### E5 — "Leader outplayed on xG" rules on 2024/25 · §7.4, §7.5, §7.7 · `src.experiments.ht_leader_rule`
- **Hypothesis:** a team leading at half-time while behind on first-half xG under-performs after
  the break (and its opponent over-performs).
- **Variants (in order):**
  - §7.4 12-cell grid applied to 2024/25: overall +0.0006 ± 0.0064 (no gain); leader cell
    −0.092 ± 0.033 on 32 matches; two other starred cells reversed. *Historical; implementation
    not retained.*
  - §7.5 linear slope in the xG deficit: β = −0.090 ± 0.110 in training (not significant);
    2024/25 gain −0.0020 ± 0.0007. *Historical; implementation not retained.*
  - §7.7 capped step rule (final, ξ = 0.0046): leader with deficit in (0, 0.5) −0.2052 ± 0.0760
    (n = 125); opponent +0.1123 ± 0.0781; deficits ≥ 0.5: +0.027 ± 0.107 (no modifier).
    2024/25: 2.3105 → 2.3038 (−0.0067 ± 0.0030) overall; −0.080 ± 0.033 on 32 triggered matches.
- **Status:** **Superseded.** 2024/25 had already been examined for this effect (same 32 matches),
  so these gains are not independent evidence. The rule was frozen and tested in E6.

### E6 — Frozen test of the leader rule on 2025/26 · §7.8–7.9 · `src.experiments.ht_frozen_test`
- **Pre-registered:** ξ = 0.0046, r = 0.5524, leader −0.205, trailer +0.112, cap 0.5 xG; only
  pre-match ratings refitted.
- **Test:** 2025/26 (380 matches; never used before).
- **Metric:** second-half Poisson log-loss vs naive (r × pre-match rate).
- **Result:** naive 2.3202 vs rule **2.3233 (+0.0030 ± 0.0044)**; 37 triggered matches +0.031 ± 0.045.
  Leaders −0.140 ± 0.134 (weaker than training), trailers −0.121 ± 0.132 (reversed sign).
- **Status:** **Failed.**
- **Post-hoc:** a leader-only rule would also not have helped (+0.0009 overall).
  Exploratory 2026/27 check (50 matches to 2026-09-20): 2.2557 vs 2.2564, 2 triggers — not evidence.

### E7 — Systematic half-time state search · §7.12 · `src.experiments.ht_anomaly_search`
- **Question:** do multi-variable half-time states (venue, HT score, HT xG, pre-match strength,
  rest, rest difference) show persistent second-half deviations?
- **Data:** 2020/21–2023/24, 3,040 team-rows; match-clustered SEs.
- **Result:** 1,188 states tested; |z| > 2: 87 vs ~54 expected by chance; |z| > 3: 10 vs ~3.2;
  Benjamini-Hochberg q < 0.10: 3; Bonferroni: 1 (xG-inconsistent, weak replication).
- **Selection rules (fixed before the blind test):** same sign in 2020–21 and 2022–23 and ≥ 3/4
  seasons; goals and xG effects agree; not built only from dimensions already examined on
  2024/25; < 50% match overlap. Frozen candidates (`results/frozen/`):
  1. underdog, dominating on HT xG, normal rest — +0.375 ± 0.112 (55 matches)
  2. underdog, level at HT, normal rest, similar rest to opponent — +0.203 ± 0.077 (150 matches)
- **Status:** **Exploratory**; candidates handed to E8.

### E8 — Blind test of the search candidates · §7.13 · `src.experiments.ht_anomaly_blind_test`
- **Pass rule (printed before results):** same sign as training AND second-half log-loss gain
  over naive > 2 SE on triggered rows.
- **Test:** 2024/25, 2025/26, 2026/27 to 2026-09-20 (810 matches).
- **Result:** candidate 1: +0.005 ± 0.108 (27 matches), gain −0.077 ± 0.051 → FAIL;
  candidate 2: +0.024 ± 0.077 (71 matches), gain −0.014 ± 0.022 → FAIL.
- **Status:** **Failed (0 of 2).**
- **Post-hoc:** relative to the test period's overall −0.091 residual both states ran ~+0.10
  above average — not significant, does not change the verdict.

### E9 — Model vs closing odds and value betting · §7.10 · `src.experiments.closing_odds`
- **Question:** is the model more accurate than bookmakers, and do its "value" bets profit?
- **Data:** 2020/21–2025/26 (2,280 matches); ξ tuned on 2020/21–2023/24, so only 2024/25–2025/26
  are fully held out.
- **Result:** log-loss model 0.977 vs Bet365 closing 0.963, market average 0.962, Pinnacle 0.956
  (model 0.971 on Pinnacle rows); held out 0.998 vs 0.990 / 0.989 / 0.974. Bets at Bet365 closing
  when EV > 0: 1,999 bets, 31.5% won, ROI −3.6% ± 3.9%; quarter-Kelly −95% bankroll.
- **Status:** **No edge.** Claimed model EV (+19.8%) reflects model error.

### E10 — Market blend and line movement · §7.11 · `src.experiments.market_blend`
- **Hypothesis:** the model carries information not in the opening line.
- **Train / test:** conditional-logit blend fitted on 2020/21–2023/24 (1,520), frozen, scored on
  2024/25–2026/27 to 2026-09-20 (810).
- **Result:** model weight +0.085 ± 0.163 (z 0.5); held-out log-loss blend 0.9994 vs opening
  0.9982 vs closing 0.9943 (blend − opening +0.0012 ± 0.0014). Line movement: model − open
  disagreement predicts open → close move, slope +0.059 ± 0.014 (z 4.1, R² 0.02); closing-line
  value of model picks −3.76% ± 0.33% (all opening prices −4.70%). Market-average odds: weight
  +0.024 ± 0.166, CLV no better than random.
- **Status:** **No significant incremental information.** The line-movement association is small
  and may partly reflect noise in early-week odds snapshots; it is not exploitable after margin.

---

## Summary

| | Experiment | Status |
|---|---|---|
| E1 | Baseline | Established |
| E2 | Time-decay ξ | **Confirmed** |
| E3 | Style archetypes | Failed |
| E4 | Half-time grid | Exploratory |
| E5 | Leader rules on 2024/25 | Superseded (2 of 3 variants historical, code not retained) |
| E6 | Frozen leader rule, 2025/26 | **Failed** |
| E7 | Half-time state search | Exploratory |
| E8 | Blind test of search candidates | **Failed (0/2)** |
| E9 | Closing odds / value bets | No edge |
| E10 | Market blend | No significant information |
