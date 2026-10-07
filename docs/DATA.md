# Data

## Sources

| Source | Access | What is used |
|---|---|---|
| **Understat** — `understat.com` | `understat` Python library 0.1.14 (JSON endpoints `getLeagueData`, `getMatchData`, `getTeamData`); no credentials | league results with match xG; every shot (`id`, `minute`, `xG`, `result`, `h_a`, player, situation, coordinates); per-team match histories (deep completions, PPDA, xPts) |
| **football-data.co.uk** — `https://www.football-data.co.uk/mmz4281/{yy}{yy+1}/E0.csv` | plain CSV download (the `www.` host redirects to the bare domain) | `HTHG`/`HTAG` exact half-time goals, `FTHG`/`FTAG`, and 1X2 odds: early-week snapshot (`B365H/D/A`, `PSH…`, `AvgH…`, `MaxH…`) and closing (`B365CH/CD/CA`, `PSCH…`, `AvgCH…`, `MaxCH…`) |

- **Seasons:** labelled by start year (2019 = 2019/20). 2019/20–2025/26 complete (380 matches
  each); 2026/27 first 50 matches (2026-08-21 to 2026-09-20). 2019/20 is used only as history
  for the first walk-forward fits.
- **xG source:** Understat only. Understat's own post-match `forecast` field is not used.
- **Half-time score source:** football-data.co.uk (exact). Understat shot data alone mislabels
  the half-time score in 3.9% of matches.
- **Odds source:** football-data.co.uk. "Opening" odds are early-week snapshots (Friday for
  weekend fixtures, Tuesday for midweek), not true market openings. Pinnacle closing odds are
  missing for 170 matches in 2025/26 and absent in 2026/27.

## Joining the sources

- football-data team names are mapped to Understat names (`FD_TEAM_ALIASES` in
  `src/ingest/football_data.py`: `Man City`, `Man United`, `Newcastle`, `Nott'm Forest`,
  `West Brom`, `Wolves`, …). An unmapped name raises an error.
- Join key: (season, home team, away team), unique within a season. The join raises if any
  fixture of a completed season fails to match one-to-one, if a full-time score differs between
  sources, or if a half-time score exceeds the full-time score. In the in-progress season,
  fixtures present in only one source are skipped with a warning. Result: all 2,710 fixtures
  matched and every full-time score agreed.
- Understat team histories have no match id; they are joined on (kick-off time, team, venue).

## Half-time split methodology (`src/ingest/halves.py`)

Understat shots have no period field and minutes count from 0, so first-half stoppage time is
logged as 45', 46', 47' … while the second half also starts at 45'. Shot counts per minute
(2019–2024) show the overlap: 44': 550, **45': 921, 46': 1,045, 47': 902, 48': 865**, 50': 741.

1. **Order:** home and away shots are stored separately; shot `id` is chronological across both
   teams, so a match's shots are merged and sorted by id.
2. **Minute drop:** a fall in minute (e.g. 48' → 46') marks half-time if the previous minute is in
   [45, 60] and the new one ≥ 45; at most one switch. Present in 325 of 2,280 matches (14%).
3. **Own goals** are logged under the conceding team (result `OwnGoal`, xG 0); they are credited to
   the opponent and excluded from shots/xG. This reproduces 100% of final scores (90.4% otherwise).
4. **Exact half-time score:** the positions consistent with the football-data HT score and minute
   limits (first-half shots ≤ 60', second-half shots ≥ 45') form an interval; a minute drop inside it
   is exact. Sources per match (2019–2024): `ht_window` 1,813, `drop` 325, `ht_score` 142; zero
   conflicts with the half-time scores.
5. **Remaining ambiguity:** inside the interval, each shot gets P(first half | minute), estimated from
   the kick-off analogy (second-half minute 45 + j behaves like first-half minute j, scaled by the
   2H/1H shot-rate ratio ≈ 1.32): 45': 0.61, 46': 0.43, 47': 0.26, 48': 0.25, 49': 0.05, ≥ 51': ≈ 0.01.
   These probabilities are frozen to 2019–2024 shot minutes (`MINUTE_PROB_SEASONS`).
6. **Outputs:** `half` (most likely half; identical to a minute-45 cut-off), `p_first_half`, and per
   team `half_*_xg_soft` (xG weighted by P(first half); unbiased in aggregate). Expected misassigned
   xG ≈ 0.027 per match (upper bound 0.287 if every unproven shot were wrong). Half-time **goals**
   are always the exact football-data score.

## Files

Built by `python -m src.data_fetcher` (git-ignored):

| File | Content |
|---|---|
| `data/raw/results_{season}.json` | Understat league results |
| `data/raw/shots_{season}.json` | Understat shots per match |
| `data/raw/teams_{season}.json` | Understat team match histories |
| `data/raw/fd_{season}.csv` | football-data.co.uk results and odds |
| `data/processed/matches.csv` | one row per match: date, season, teams, xG, goals |
| `data/processed/shots.csv` | one row per shot with `half`, `half_source`, `half_uncertain`, `p_first_half` |
| `data/processed/team_matches.csv` | per team per match: deep completions, PPDA, xPts |
| `data/processed/ht_scores.csv` | exact half-time score per match |
| `data/processed/half_time_stats.csv` | per team per match: xG / shots / goals by half (hard and soft xG split) |
| `data/cache/walk_forward/*.pkl` | cached walk-forward predictions |

Tracked: `data/MANIFEST.sha256` — SHA-256 of each raw file as used for the reported results
(CSV hashed with carriage returns removed, so Windows and Linux copies agree).

## Rebuilding and checking the data

```bash
python -m src.data_fetcher        # downloads anything not cached, rebuilds data/processed
python -m src.ingest.manifest     # lists raw files that differ from the manifest
```

- Completed seasons are downloaded once and cached; the in-progress season (2026/27,
  `CURRENT_SEASON`) is re-downloaded on every build.
- **Analysis cutoff:** all analyses read data through loaders that drop matches after
  2026-09-20 (`ANALYSIS_CUTOFF` in `src/ingest/config.py`; override with
  `EPL_ANALYSIS_CUTOFF=YYYY-MM-DD` or `none`). Rebuilding later in the season therefore does not
  change the reported results, although the 2026/27 raw files will no longer match the manifest.
- A mismatch on a completed season means the provider has revised its data; results may then
  differ slightly from those reported.

## Known data limitations

- Single xG provider; no lineups, injuries, or player availability.
- Rest days count Premier League fixtures only (no cup or European matches).
- Summed shot xG differs from Understat's match xG by > 0.05 in 497 of 4,560 team-matches
  (2019–2024; usually higher); half-level xG sums to shot xG.
- Understat and football-data clock times differ by about an hour (UTC vs UK local); dates agree.
- One ordering glitch is known (Brentford v Wolves 2021/22, match 16597: 63' and 70' shots logged
  before a 46' shot); the minute-window rule ignores it.
