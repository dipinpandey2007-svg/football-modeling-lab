"""
Data ingestion package.

    config         paths, season constants, request limits
    storage        JSON/CSV persistence, retry helper, loaders for processed tables
    understat_api  Understat downloads (results, shots, team histories) + table builders
    football_data  football-data.co.uk downloads, team-name mapping, HT-score join
    halves         half-time boundary inference, shot table, per-half team stats

`src.data_fetcher` orchestrates these and re-exports the public names, so
existing imports (`from src.data_fetcher import load_matches`, ...) keep working.
A new data source should get its own module here with an async `fetch_*`
function that caches raw responses under RAW_DIR and a pure `*_to_frame`
function that turns them into a table.
"""
