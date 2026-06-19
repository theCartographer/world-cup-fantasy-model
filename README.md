# FIFA World Cup Fantasy Pipeline

Simple pandas pipeline for building transparent FIFA World Cup Fantasy shortlists.

## Inputs

- `world-cup_2026.csv`: local fixture CSV. By default the pipeline looks for it in this folder.
- `data/raw/wc2026_players.csv`: fantasy player list from `jlbgouveia/fifa-wc2026-fantasy-analytics`.
- `data/raw/wc2026_player_stats_apifootball.csv`: historical player stats from the same repo.
- `data/raw/wc2026_team_stats*.csv`: optional national-team stats from the same repo.
- Optional current FIFA Fantasy export as CSV or TXT.
- Optional team strength files from `Hicruben/world-cup-2026-prediction-model` and the linked cup26matches.com open data endpoints.

The downloader only fetches raw GitHub files and explicitly linked open-data files. It does not scrape web pages.

## Quick Start

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_rankings.py --fixture-csv world-cup_2026.csv
```

## BALLDONTLIE Optional Enrichment

Set the API key in the current PowerShell session before running optional BALLDONTLIE downloads:

```powershell
$env:BALLDONTLIE_API_KEY = "<your-api-key>"
```

Keep that key out of source control and never commit it.

Run a small test download first:

```powershell
python scripts/download_data.py --balldontlie --balldontlie-endpoints teams stadiums --balldontlie-rate-limit-seconds 13
```

Run the more useful World Cup endpoints with the same trial-safe delay:

```powershell
python scripts/download_data.py --balldontlie --balldontlie-endpoints players rosters player_injuries matches match_lineups match_events player_match_stats team_match_stats match_shots --balldontlie-rate-limit-seconds 13
```

Then flatten the downloaded JSON into processed CSV files:

```powershell
python scripts/flatten_balldontlie.py
```

With a current fantasy export. CSV exports are supported, and copied FIFA Fantasy
player-list TXT exports like `Player / Total pts / Action` blocks are parsed too:

```powershell
python scripts/build_rankings.py --fixture-csv world-cup_2026.csv --current-export current_fifa_export.csv
```

For a specific round or matchday slate, use `--target-round`. The fixture CSV
must have either a `round` or `matchday` column:

```powershell
python scripts/build_rankings.py --fixture-csv world-cup_2026.csv --target-round 2
```

To choose the markdown report path:

```powershell
python scripts/build_rankings.py --fixture-csv world-cup_2026.csv --target-round 2 --report-md outputs/matchday_2_report.md
```

`--target-round` attaches players to that exact round/matchday. Without it,
`--as-of` is used instead: matches before the cutoff are treated as played, and
the next unplayed fixture is attached for each team.

## Team Aliases

Team names are normalized before joining players to fixtures. The pipeline uses
built-in aliases plus the editable CSV at `config/team_aliases.csv`.

The alias file has two columns:

```csv
alias,canonical
Czech Republic,Czechia
Bosnia & Herzegovina,Bosnia and Herzegovina
Türkiye,Turkey
```

Use a custom alias file with:

```powershell
python scripts/build_rankings.py --fixture-csv world-cup_2026.csv --team-aliases config/team_aliases.csv
```

If a player team still cannot be matched to a fixture team after normalization,
the markdown report includes it under `Team Match Warnings`.

## Team Strength

You can provide an optional team-strength CSV:

```csv
team,strength_score,win_probability_next_match,expected_goals,clean_sheet_probability,source
Czech Republic,0.72,0.58,1.7,0.34,manual model
USA,68,55%,1.4,28%,manual model
```

Run with:

```powershell
python scripts/build_rankings.py --fixture-csv world-cup_2026.csv --target-round 2 --team-strength-csv data/team_strength.csv
```

`strength_score` may be 0-1 or 0-100. Optional `expected_goals` helps rank
midfielders and forwards; optional `clean_sheet_probability` helps rank
defenders and goalkeepers. If those optional columns are missing, the model uses
`strength_score` only.

## Outputs

The ranking script writes to `output/` by default:

- `player_rankings_all.csv`
- `player_rankings_GK.csv`
- `player_rankings_DEF.csv`
- `player_rankings_MID.csv`
- `player_rankings_FWD.csv`
- `fantasy_shortlist_report.md`

Use `--report-md` to write the markdown report somewhere else.

## Score

This is a practical office-league fantasy model, not a betting model. It is meant
to produce transparent shortlists using fantasy-relevant signals rather than to
price match outcomes.

`Fantasy Shortlist Score` is a weighted score out of 100:

- 25% starting/minutes score
- 25% fixture score
- 20% capped historical player score
- 15% team strength
- 10% value for price
- 5% current fantasy form

After that weighted score, the model applies a starting-likelihood multiplier:

- Likely Starter: `1.00`
- Maybe Starter: `0.85`
- Unknown: `0.80`
- Unlikely Starter: `0.60`

Historical production is normalized by position and capped before weighting so a
single extreme historical field does not dominate. Fixture score uses Elo
difference when both teams have Elo; otherwise it stays neutral at `0.5` with
fixture difficulty `3`. Each component is exported so the score can be inspected
and adjusted.
