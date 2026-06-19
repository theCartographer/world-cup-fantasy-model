from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_INPUT_DIR = Path("data/processed/balldontlie")
DEFAULT_OUTPUT_CSV = Path("data/processed/balldontlie/player_live_features.csv")


def clean_column_name(value: object) -> str:
    text = str(value).strip().lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [clean_column_name(column) for column in out.columns]
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build optional BALLDONTLIE player-level enrichment features."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output CSV.")
    return parser


def warn(message: str) -> None:
    print(f"Warning: {message}")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        warn(f"missing {path}")
        return pd.DataFrame()
    try:
        return normalize_columns(pd.read_csv(path, encoding="utf-8-sig"))
    except Exception as exc:  # pragma: no cover - surfaced to the CLI
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def first_present(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def first_non_empty(series: pd.Series) -> object:
    values = series.astype("string").str.strip()
    values = values.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    values = values.dropna()
    if values.empty:
        return pd.NA
    return values.iloc[0]


def to_bool_series(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    return values.isin({"true", "1", "yes", "y", "t"})


def to_player_id(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    player_id_col = first_present(out, ["player_id", "id"])
    if player_id_col is None:
        return pd.DataFrame()
    out["balldontlie_player_id"] = pd.to_numeric(out[player_id_col], errors="coerce").astype("Int64")
    return out[out["balldontlie_player_id"].notna()].copy()


def normalize_position(value: object) -> str:
    if pd.isna(value):
        text = ""
    else:
        text = str(value).strip().upper()
    mapping = {
        "G": "GK",
        "GK": "GK",
        "D": "DEF",
        "DEF": "DEF",
        "M": "MID",
        "MID": "MID",
        "F": "FWD",
        "FWD": "FWD",
    }
    return mapping.get(text, text)


def summarize_base_source(frame: pd.DataFrame, name_candidates: list[str], team_candidates: list[str], position_candidates: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["balldontlie_player_id", "player_name", "team_name", "position"])

    work = to_player_id(frame)
    if work.empty:
        return pd.DataFrame(columns=["balldontlie_player_id", "player_name", "team_name", "position"])

    name_col = first_present(work, name_candidates)
    team_col = first_present(work, team_candidates)
    position_col = first_present(work, position_candidates)

    aggregations: dict[str, tuple[str, object]] = {}
    if name_col:
        aggregations[name_col] = (name_col, first_non_empty)
    if team_col:
        aggregations[team_col] = (team_col, first_non_empty)
    if position_col:
        aggregations[position_col] = (position_col, first_non_empty)

    grouped = (
        work.groupby("balldontlie_player_id", as_index=False).agg(**aggregations)
        if aggregations
        else work[["balldontlie_player_id"]].drop_duplicates()
    )

    out = pd.DataFrame({"balldontlie_player_id": grouped["balldontlie_player_id"]})
    out["player_name"] = grouped[name_col] if name_col and name_col in grouped.columns else pd.NA
    out["team_name"] = grouped[team_col] if team_col and team_col in grouped.columns else pd.NA
    out["position"] = grouped[position_col].map(normalize_position) if position_col and position_col in grouped.columns else pd.NA
    return out


def aggregate_base_players(players: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    pieces = [
        summarize_base_source(players, ["name", "short_name"], ["country_name", "country_code"], ["position"]),
        summarize_base_source(rosters, ["player_name", "player_short_name"], ["player_country_name", "player_country_code"], ["player_position", "position"]),
    ]
    pieces = [piece for piece in pieces if not piece.empty]
    if not pieces:
        return pd.DataFrame(columns=["balldontlie_player_id", "player_name", "team_name", "position"])

    base = pd.concat(pieces, ignore_index=True, sort=False)
    out = (
        base.groupby("balldontlie_player_id", as_index=False)
        .agg(
            {
                "player_name": first_non_empty,
                "team_name": first_non_empty,
                "position": first_non_empty,
            }
        )
    )
    out["position"] = out["position"].map(normalize_position)
    return out


def aggregate_rosters(rosters: pd.DataFrame) -> pd.DataFrame:
    if rosters.empty:
        return pd.DataFrame(columns=["balldontlie_player_id"])

    work = to_player_id(rosters)
    if work.empty:
        return pd.DataFrame(columns=["balldontlie_player_id"])

    numeric_columns = ["starts", "minutes_played", "goals", "assists", "yellow_cards", "red_cards"]
    for column in numeric_columns:
        if column not in work.columns:
            warn(f"rosters.csv is missing '{column}'; that feature will fall back to other sources if available.")

    aggregations: dict[str, tuple[str, str]] = {}
    for column in numeric_columns:
        if column in work.columns:
            aggregations[f"{column}_rosters"] = (column, "sum")
    if not aggregations:
        return work[["balldontlie_player_id"]].drop_duplicates()
    return work.groupby("balldontlie_player_id", as_index=False).agg(**aggregations)


def aggregate_lineups(lineups: pd.DataFrame) -> pd.DataFrame:
    if lineups.empty:
        return pd.DataFrame(columns=["balldontlie_player_id"])

    work = to_player_id(lineups)
    if work.empty:
        return pd.DataFrame(columns=["balldontlie_player_id"])

    if "match_id" not in work.columns:
        warn("match_lineups.csv is missing 'match_id'; lineup_rows may be understated.")
    if "is_starter" not in work.columns:
        warn("match_lineups.csv is missing 'is_starter'; starts will fall back to rosters if available.")

    if "is_starter" in work.columns:
        work["is_starter"] = work["is_starter"].astype("string").str.lower().isin({"true", "1", "yes", "y"})

    aggregations: dict[str, tuple[str, str]] = {}
    aggregations["lineup_rows"] = ("match_id", "count") if "match_id" in work.columns else ("balldontlie_player_id", "size")
    if "is_starter" in work.columns:
        aggregations["starts_lineups"] = ("is_starter", "sum")

    return work.groupby("balldontlie_player_id", as_index=False).agg(**aggregations)


def aggregate_player_match_stats(stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty:
        return pd.DataFrame(columns=["balldontlie_player_id"])

    work = to_player_id(stats)
    if work.empty:
        return pd.DataFrame(columns=["balldontlie_player_id"])

    feature_columns = {
        "match_id": "matches_from_stats",
        "minutes_played": "minutes_stats",
        "goals": "goals_stats",
        "assists": "assists_stats",
        "yellow_cards": "yellow_cards_stats",
        "red_cards": "red_cards_stats",
        "shots_on_target": "shots_on_target_stats",
        "expected_goals": "xg_stats",
    }
    for column, feature in feature_columns.items():
        if column not in work.columns:
            warn(f"player_match_stats.csv is missing '{column}'; {feature} will fall back to another source if available.")

    aggregations: dict[str, tuple[str, str]] = {}
    if "match_id" in work.columns:
        aggregations["matches_from_stats"] = ("match_id", "nunique")
    if "minutes_played" in work.columns:
        aggregations["minutes_stats"] = ("minutes_played", "sum")
    if "goals" in work.columns:
        aggregations["goals_stats"] = ("goals", "sum")
    if "assists" in work.columns:
        aggregations["assists_stats"] = ("assists", "sum")
    if "yellow_cards" in work.columns:
        aggregations["yellow_cards_stats"] = ("yellow_cards", "sum")
    if "red_cards" in work.columns:
        aggregations["red_cards_stats"] = ("red_cards", "sum")
    if "shots_on_target" in work.columns:
        aggregations["shots_on_target_stats"] = ("shots_on_target", "sum")
    if "expected_goals" in work.columns:
        aggregations["xg_stats"] = ("expected_goals", "sum")

    if not aggregations:
        return work[["balldontlie_player_id"]].drop_duplicates()
    return work.groupby("balldontlie_player_id", as_index=False).agg(**aggregations)


def aggregate_match_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["balldontlie_player_id"])

    work = to_player_id(events)
    if work.empty:
        return pd.DataFrame(columns=["balldontlie_player_id"])

    if "incident_type" not in work.columns:
        warn("match_events.csv is missing 'incident_type'; event-based goals/cards will be skipped.")
        return pd.DataFrame(columns=["balldontlie_player_id"])

    work["incident_type"] = work["incident_type"].astype("string").str.lower()
    if "incident_class" in work.columns:
        work["incident_class"] = work["incident_class"].astype("string").str.lower()
    else:
        work["incident_class"] = ""

    out = pd.DataFrame({"balldontlie_player_id": sorted(work["balldontlie_player_id"].dropna().unique())})

    goal_mask = work["incident_type"].str.contains("goal", na=False)
    yellow_mask = (
        work["incident_type"].str.contains("yellow", na=False)
        | work["incident_class"].str.contains("yellow", na=False)
        | work["incident_class"].str.contains("booking", na=False)
        | work["incident_class"].str.contains("card", na=False)
    ) & ~work["incident_type"].str.contains("red", na=False)
    red_mask = (
        work["incident_type"].str.contains("red", na=False)
        | work["incident_class"].str.contains("red", na=False)
        | work["incident_class"].str.contains("card", na=False) & work["incident_type"].str.contains("second", na=False)
    )
    if not yellow_mask.any():
        warn("Could not clearly derive yellow cards from match_events.csv; leaving yellow_cards at 0.")
    if not red_mask.any():
        warn("Could not clearly derive red cards from match_events.csv; leaving red_cards at 0.")

    if goal_mask.any():
        goals = work.loc[goal_mask].groupby("balldontlie_player_id", as_index=False).size().rename(columns={"size": "goals_events"})
        out = out.merge(goals, on="balldontlie_player_id", how="left")
    else:
        out["goals_events"] = pd.NA

    if "assist_player_id" in work.columns:
        assists = work.loc[goal_mask & work["assist_player_id"].notna(), ["assist_player_id"]].copy()
        if not assists.empty:
            assists["balldontlie_player_id"] = pd.to_numeric(assists["assist_player_id"], errors="coerce").astype("Int64")
            assists = assists[assists["balldontlie_player_id"].notna()]
            assists = assists.groupby("balldontlie_player_id", as_index=False).size().rename(columns={"size": "assists_events"})
            out = out.merge(assists, on="balldontlie_player_id", how="left")
        else:
            out["assists_events"] = pd.NA
    else:
        warn("match_events.csv is missing 'assist_player_id'; assist events will be skipped.")
        out["assists_events"] = pd.NA

    if yellow_mask.any():
        yellow = work.loc[yellow_mask].groupby("balldontlie_player_id", as_index=False).size().rename(columns={"size": "yellow_cards_events"})
        out = out.merge(yellow, on="balldontlie_player_id", how="left")
    else:
        out["yellow_cards_events"] = pd.NA

    if red_mask.any():
        red = work.loc[red_mask].groupby("balldontlie_player_id", as_index=False).size().rename(columns={"size": "red_cards_events"})
        out = out.merge(red, on="balldontlie_player_id", how="left")
    else:
        out["red_cards_events"] = pd.NA

    return out


def aggregate_match_shots(shots: pd.DataFrame) -> pd.DataFrame:
    if shots.empty:
        return pd.DataFrame(columns=["balldontlie_player_id"])

    work = to_player_id(shots)
    if work.empty:
        return pd.DataFrame(columns=["balldontlie_player_id"])

    for column in ["match_id", "shot_type", "xg", "xgot"]:
        if column not in work.columns:
            warn(f"match_shots.csv is missing '{column}'; that shot feature will fall back or stay blank.")

    shots_out = work.groupby("balldontlie_player_id", as_index=False).size().rename(columns={"size": "shots_shots"})
    if "match_id" in work.columns:
        matches = work.groupby("balldontlie_player_id", as_index=False)["match_id"].nunique().rename(columns={"match_id": "matches_from_shots"})
        shots_out = shots_out.merge(matches, on="balldontlie_player_id", how="left")

    if "shot_type" in work.columns:
        on_target_mask = work["shot_type"].astype("string").str.lower().isin({"goal", "save"})
        on_target = work.loc[on_target_mask].groupby("balldontlie_player_id", as_index=False).size().rename(columns={"size": "shots_on_target_shots"})
        shots_out = shots_out.merge(on_target, on="balldontlie_player_id", how="left")
    else:
        shots_out["shots_on_target_shots"] = pd.NA

    if "xg" in work.columns:
        xg = work.groupby("balldontlie_player_id", as_index=False)["xg"].sum().rename(columns={"xg": "xg_shots"})
        shots_out = shots_out.merge(xg, on="balldontlie_player_id", how="left")
    else:
        shots_out["xg_shots"] = pd.NA

    if "xgot" in work.columns:
        xgot = work.groupby("balldontlie_player_id", as_index=False)["xgot"].sum().rename(columns={"xgot": "xgot_shots"})
        shots_out = shots_out.merge(xgot, on="balldontlie_player_id", how="left")
    else:
        shots_out["xgot_shots"] = pd.NA

    return shots_out


def aggregate_injuries(injuries: pd.DataFrame) -> pd.DataFrame:
    if injuries.empty:
        return pd.DataFrame(columns=["balldontlie_player_id"])

    work = to_player_id(injuries)
    if work.empty:
        return pd.DataFrame(columns=["balldontlie_player_id"])

    if "updated_at" in work.columns:
        work["updated_at"] = pd.to_datetime(work["updated_at"], errors="coerce")
        work = work.sort_values("updated_at", ascending=False, na_position="last")
    else:
        warn("player_injuries.csv is missing 'updated_at'; latest injury status may be arbitrary.")

    if "status" not in work.columns:
        warn("player_injuries.csv is missing 'status'; injury status will be blank.")
    if "injury_type" not in work.columns:
        warn("player_injuries.csv is missing 'injury_type'; injury description will be blank.")

    latest = work.drop_duplicates("balldontlie_player_id", keep="first")
    out = latest[["balldontlie_player_id"]].copy()
    out["injury_status"] = latest["status"] if "status" in latest.columns else pd.NA
    out["injury_description"] = latest["injury_type"] if "injury_type" in latest.columns else pd.NA
    return out


def build_match_context(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    if matches.empty:
        empty = pd.DataFrame(columns=["match_id", "match_order", "_match_datetime"])
        return empty, empty, False

    work = matches.copy()
    match_id_col = first_present(work, ["match_id", "id"])
    if match_id_col is None:
        warn("matches.csv is missing a match id column; current-minute ordering will be approximate.")
        empty = pd.DataFrame(columns=["match_id", "match_order", "_match_datetime"])
        return empty, empty, False

    work["match_id"] = pd.to_numeric(work[match_id_col], errors="coerce").astype("Int64")
    work = work[work["match_id"].notna()].copy()
    if work.empty:
        empty = pd.DataFrame(columns=["match_id", "match_order", "_match_datetime"])
        return empty, empty, False

    season_col = first_present(work, ["season_year", "season"])
    if season_col:
        season_values = pd.to_numeric(work[season_col].astype("string").str.extract(r"(\d{4})", expand=False), errors="coerce")
        if season_values.notna().any():
            work = work[season_values.eq(2026)].copy()
        else:
            warn("matches.csv has a season column, but no usable 2026 values; current-minute features will rely on matched rows only.")
    if work.empty:
        empty = pd.DataFrame(columns=["match_id", "match_order", "_match_datetime"])
        return empty, empty, False

    datetime_col = first_present(work, ["datetime", "date", "utc_date", "kickoff"])
    if datetime_col:
        work["_match_datetime"] = pd.to_datetime(work[datetime_col], errors="coerce", utc=True)
        work = work[work["_match_datetime"].notna()].copy()
        if work.empty:
            warn("matches.csv has a datetime column, but no usable current-tournament dates; current-minute features will be conservative.")
            empty = pd.DataFrame(columns=["match_id", "match_order", "_match_datetime"])
            return empty, empty, False
        work = work.sort_values(["_match_datetime", "match_id"], na_position="last")
    else:
        warn("matches.csv has no usable datetime column; falling back to match_id ordering for current-minute features.")
        work["_match_datetime"] = pd.NaT
        work = work.sort_values("match_id", ascending=True)

    work["match_order"] = range(1, len(work) + 1)
    valid_matches = work[["match_id", "match_order", "_match_datetime"]].copy()

    status_col = first_present(work, ["status", "match_status", "state"])
    if status_col:
        status_text = work[status_col].astype("string").str.lower()
        played_mask = status_text.str.contains(r"(live|complete|completed|finished|played|full time|ft)", na=False)
        played_matches = work.loc[played_mask, ["match_id", "match_order", "_match_datetime"]].copy()
        if played_matches.empty:
            warn("matches.csv status values did not identify any completed/live matches; current-minute features will use matched rows only.")
            return valid_matches, played_matches, True
    else:
        played_matches = valid_matches.copy()

    return valid_matches, played_matches, status_col is not None


def aggregate_current_minutes(player_match_stats: pd.DataFrame, lineups: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "balldontlie_player_id",
        "bd_minutes_total",
        "bd_matches_played",
        "bd_starts_total",
        "bd_minutes_last_match",
        "bd_started_last_match",
        "bd_appearance_rate",
        "bd_start_rate",
        "bd_has_current_minutes_data",
        "bd_valid_current_match_rows",
        "bd_ignored_unmatched_lineup_rows",
        "bd_minutes_source_warning",
    ]
    if player_match_stats.empty and lineups.empty:
        return pd.DataFrame(columns=columns)

    valid_matches, played_matches, has_status = build_match_context(matches)
    current_matches = played_matches if not played_matches.empty else (valid_matches if not has_status else pd.DataFrame(columns=valid_matches.columns))
    current_match_ids = set(current_matches["match_id"].dropna().astype(int).tolist()) if not current_matches.empty else set()

    stats_all = to_player_id(player_match_stats)
    lineups_all = to_player_id(lineups)

    if not stats_all.empty and "minutes_played" in stats_all.columns:
        stats_all["minutes_played"] = pd.to_numeric(stats_all["minutes_played"], errors="coerce")
    if not lineups_all.empty and "is_starter" in lineups_all.columns:
        lineups_all["is_starter"] = to_bool_series(lineups_all["is_starter"])

    player_ids = set()
    if not stats_all.empty:
        player_ids.update(stats_all["balldontlie_player_id"].dropna().astype(int).tolist())
    if not lineups_all.empty:
        player_ids.update(lineups_all["balldontlie_player_id"].dropna().astype(int).tolist())
    if not player_ids:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame({"balldontlie_player_id": sorted(player_ids)})
    out["bd_has_current_minutes_data"] = False
    out["bd_valid_current_match_rows"] = 0
    out["bd_ignored_unmatched_lineup_rows"] = 0
    out["bd_minutes_source_warning"] = ""

    # Only current-tournament rows that actually match the matches table are
    # counted. If stats are missing for a valid current lineup row, we use a
    # conservative starter fallback: starter -> 90 minutes, substitute -> 0.
    stats_current = pd.DataFrame(columns=["balldontlie_player_id", "match_id", "minutes_played"])
    lineups_current = pd.DataFrame(columns=["balldontlie_player_id", "match_id", "is_starter"])
    if current_match_ids:
        if not stats_all.empty and "match_id" in stats_all.columns:
            stats_current = stats_all[stats_all["match_id"].isin(current_match_ids)].copy()
        if not lineups_all.empty and "match_id" in lineups_all.columns:
            lineups_current = lineups_all[lineups_all["match_id"].isin(current_match_ids)].copy()

    ignored_lineups = pd.DataFrame(columns=["balldontlie_player_id"])
    if not lineups_all.empty:
        if "match_id" in lineups_all.columns:
            ignored_lineups = lineups_all[~lineups_all["match_id"].isin(current_match_ids)].copy()
        else:
            ignored_lineups = lineups_all.copy()

    if not ignored_lineups.empty:
        ignored_counts = ignored_lineups.groupby("balldontlie_player_id", as_index=False).size().rename(
            columns={"size": "bd_ignored_unmatched_lineup_rows"}
        )
        out = out.merge(ignored_counts, on="balldontlie_player_id", how="left")

    combined = pd.DataFrame(columns=["balldontlie_player_id", "match_id"])
    if not stats_current.empty or not lineups_current.empty:
        combined = lineups_current.merge(
            stats_current[["balldontlie_player_id", "match_id", "minutes_played"]],
            on=["balldontlie_player_id", "match_id"],
            how="outer",
        )
        combined = combined.merge(current_matches, on="match_id", how="left")
        if "minutes_played" in combined.columns:
            combined["minutes_played"] = pd.to_numeric(combined["minutes_played"], errors="coerce")
        if "is_starter" in combined.columns:
            combined["is_starter"] = to_bool_series(combined["is_starter"].fillna(False))
        else:
            combined["is_starter"] = False
        if "minutes_played" not in combined.columns:
            combined["minutes_played"] = pd.NA
        combined["minutes_source_warning"] = ""
        fallback_mask = combined["minutes_played"].isna() & combined["is_starter"].fillna(False)
        combined.loc[fallback_mask, "minutes_played"] = 90
        combined.loc[fallback_mask, "minutes_source_warning"] = "starter fallback used for missing stats minutes"
        substitute_mask = combined["minutes_played"].isna() & ~combined["is_starter"].fillna(False)
        combined.loc[substitute_mask, "minutes_played"] = 0
        combined.loc[substitute_mask, "minutes_source_warning"] = "substitute with no stats minutes"
        if "match_order" not in combined.columns:
            combined["match_order"] = pd.NA
        if "_match_datetime" not in combined.columns:
            combined["_match_datetime"] = pd.NaT

        valid_rows = combined.groupby("balldontlie_player_id", as_index=False).size().rename(
            columns={"size": "bd_valid_current_match_rows"}
        )
        out = out.merge(valid_rows, on="balldontlie_player_id", how="left")

        minutes_total = combined.groupby("balldontlie_player_id", as_index=False)["minutes_played"].sum().rename(
            columns={"minutes_played": "bd_minutes_total"}
        )
        out = out.merge(minutes_total, on="balldontlie_player_id", how="left")

        starts_total = combined.groupby("balldontlie_player_id", as_index=False)["is_starter"].sum().rename(
            columns={"is_starter": "bd_starts_total"}
        )
        out = out.merge(starts_total, on="balldontlie_player_id", how="left")

        matches_played = combined.dropna(subset=["match_id"]).groupby("balldontlie_player_id", as_index=False)["match_id"].nunique().rename(
            columns={"match_id": "bd_matches_played"}
        )
        out = out.merge(matches_played, on="balldontlie_player_id", how="left")

        sort_columns = [column for column in ["_match_datetime", "match_order", "match_id"] if column in combined.columns]
        if sort_columns:
            latest_matches = combined.sort_values(sort_columns, na_position="last").groupby("balldontlie_player_id", as_index=False).tail(1)
            if "minutes_played" in latest_matches.columns:
                out = out.merge(
                    latest_matches[["balldontlie_player_id", "minutes_played"]].rename(
                        columns={"minutes_played": "bd_minutes_last_match"}
                    ),
                    on="balldontlie_player_id",
                    how="left",
                )
            if "is_starter" in latest_matches.columns:
                out = out.merge(
                    latest_matches[["balldontlie_player_id", "is_starter"]].rename(
                        columns={"is_starter": "bd_started_last_match"}
                    ),
                    on="balldontlie_player_id",
                    how="left",
                )

        warning_rows = combined.loc[combined["minutes_source_warning"].astype("string").ne(""), ["balldontlie_player_id", "minutes_source_warning"]]
        if not warning_rows.empty:
            warnings = (
                warning_rows.groupby("balldontlie_player_id", as_index=False)["minutes_source_warning"]
                .agg(lambda values: "; ".join(sorted({str(value) for value in values if str(value).strip()})))
            )
            out = out.merge(warnings, on="balldontlie_player_id", how="left", suffixes=("", "_combined"))
            out["bd_minutes_source_warning"] = out.get("minutes_source_warning", pd.Series("", index=out.index)).fillna("").astype("string")
            out = out.drop(columns=["minutes_source_warning"], errors="ignore")

    for column in [
        "bd_minutes_total",
        "bd_matches_played",
        "bd_starts_total",
        "bd_minutes_last_match",
        "bd_appearance_rate",
        "bd_start_rate",
        "bd_has_current_minutes_data",
        "bd_valid_current_match_rows",
        "bd_ignored_unmatched_lineup_rows",
    ]:
        if column not in out.columns:
            out[column] = pd.NA

    out["bd_matches_played"] = pd.to_numeric(out["bd_matches_played"], errors="coerce").fillna(0)
    out["bd_minutes_total"] = pd.to_numeric(out["bd_minutes_total"], errors="coerce").fillna(0)
    out["bd_starts_total"] = pd.to_numeric(out["bd_starts_total"], errors="coerce").fillna(0)
    out["bd_minutes_last_match"] = pd.to_numeric(out["bd_minutes_last_match"], errors="coerce").fillna(0)
    out["bd_valid_current_match_rows"] = pd.to_numeric(out["bd_valid_current_match_rows"], errors="coerce").fillna(0)
    out["bd_ignored_unmatched_lineup_rows"] = pd.to_numeric(out["bd_ignored_unmatched_lineup_rows"], errors="coerce").fillna(0)
    out["bd_started_last_match"] = to_bool_series(out["bd_started_last_match"].fillna(False)) if "bd_started_last_match" in out.columns else False
    out["bd_has_current_minutes_data"] = out["bd_valid_current_match_rows"].gt(0)
    out["bd_minutes_source_warning"] = out["bd_minutes_source_warning"].fillna("").astype("string")

    ignored_warning = pd.Series("", index=out.index, dtype="string")
    ignored_mask = out["bd_ignored_unmatched_lineup_rows"].gt(0)
    ignored_warning.loc[ignored_mask] = out.loc[ignored_mask, "bd_ignored_unmatched_lineup_rows"].astype("Int64").map(
        lambda value: f"ignored {int(value)} unmatched/historical lineup rows"
    )
    missing_match_mask = out["bd_has_current_minutes_data"].eq(False)
    ignored_warning.loc[missing_match_mask & ignored_warning.eq("")] = "no valid current match rows"
    existing_warning = out["bd_minutes_source_warning"].fillna("").astype("string")
    out["bd_minutes_source_warning"] = existing_warning
    append_mask = ignored_warning.ne("")
    out.loc[append_mask & existing_warning.eq(""), "bd_minutes_source_warning"] = ignored_warning.loc[append_mask & existing_warning.eq("")]
    out.loc[append_mask & existing_warning.ne(""), "bd_minutes_source_warning"] = (
        existing_warning.loc[append_mask & existing_warning.ne("")] + "; " + ignored_warning.loc[append_mask & existing_warning.ne("")]
    )
    out["bd_minutes_source_warning"] = out["bd_minutes_source_warning"].fillna("").astype("string")

    matches_played = out["bd_matches_played"].replace(0, pd.NA)
    out["bd_appearance_rate"] = (out["bd_minutes_total"] / (matches_played * 90)).fillna(0).clip(0, 1)
    out["bd_start_rate"] = (out["bd_starts_total"] / matches_played).fillna(0).clip(0, 1)
    return out


def coalesce_series(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    for column in candidates:
        if column in frame.columns:
            result = result.combine_first(frame[column])
    return result


def build_feature_frame(input_dir: Path) -> pd.DataFrame:
    players = read_csv(input_dir / "players.csv")
    rosters = read_csv(input_dir / "rosters.csv")
    injuries = read_csv(input_dir / "player_injuries.csv")
    lineups = read_csv(input_dir / "match_lineups.csv")
    events = read_csv(input_dir / "match_events.csv")
    player_match_stats = read_csv(input_dir / "player_match_stats.csv")
    shots = read_csv(input_dir / "match_shots.csv")
    matches = read_csv(input_dir / "matches.csv")

    base = aggregate_base_players(players, rosters)
    if base.empty:
        return pd.DataFrame(
            columns=[
                "balldontlie_player_id",
                "player_name",
                "team_name",
                "position",
                "matches_in_stats",
                "lineup_rows",
                "starts",
                "minutes",
                "goals",
                "assists",
                "yellow_cards",
                "red_cards",
                "shots",
                "shots_on_target",
                "xg",
                "xgot",
                "injury_status",
                "injury_description",
                "bd_minutes_total",
                "bd_matches_played",
                "bd_starts_total",
                "bd_minutes_last_match",
                "bd_started_last_match",
                "bd_appearance_rate",
                "bd_start_rate",
                "bd_has_current_minutes_data",
                "bd_valid_current_match_rows",
                "bd_ignored_unmatched_lineup_rows",
                "bd_minutes_source_warning",
                "live_form_score",
            ]
        )

    work = base.copy()

    rosters_agg = aggregate_rosters(rosters)
    if not rosters_agg.empty:
        work = work.merge(rosters_agg, on="balldontlie_player_id", how="left")

    lineups_agg = aggregate_lineups(lineups)
    if not lineups_agg.empty:
        work = work.merge(lineups_agg, on="balldontlie_player_id", how="left")

    stats_agg = aggregate_player_match_stats(player_match_stats)
    if not stats_agg.empty:
        work = work.merge(stats_agg, on="balldontlie_player_id", how="left")

    events_agg = aggregate_match_events(events)
    if not events_agg.empty:
        work = work.merge(events_agg, on="balldontlie_player_id", how="left")

    shots_agg = aggregate_match_shots(shots)
    if not shots_agg.empty:
        work = work.merge(shots_agg, on="balldontlie_player_id", how="left")

    injuries_agg = aggregate_injuries(injuries)
    if not injuries_agg.empty:
        work = work.merge(injuries_agg, on="balldontlie_player_id", how="left")

    current_minutes_agg = aggregate_current_minutes(player_match_stats, lineups, matches)
    current_minutes_columns = [
        "bd_minutes_total",
        "bd_matches_played",
        "bd_starts_total",
        "bd_minutes_last_match",
        "bd_started_last_match",
        "bd_appearance_rate",
        "bd_start_rate",
        "bd_has_current_minutes_data",
        "bd_valid_current_match_rows",
        "bd_ignored_unmatched_lineup_rows",
        "bd_minutes_source_warning",
    ]
    if not current_minutes_agg.empty:
        work = work.merge(current_minutes_agg, on="balldontlie_player_id", how="left")
    else:
        bool_defaults = {"bd_started_last_match", "bd_has_current_minutes_data"}
        for column in current_minutes_columns:
            if column not in work.columns:
                if column == "bd_minutes_source_warning":
                    work[column] = ""
                elif column in bool_defaults:
                    work[column] = False
                else:
                    work[column] = 0

    match_sources = []
    for frame, source_name in [
        (player_match_stats, "player_match_stats.csv"),
        (lineups, "match_lineups.csv"),
        (events, "match_events.csv"),
        (shots, "match_shots.csv"),
    ]:
        if not frame.empty and {"player_id", "match_id"}.issubset(frame.columns):
            match_sources.append(
                frame[["player_id", "match_id"]].rename(columns={"player_id": "balldontlie_player_id"})
            )
        elif not frame.empty:
            warn(f"{source_name} could not contribute to matches_in_stats because player_id or match_id is missing.")

    if match_sources:
        match_counts = (
            pd.concat(match_sources, ignore_index=True)
            .dropna()
            .drop_duplicates()
            .groupby("balldontlie_player_id", as_index=False)["match_id"]
            .nunique()
            .rename(columns={"match_id": "matches_in_stats"})
        )
        work = work.merge(match_counts, on="balldontlie_player_id", how="left")
    else:
        warn("No player-level match_id columns were available; matches_in_stats will be blank.")
        work["matches_in_stats"] = pd.NA

    work["player_name"] = coalesce_series(work, ["player_name", "name", "short_name"]).astype("string")
    work["team_name"] = coalesce_series(work, ["team_name", "country_name", "country_code"]).astype("string")
    work["position"] = coalesce_series(work, ["position", "player_position", "position_rosters"]).map(normalize_position)

    work["starts"] = coalesce_series(work, ["starts_lineups", "starts_rosters"])
    work["lineup_rows"] = coalesce_series(work, ["lineup_rows"])
    work["minutes"] = coalesce_series(work, ["minutes_stats", "minutes_rosters"])
    work["goals"] = coalesce_series(work, ["goals_stats", "goals_rosters"])
    work["assists"] = coalesce_series(work, ["assists_stats", "assists_rosters"])
    work["yellow_cards"] = coalesce_series(work, ["yellow_cards_events", "yellow_cards_rosters"])
    work["red_cards"] = coalesce_series(work, ["red_cards_events", "red_cards_rosters"])
    work["shots"] = coalesce_series(work, ["shots_shots"])
    work["shots_on_target"] = coalesce_series(work, ["shots_on_target_stats"])
    work["xg"] = coalesce_series(work, ["xg_stats", "xg_shots"])
    work["xgot"] = coalesce_series(work, ["xgot_shots"])
    work["injury_status"] = coalesce_series(work, ["injury_status"])
    work["injury_description"] = coalesce_series(work, ["injury_description"])
    for column in [
        "bd_minutes_total",
        "bd_matches_played",
        "bd_starts_total",
        "bd_minutes_last_match",
        "bd_appearance_rate",
        "bd_start_rate",
        "lineup_rows",
        "bd_valid_current_match_rows",
        "bd_ignored_unmatched_lineup_rows",
    ]:
        if column not in work.columns:
            work[column] = 0
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0)

    work["bd_started_last_match"] = to_bool_series(work["bd_started_last_match"].fillna(False)) if "bd_started_last_match" in work.columns else False
    work["bd_has_current_minutes_data"] = to_bool_series(work["bd_has_current_minutes_data"].fillna(False)) if "bd_has_current_minutes_data" in work.columns else False
    if "bd_has_current_minutes_data" in work.columns:
        work["bd_has_current_minutes_data"] = work["bd_has_current_minutes_data"].fillna(False)

    numeric_columns = [
        "matches_in_stats",
        "lineup_rows",
        "starts",
        "minutes",
        "goals",
        "assists",
        "yellow_cards",
        "red_cards",
        "shots",
        "shots_on_target",
        "xg",
        "xgot",
    ]
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0)

    if work["yellow_cards"].eq(0).all():
        warn("yellow_cards remained 0 after event parsing.")
    if work["red_cards"].eq(0).all():
        warn("red_cards remained 0 after event parsing.")

    injury_flag_text = work["injury_status"].astype("string").str.strip().str.lower()
    injury_flag = injury_flag_text.notna() & ~injury_flag_text.isin({"", "nan", "none", "available", "fit", "active"})

    # First-pass live form score:
    # starts and minutes nudge the score up, attacking actions add more, and
    # cards or injury status push it down. This is intentionally simple and is
    # not meant to be the final ranking signal.
    work["live_form_score"] = (
        2.0 * work["starts"]
        + 0.02 * work["minutes"]
        + 4.0 * work["goals"]
        + 2.5 * work["assists"]
        + 0.35 * work["shots"]
        + 0.60 * work["shots_on_target"]
        + 1.25 * work["xg"]
        - 1.25 * work["yellow_cards"]
        - 2.75 * work["red_cards"]
        - 3.0 * injury_flag.astype(float)
    ).round(3)

    output_columns = [
        "balldontlie_player_id",
        "player_name",
        "team_name",
        "position",
        "matches_in_stats",
        "lineup_rows",
        "starts",
        "minutes",
        "goals",
        "assists",
        "yellow_cards",
        "red_cards",
        "shots",
        "shots_on_target",
        "xg",
        "xgot",
        "injury_status",
        "injury_description",
        "bd_minutes_total",
        "bd_matches_played",
        "bd_starts_total",
        "bd_minutes_last_match",
        "bd_started_last_match",
        "bd_appearance_rate",
        "bd_start_rate",
        "bd_has_current_minutes_data",
        "bd_valid_current_match_rows",
        "bd_ignored_unmatched_lineup_rows",
        "bd_minutes_source_warning",
        "live_form_score",
    ]
    for column in output_columns:
        if column not in work.columns:
            work[column] = pd.NA

    return work[output_columns].sort_values(["live_form_score", "minutes", "goals"], ascending=False)


def main() -> int:
    args = build_parser().parse_args()
    if args.output_csv.exists() and not args.overwrite:
        print(f"Skipping existing {args.output_csv}")
        return 0

    feature_frame = build_feature_frame(args.input_dir)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(feature_frame)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
