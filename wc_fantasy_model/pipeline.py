"""Fantasy ranking pipeline built with pandas."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .download import download_all
from .normalize import display_team_from_key, load_team_aliases, normalize_player_name, normalize_team, strip_accents


POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]
REPORT_COLUMNS = [
    "Player",
    "Position",
    "Team",
    "Opponent",
    "Round/Matchday",
    "Price",
    "Fixture difficulty",
    "Starting likelihood",
    "Historical score",
    "Final score",
    "Risk note",
]

BDL_REPORT_COLUMNS = [
    "Player",
    "Team",
    "Position",
    "balldontlie_player_id",
    "live_form_score",
    "bd_live_form_adjustment",
    "bd_injury_penalty",
    "bd_minutes",
    "bd_starts",
    "bd_goals",
    "bd_assists",
    "bd_xg",
    "bd_xgot",
    "bd_injury_status",
    "bd_injury_description",
]

STARTING_LIKELIHOOD = {
    "likely starter": 1.0,
    "starter": 0.9,
    "maybe starter": 0.55,
    "rotation": 0.45,
    "unlikely starter": 0.2,
    "bench": 0.15,
}

STARTING_PENALTIES = {
    "Likely Starter": 1.0,
    "Maybe Starter": 0.85,
    "Unknown": 0.80,
    "Unlikely Starter": 0.60,
}

TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252")

FIFA_TEXT_DETAIL_RE = re.compile(
    r"^(?P<position>GK|DEF|MID|FWD)\s*\|\s*"
    r"(?P<home>[A-Z]{2,3})\s+v\s+(?P<away>[A-Z]{2,3})\s*\|\s*"
    r"\$(?P<price>\d+(?:\.\d+)?)m$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_column_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("%", " pct ")
    text = strip_accents(text)
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [clean_column_name(col) for col in out.columns]
    return out


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV/TXT/JSON with forgiving defaults for user exports."""
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".txt" and looks_like_fifa_text_export(path):
        return parse_fifa_text_export(path)

    if suffix == ".json":
        data = json.loads(read_text_with_fallback(path))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            for key in ("teams", "data", "probabilities", "rows"):
                if isinstance(data.get(key), list):
                    return pd.DataFrame(data[key])
            return pd.DataFrame([data])

    readers = []
    if suffix == ".txt":
        readers = [
            {"sep": "\t", "engine": "python"},
            {"sep": None, "engine": "python"},
            {"sep": ","},
        ]
    else:
        readers = [
            {"sep": ","},
            {"sep": None, "engine": "python"},
            {"sep": "\t", "engine": "python"},
        ]

    last_error: Exception | None = None
    for encoding in TEXT_ENCODINGS:
        for kwargs in readers:
            try:
                return pd.read_csv(path, encoding=encoding, on_bad_lines="warn", **kwargs)
            except Exception as exc:  # pragma: no cover - last-error relay
                last_error = exc
    raise ValueError(f"Could not read {path}: {last_error}")


def read_text_with_fallback(path: Path) -> str:
    """Read text with the same encoding policy used for CSV/TXT inputs."""
    last_error: Exception | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "text",
        b"",
        0,
        1,
        f"Could not decode {path} with {', '.join(TEXT_ENCODINGS)}: {last_error}",
    )


def looks_like_fifa_text_export(path: Path) -> bool:
    text = read_text_with_fallback(path)
    return bool(FIFA_TEXT_DETAIL_RE.search(text.replace("\r\n", "\n")))


def parse_fifa_text_export(path: Path) -> pd.DataFrame:
    """Parse FIFA Fantasy's copied player-list text layout.

    The export is not a delimited table. After the header it repeats:
    short name, display name, "POS | AAA v BBB | $X.Xm", blank, total points.
    """
    lines = [
        line.strip()
        for line in read_text_with_fallback(path).splitlines()
        if line.strip()
    ]
    lines = [line for line in lines if line.lower() not in {"player", "total pts", "action"}]

    rows: list[dict[str, Any]] = []
    index = 0
    while index + 3 < len(lines):
        short_name = lines[index]
        display_name = lines[index + 1]
        detail = lines[index + 2]
        points = lines[index + 3]
        match = FIFA_TEXT_DETAIL_RE.match(detail)
        if not match:
            index += 1
            continue

        home_abbr = match.group("home").upper()
        away_abbr = match.group("away").upper()
        rows.append(
            {
                "short_name": short_name,
                "display_name": display_name,
                "fantasy_position": match.group("position").upper(),
                "fantasy_price": float(match.group("price")),
                "total_points": pd.to_numeric(points, errors="coerce"),
                "fixture": f"{home_abbr} v {away_abbr}",
                "fixture_home_abbr": home_abbr,
                "fixture_away_abbr": away_abbr,
                "fixture_home_key": normalize_team(home_abbr),
                "fixture_away_key": normalize_team(away_abbr),
                "current_export_source": str(path),
            }
        )
        index += 4

    return pd.DataFrame(rows)


def clean_id(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return values.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def first_containing(df: pd.DataFrame, required: list[str], excluded: list[str] | None = None) -> str | None:
    excluded = excluded or []
    for column in df.columns:
        if all(part in column for part in required) and not any(part in column for part in excluded):
            return column
    return None


def to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def minmax(series: pd.Series, default: float = 0.5) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty or valid.max() == valid.min():
        return pd.Series(default, index=series.index, dtype="float64")
    return ((values - valid.min()) / (valid.max() - valid.min())).fillna(default)


def minmax_by_position(df: pd.DataFrame, column: str, default: float = 0.5) -> pd.Series:
    pieces = []
    for _, group in df.groupby("position", dropna=False):
        pieces.append(minmax(group[column], default=default))
    if not pieces:
        return pd.Series(dtype="float64")
    return pd.concat(pieces).sort_index()


def load_players(raw_dir: Path, current_export: Path | None = None) -> pd.DataFrame:
    players_path = raw_dir / "wc2026_players.csv"
    players = normalize_columns(read_table(players_path))

    name_col = first_present(players, ["display_name", "name", "player", "full_name"])
    team_col = first_present(players, ["country", "team", "nation", "squad", "country_name"])
    abbr_col = first_present(players, ["country_abbr", "team_abbr", "abbr"])
    pos_col = first_present(players, ["fantasy_position", "position", "pos"])

    if name_col is None or (team_col is None and abbr_col is None):
        raise ValueError("Player file needs a player name and country/team column.")

    players["display_name"] = players[name_col].astype("string").fillna("")
    players["team"] = players[team_col].astype("string").fillna("") if team_col else players[abbr_col].astype("string").fillna("")
    players["team_key"] = players["team"].map(normalize_team)
    if abbr_col:
        missing_team = players["team_key"].eq("")
        players.loc[missing_team, "team_key"] = players.loc[missing_team, abbr_col].map(normalize_team)
    players["player_key"] = players["display_name"].map(normalize_player_name)
    players["player_name_standardized"] = players["player_key"]
    players["team_standardized"] = players["team_key"].map(display_team_from_key)
    players["position"] = players[pos_col].astype("string").str.upper().fillna("UNK") if pos_col else "UNK"

    if "player_id" in players.columns:
        players["player_id"] = clean_id(players["player_id"])

    numeric_columns = [
        "fantasy_price",
        "percent_selected",
        "total_points",
        "avg_points",
        "form",
        "club_goals_4y",
        "club_assists_4y",
        "club_games_4y",
        "nt_goals_4y",
        "nt_assists_4y",
        "nt_games_4y",
    ]
    players = to_numeric(players, numeric_columns)

    if current_export:
        players = merge_current_export(players, current_export)

    return players


def merge_current_export(players: pd.DataFrame, current_export: Path) -> pd.DataFrame:
    current = normalize_columns(read_table(current_export))
    if current.empty:
        return players

    current = prepare_current_export(current)
    players_indexed = players.reset_index().rename(columns={"index": "_player_row_id"})
    players_indexed["player_last_key"] = players_indexed["player_key"].map(last_name_key)

    current = current.reset_index(drop=True)
    current["_current_record_id"] = range(len(current))

    match_frames: list[pd.DataFrame] = []

    if "player_id" in players_indexed.columns and "player_id" in current.columns:
        id_matches = players_indexed[["_player_row_id", "player_id"]].dropna().merge(
            current[["_current_record_id", "player_id"]].dropna(),
            on="player_id",
            how="inner",
        )
        add_unique_matches(match_frames, id_matches, 1, "player_id")

    exact_team_matches = build_name_team_matches(players_indexed, current, use_last_name=False)
    add_unique_matches(match_frames, exact_team_matches, 2, "name_and_team_or_fixture")

    exact_unique_matches = build_unique_name_matches(players_indexed, current)
    add_unique_matches(match_frames, exact_unique_matches, 3, "unique_player_name")

    last_name_team_matches = build_name_team_matches(players_indexed, current, use_last_name=True)
    add_unique_matches(match_frames, last_name_team_matches, 4, "last_name_and_fixture_team")

    if not match_frames:
        unmatched = players.copy()
        unmatched["current_export_match_method"] = pd.NA
        return apply_current_overrides(unmatched)

    matches = pd.concat(match_frames, ignore_index=True)
    matches = matches.sort_values(["match_priority", "match_method"])
    matches = matches.drop_duplicates("_player_row_id", keep="first")
    matches = matches.drop_duplicates("_current_record_id", keep="first")

    payload = current.copy()
    payload = payload.rename(
        columns={
            col: col if col in {"_current_record_id"} or col.startswith("current_") else f"current_{col}"
            for col in payload.columns
        }
    )
    merged = players_indexed.merge(
        matches[["_player_row_id", "_current_record_id", "match_method"]],
        on="_player_row_id",
        how="left",
    ).merge(payload, on="_current_record_id", how="left")
    merged = merged.rename(columns={"match_method": "current_export_match_method"})
    merged = merged.drop(columns=[col for col in ["_player_row_id", "_current_record_id", "player_last_key"] if col in merged.columns])
    return apply_current_overrides(merged)


def load_balldontlie_features(path: Path | None) -> pd.DataFrame:
    """Read optional BALLDONTLIE enrichment features defensively."""
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        print(f"Warning: BALLDONTLIE features file not found: {path}")
        return pd.DataFrame()

    frame = normalize_columns(read_table(path))
    if frame.empty:
        return frame

    player_col = first_present(frame, ["player_name", "name"])
    team_col = first_present(frame, ["team_name", "team", "country_name"])
    if player_col is None or team_col is None:
        print(f"Warning: BALLDONTLIE features file {path} is missing player_name or team_name.")
        return pd.DataFrame()

    work = frame.copy()
    work["bd_player_key"] = work[player_col].map(normalize_player_name)
    work["bd_team_key"] = work[team_col].map(normalize_team)
    if "position" in work.columns:
        work["bd_position_key"] = work["position"].astype("string").str.upper()

    if "balldontlie_player_id" in work.columns:
        work["balldontlie_player_id"] = clean_id(work["balldontlie_player_id"])

    numeric_columns = [
        "live_form_score",
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
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    rename_map = {
        "minutes": "bd_minutes",
        "starts": "bd_starts",
        "goals": "bd_goals",
        "assists": "bd_assists",
        "xg": "bd_xg",
        "xgot": "bd_xgot",
        "injury_status": "bd_injury_status",
        "injury_description": "bd_injury_description",
        "lineup_rows": "bd_lineup_rows",
        "matches_in_stats": "bd_matches_in_stats",
        "shots": "bd_shots",
        "shots_on_target": "bd_shots_on_target",
        "yellow_cards": "bd_yellow_cards",
        "red_cards": "bd_red_cards",
    }
    for source, target in rename_map.items():
        if source in work.columns:
            work[target] = work[source]

    work = work.drop_duplicates(subset=["bd_player_key", "bd_team_key"], keep="first")
    return work


def attach_balldontlie_features(players: pd.DataFrame, features_path: Path | None) -> pd.DataFrame:
    """Join optional BALLDONTLIE enrichment data onto the ranking table."""
    features = load_balldontlie_features(features_path)
    if features.empty:
        return players

    out = players.copy()
    if "player_name_standardized" not in out.columns:
        out["player_name_standardized"] = out.get("display_name", pd.Series("", index=out.index)).map(normalize_player_name)
    if "team_key" not in out.columns:
        out["team_key"] = out.get("team", pd.Series("", index=out.index)).map(normalize_team)

    feature_columns = [
        "balldontlie_player_id",
        "live_form_score",
        "bd_minutes",
        "bd_starts",
        "bd_goals",
        "bd_assists",
        "bd_xg",
        "bd_xgot",
        "bd_injury_status",
        "bd_injury_description",
        "bd_lineup_rows",
        "bd_matches_in_stats",
        "bd_shots",
        "bd_shots_on_target",
        "bd_yellow_cards",
        "bd_red_cards",
    ]
    feature_subset = [col for col in feature_columns if col in features.columns]
    features = features[["bd_player_key", "bd_team_key", *feature_subset]].drop_duplicates(
        subset=["bd_player_key", "bd_team_key"],
        keep="first",
    )

    merged = out.merge(
        features,
        left_on=["player_name_standardized", "team_key"],
        right_on=["bd_player_key", "bd_team_key"],
        how="left",
    )
    merged = merged.drop(columns=[col for col in ["bd_player_key", "bd_team_key"] if col in merged.columns])
    return merged


def prepare_current_export(current: pd.DataFrame) -> pd.DataFrame:
    current = current.copy()
    name_col = first_present(current, ["display_name", "name", "player", "full_name", "player_name"])
    team_col = first_present(current, ["country", "team", "nation", "squad"])
    pos_col = first_present(current, ["fantasy_position", "position", "pos"])
    short_name_col = first_present(current, ["short_name", "web_name", "known_as", "last_name"])

    if "player_id" in current.columns:
        current["player_id"] = clean_id(current["player_id"])
    elif "id" in current.columns:
        current["player_id"] = clean_id(current["id"])

    if name_col:
        current["player_key"] = current[name_col].map(normalize_player_name)
    if short_name_col:
        current["short_player_key"] = current[short_name_col].map(normalize_player_name)
    if team_col:
        current["team_key"] = current[team_col].map(normalize_team)
    if pos_col and "current_position" not in current.columns:
        current["current_position"] = current[pos_col].astype("string").str.upper()

    for column in ["fantasy_price", "price", "cost", "value", "total_points"]:
        if column in current.columns:
            current[column] = pd.to_numeric(current[column], errors="coerce")
    return current


def add_unique_matches(
    match_frames: list[pd.DataFrame],
    candidates: pd.DataFrame,
    priority: int,
    method: str,
) -> None:
    if candidates.empty:
        return
    work = candidates[["_player_row_id", "_current_record_id"]].dropna().drop_duplicates()
    if work.empty:
        return
    current_counts = work.groupby("_current_record_id")["_player_row_id"].transform("nunique")
    player_counts = work.groupby("_player_row_id")["_current_record_id"].transform("nunique")
    work = work[(current_counts == 1) & (player_counts == 1)].copy()
    if work.empty:
        return
    work["match_priority"] = priority
    work["match_method"] = method
    match_frames.append(work)


def build_current_name_variants(current: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for column in ["player_key", "short_player_key"]:
        if column in current.columns:
            piece = current[["_current_record_id", column]].rename(columns={column: "match_name_key"})
            pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=["_current_record_id", "match_name_key"])
    variants = pd.concat(pieces, ignore_index=True)
    variants["match_name_key"] = variants["match_name_key"].astype("string")
    return variants[variants["match_name_key"].notna() & variants["match_name_key"].ne("")].drop_duplicates()


def build_current_team_variants(current: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for column in ["team_key", "fixture_home_key", "fixture_away_key"]:
        if column in current.columns:
            piece = current[["_current_record_id", column]].rename(columns={column: "match_team_key"})
            pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=["_current_record_id", "match_team_key"])
    variants = pd.concat(pieces, ignore_index=True)
    variants["match_team_key"] = variants["match_team_key"].astype("string")
    return variants[variants["match_team_key"].notna() & variants["match_team_key"].ne("")].drop_duplicates()


def build_name_team_matches(players: pd.DataFrame, current: pd.DataFrame, use_last_name: bool) -> pd.DataFrame:
    names = build_current_name_variants(current)
    teams = build_current_team_variants(current)
    if names.empty or teams.empty:
        return pd.DataFrame()
    current_candidates = names.merge(teams, on="_current_record_id", how="inner")
    player_name_col = "player_last_key" if use_last_name else "player_key"
    current_candidates = current_candidates.rename(
        columns={"match_name_key": player_name_col, "match_team_key": "team_key"}
    )
    return players[["_player_row_id", player_name_col, "team_key"]].merge(
        current_candidates,
        on=[player_name_col, "team_key"],
        how="inner",
    )


def build_unique_name_matches(players: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    names = build_current_name_variants(current)
    if names.empty:
        return pd.DataFrame()

    player_names = players[["_player_row_id", "player_key"]].dropna()
    unique_player_names = player_names[
        player_names.groupby("player_key")["_player_row_id"].transform("nunique") == 1
    ]
    names = names.rename(columns={"match_name_key": "player_key"})
    unique_current_names = names[
        names.groupby("player_key")["_current_record_id"].transform("nunique") == 1
    ]
    return unique_player_names.merge(unique_current_names, on="player_key", how="inner")


def last_name_key(player_key: Any) -> str:
    key = normalize_player_name(player_key)
    if not key:
        return ""
    return key.split()[-1]


def apply_current_overrides(players: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    price_col = first_present(
        out,
        ["current_fantasy_price", "current_price", "current_cost", "current_value"],
    )
    if price_col:
        current_price = pd.to_numeric(out[price_col], errors="coerce")
        if "fantasy_price" in out.columns:
            if "fantasy_price_original" not in out.columns:
                out["fantasy_price_original"] = out["fantasy_price"]
            out["fantasy_price"] = current_price.combine_first(out["fantasy_price"])
            out["fantasy_price_source"] = current_price.notna().map({True: "current_export", False: "base"})

    current_position_col = first_present(out, ["current_current_position", "current_fantasy_position", "current_position"])
    if current_position_col and "position" in out.columns:
        current_position = out[current_position_col].astype("string").str.upper()
        missing_position = out["position"].astype("string").str.upper().isin(["", "UNK", "<NA>"])
        out.loc[missing_position & current_position.notna(), "position"] = current_position
    return out


def aggregate_stats(raw_dir: Path) -> pd.DataFrame:
    stats_path = raw_dir / "wc2026_player_stats_apifootball.csv"
    if not stats_path.exists():
        return pd.DataFrame()

    stats = normalize_columns(read_table(stats_path))
    if "player_id" in stats.columns:
        stats["player_id"] = clean_id(stats["player_id"])

    name_col = first_present(stats, ["display_name", "name", "player", "player_name"])
    team_col = first_present(stats, ["country", "nation", "national_team", "team_country"])
    if name_col:
        stats["player_key"] = stats[name_col].map(normalize_player_name)
    if team_col:
        stats["team_key"] = stats[team_col].map(normalize_team)

    count_columns = [
        "games",
        "started",
        "minutes",
        "goals",
        "assists",
        "tackles",
        "interceptions",
        "duels",
        "duels_won",
        "saves",
        "goals_conceded",
    ]
    stats = to_numeric(stats, count_columns + ["rating", "season_start_year"])
    available_counts = [col for col in count_columns if col in stats.columns]

    if "player_id" in stats.columns and stats["player_id"].notna().any():
        keys = ["player_id"]
    elif {"player_key", "team_key"}.issubset(stats.columns):
        keys = ["player_key", "team_key"]
    else:
        return pd.DataFrame()

    aggregate = stats.groupby(keys, dropna=False)[available_counts].sum(min_count=1).reset_index()
    aggregate = aggregate.rename(columns={col: f"hist_{col}" for col in available_counts})

    if "rating" in stats.columns:
        rating = weighted_rating(stats, keys)
        aggregate = aggregate.merge(rating, on=keys, how="left")

    if "season_start_year" in stats.columns and stats["season_start_year"].notna().any():
        max_year = int(stats["season_start_year"].max())
        recent = stats[stats["season_start_year"] >= max_year - 1]
        recent_agg = recent.groupby(keys, dropna=False)[available_counts].sum(min_count=1).reset_index()
        recent_agg = recent_agg.rename(columns={col: f"recent_{col}" for col in available_counts})
        aggregate = aggregate.merge(recent_agg, on=keys, how="left")

    if "comp_type" in stats.columns:
        national = stats[stats["comp_type"].astype("string").str.lower().eq("national")]
        if not national.empty:
            national_agg = national.groupby(keys, dropna=False)[available_counts].sum(min_count=1).reset_index()
            national_agg = national_agg.rename(columns={col: f"nt_{col}" for col in available_counts})
            aggregate = aggregate.merge(national_agg, on=keys, how="left")

    aggregate = add_rate_columns(aggregate)
    return aggregate


def weighted_rating(stats: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    work = stats[stats["rating"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=keys + ["hist_rating"])
    if "minutes" in work.columns:
        work["_rating_weight"] = work["minutes"].fillna(0)
    elif "games" in work.columns:
        work["_rating_weight"] = work["games"].fillna(0)
    else:
        work["_rating_weight"] = 1
    work.loc[work["_rating_weight"] <= 0, "_rating_weight"] = 1
    work["_weighted_rating"] = work["rating"] * work["_rating_weight"]
    grouped = work.groupby(keys, dropna=False)[["_weighted_rating", "_rating_weight"]].sum().reset_index()
    grouped["hist_rating"] = grouped["_weighted_rating"] / grouped["_rating_weight"]
    return grouped[keys + ["hist_rating"]]


def add_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    minutes = pd.to_numeric(out.get("hist_minutes", pd.Series(0, index=out.index)), errors="coerce").replace(0, pd.NA)
    games = pd.to_numeric(out.get("hist_games", pd.Series(0, index=out.index)), errors="coerce").replace(0, pd.NA)

    for stat in ["goals", "assists", "tackles", "interceptions", "saves", "goals_conceded"]:
        column = f"hist_{stat}"
        if column in out.columns:
            out[f"{stat}_per90"] = (pd.to_numeric(out[column], errors="coerce") / minutes * 90).fillna(0)

    if "hist_started" in out.columns:
        out["historical_start_rate"] = (pd.to_numeric(out["hist_started"], errors="coerce") / games).fillna(0)
    else:
        out["historical_start_rate"] = 0

    if {"hist_goals", "hist_assists"}.issubset(out.columns):
        out["goal_assist_per90"] = (
            (pd.to_numeric(out["hist_goals"], errors="coerce").fillna(0) + pd.to_numeric(out["hist_assists"], errors="coerce").fillna(0))
            / minutes
            * 90
        ).fillna(0)

    return out


def merge_player_stats(players: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty:
        return players
    if "player_id" in players.columns and "player_id" in stats.columns:
        return players.merge(stats.drop_duplicates("player_id"), on="player_id", how="left", suffixes=("", "_stats"))
    if {"player_key", "team_key"}.issubset(players.columns) and {"player_key", "team_key"}.issubset(stats.columns):
        return players.merge(stats.drop_duplicates(["player_key", "team_key"]), on=["player_key", "team_key"], how="left", suffixes=("", "_stats"))
    return players


def load_fixtures(
    fixture_csv: Path,
    as_of: pd.Timestamp,
    target_round: int | None = None,
) -> pd.DataFrame:
    fixtures = normalize_columns(read_table(fixture_csv))
    home_col = first_present(fixtures, ["home_team", "home", "team_1", "team1", "team_a"]) or first_containing(fixtures, ["home"], ["score"])
    away_col = first_present(fixtures, ["away_team", "away", "team_2", "team2", "team_b"]) or first_containing(fixtures, ["away"], ["score"])
    date_col = first_present(fixtures, ["date", "match_date", "fixture_date", "datetime", "utc_date", "kickoff"]) or first_containing(fixtures, ["date"])

    if home_col is None or away_col is None:
        raise ValueError(
            f"Could not identify home/away team columns in {fixture_csv}. Found columns: {list(fixtures.columns)}"
        )

    target_round_col = first_present(fixtures, ["round", "matchday"])
    round_col = target_round_col or first_present(fixtures, ["stage", "group", "phase"])
    if target_round is not None and target_round_col is None:
        raise ValueError(
            "--target-round requires the fixture CSV to include either a 'round' or 'matchday' column. "
            f"Found columns after normalization: {list(fixtures.columns)}"
        )

    venue_col = first_present(fixtures, ["venue", "stadium", "location", "city"])
    status_col = first_present(fixtures, ["status", "match_status", "state"])
    result_col = first_present(fixtures, ["result", "score", "ft", "full_time"])
    home_score_col = first_present(fixtures, ["home_score", "home_goals", "home_team_score", "score_home"])
    away_score_col = first_present(fixtures, ["away_score", "away_goals", "away_team_score", "score_away"])

    work = fixtures.copy()
    work["_match_order"] = range(len(work))
    work["_home_team"] = work[home_col].astype("string")
    work["_away_team"] = work[away_col].astype("string")
    work["_home_key"] = work["_home_team"].map(normalize_team)
    work["_away_key"] = work["_away_team"].map(normalize_team)
    work["_fixture_date"] = pd.NaT
    if date_col:
        work["_fixture_date"] = pd.to_datetime(work[date_col], errors="coerce", utc=True)
    if round_col:
        work["_round"] = work[round_col]

    if target_round is not None:
        work = work[round_value_matches(work[target_round_col], target_round)].copy()
        if work.empty:
            raise ValueError(f"No fixtures found for --target-round {target_round}.")

    finished = pd.Series(False, index=work.index)
    if target_round is None:
        if status_col:
            status = work[status_col].astype("string").str.lower()
            finished = finished | status.str.contains("final|full time|finished|played|complete|ft", na=False)
        if result_col:
            finished = finished | work[result_col].astype("string").str.contains(r"\d+\s*[-:]\s*\d+", regex=True, na=False)
        if home_score_col and away_score_col:
            finished = finished | (pd.to_numeric(work[home_score_col], errors="coerce").notna() & pd.to_numeric(work[away_score_col], errors="coerce").notna())
        if work["_fixture_date"].notna().any():
            finished = finished | (work["_fixture_date"] < as_of)
    work["_finished"] = finished

    common = ["_match_order", "_fixture_date", "_finished"]
    if round_col:
        common.append("_round")
    if venue_col:
        work["_venue"] = work[venue_col]
        common.append("_venue")

    home = work[common + ["_home_key", "_away_key", "_away_team"]].rename(
        columns={"_home_key": "team_key", "_away_key": "opponent_key", "_away_team": "next_opponent"}
    )
    home["is_home"] = True

    away = work[common + ["_away_key", "_home_key", "_home_team"]].rename(
        columns={"_away_key": "team_key", "_home_key": "opponent_key", "_home_team": "next_opponent"}
    )
    away["is_home"] = False

    long = pd.concat([home, away], ignore_index=True)
    upcoming = long[long["team_key"].ne("")].copy()
    if target_round is None:
        upcoming = upcoming[~upcoming["_finished"]].copy()
    if upcoming.empty:
        return pd.DataFrame(columns=["team_key", "opponent_key", "next_opponent", "is_home", "next_fixture_date", "next_round", "next_venue"])

    upcoming = upcoming.sort_values(["_fixture_date", "_match_order"], na_position="last")
    next_fixture = upcoming.groupby("team_key", as_index=False).first()
    next_fixture = next_fixture.rename(
        columns={"_fixture_date": "next_fixture_date", "_round": "next_round", "_venue": "next_venue"}
    )
    keep = ["team_key", "opponent_key", "next_opponent", "is_home", "next_fixture_date", "next_round", "next_venue"]
    return next_fixture[[col for col in keep if col in next_fixture.columns]]


def round_value_matches(series: pd.Series, target_round: int) -> pd.Series:
    """Match round values such as 2, '2', 'Round 2', or 'Matchday 2'."""
    text = series.astype("string").str.lower().str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    direct_match = numeric.eq(target_round)
    extracted = pd.to_numeric(text.str.extract(r"(\d+)", expand=False), errors="coerce")
    return direct_match | extracted.eq(target_round)


def load_strength(raw_dir: Path, team_strength_csv: Path | None = None) -> pd.DataFrame:
    strength = pd.DataFrame(columns=["team_key"])
    elo_path = raw_dir / "elo-calibrated.json"
    if elo_path.exists():
        strength = merge_strength(strength, parse_elo_json(elo_path), "team_key")

    for path in [raw_dir / "probabilities.csv", raw_dir / "probabilities.json"]:
        if path.exists():
            strength = merge_strength(strength, parse_probabilities(path), "team_key")

    team_stats = parse_team_stats(raw_dir)
    if not team_stats.empty:
        strength = merge_strength(strength, team_stats, "team_key")

    custom_strength = parse_team_strength_csv(team_strength_csv)
    if not custom_strength.empty:
        strength = merge_strength(strength, custom_strength, "team_key")

    if strength.empty:
        return strength

    if "elo" in strength.columns:
        strength["elo_score"] = minmax(strength["elo"])
    else:
        strength["elo_score"] = 0.5

    if "title_probability" in strength.columns:
        strength["title_probability_score"] = minmax(strength["title_probability"])
    else:
        strength["title_probability_score"] = 0.5

    if "team_form_score_raw" in strength.columns:
        strength["team_form_score"] = minmax(strength["team_form_score_raw"])
    else:
        strength["team_form_score"] = 0.5

    if "custom_strength_score" in strength.columns:
        strength["custom_strength_component"] = pd.to_numeric(strength["custom_strength_score"], errors="coerce").fillna(0.5)
    else:
        strength["custom_strength_component"] = 0.5

    strength["team_strength_score"] = (
        0.45 * strength["elo_score"]
        + 0.25 * strength["title_probability_score"]
        + 0.15 * strength["team_form_score"]
        + 0.15 * strength["custom_strength_component"]
    ).fillna(0.5).clip(0, 1)
    return strength.drop_duplicates("team_key")


def merge_strength(left: pd.DataFrame, right: pd.DataFrame, key: str) -> pd.DataFrame:
    if right.empty:
        return left
    if left.empty or left[key].dropna().empty:
        return right.drop_duplicates(key)
    return left.merge(right.drop_duplicates(key), on=key, how="outer")


def parse_team_strength_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()

    frame = normalize_columns(read_table(path))
    if frame.empty:
        return pd.DataFrame()

    team_col = first_present(frame, ["team", "country", "nation", "name"])
    strength_col = first_present(frame, ["strength_score", "team_strength_score", "score", "strength"])
    if team_col is None or strength_col is None:
        raise ValueError(
            f"Team strength CSV {path} must contain at least 'team' and 'strength_score' columns."
        )

    out = pd.DataFrame(
        {
            "team_key": frame[team_col].map(normalize_team),
            "custom_strength_score": normalize_score_series(frame[strength_col]),
        }
    )

    optional_columns = {
        "win_probability_next_match": ["win_probability_next_match", "win_probability", "win_prob", "next_win_probability"],
        "expected_goals": ["expected_goals", "xg", "team_expected_goals"],
        "clean_sheet_probability": ["clean_sheet_probability", "clean_sheet_prob", "cs_probability"],
        "team_strength_source": ["source"],
    }
    for output_col, candidates in optional_columns.items():
        source_col = first_present(frame, candidates)
        if source_col is None:
            continue
        if output_col in {"win_probability_next_match", "clean_sheet_probability"}:
            out[output_col] = parse_probability_series(frame[source_col])
        elif output_col == "expected_goals":
            out[output_col] = pd.to_numeric(frame[source_col], errors="coerce")
        else:
            out[output_col] = frame[source_col].astype("string")

    out = out[out["team_key"].ne("") & out["custom_strength_score"].notna()]
    return out.drop_duplicates("team_key")


def parse_elo_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    rows: list[dict[str, Any]] = []
    if isinstance(data, dict):
        iterable = data.get("teams") or data.get("ratings") or data.get("data")
        if isinstance(iterable, list):
            rows = [row for row in iterable if isinstance(row, dict)]
        elif isinstance(iterable, dict):
            for key, value in iterable.items():
                if isinstance(value, (int, float)):
                    rows.append({"team": key, "elo": value})
                elif isinstance(value, dict):
                    row = {"team": key}
                    row.update(value)
                    rows.append(row)
        else:
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    rows.append({"team": key, "elo": value})
                elif isinstance(value, dict):
                    row = {"team": key}
                    row.update(value)
                    rows.append(row)
    elif isinstance(data, list):
        rows = [row for row in data if isinstance(row, dict)]

    frame = normalize_columns(pd.DataFrame(rows))
    if frame.empty:
        return pd.DataFrame()
    team_col = first_present(frame, ["team", "name", "country", "nation"]) or frame.columns[0]
    elo_col = first_present(frame, ["elo", "rating", "calibrated_elo", "elo_rating"]) or first_containing(frame, ["elo"])
    if elo_col is None:
        numeric = frame.select_dtypes(include="number").columns.tolist()
        elo_col = numeric[0] if numeric else None
    if elo_col is None:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "team_key": frame[team_col].map(normalize_team),
            "elo": pd.to_numeric(frame[elo_col], errors="coerce"),
        }
    ).dropna(subset=["elo"])


def parse_probabilities(path: Path) -> pd.DataFrame:
    frame = normalize_columns(read_probability_frame(path))
    if frame.empty:
        return pd.DataFrame()
    team_col = first_present(frame, ["team", "country", "nation", "name"]) or frame.columns[0]
    title_col = (
        first_present(frame, ["title_probability", "champion_probability", "win_probability", "winner_probability", "champion", "title"])
        or first_containing(frame, ["title"])
        or first_containing(frame, ["champ"])
        or first_containing(frame, ["win"])
    )
    advance_col = (
        first_present(frame, ["advance_probability", "knockout_probability", "round_of_16_probability", "r16_probability", "make_r16"])
        or first_containing(frame, ["advance"])
        or first_containing(frame, ["round", "16"])
    )

    out = pd.DataFrame({"team_key": frame[team_col].map(normalize_team)})
    if title_col:
        out["title_probability"] = parse_probability_series(frame[title_col])
    if advance_col:
        out["advance_probability"] = parse_probability_series(frame[advance_col])
    return out[out["team_key"].ne("")]


def read_probability_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".json":
        return read_table(path)

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return pd.DataFrame(data)

    if isinstance(data, dict):
        for key in ("teams", "data", "probabilities", "rows"):
            if isinstance(data.get(key), list):
                return pd.DataFrame(data[key])

        rows: list[dict[str, Any]] = []
        for team, value in data.items():
            if isinstance(value, dict):
                row = {"team": team}
                row.update(value)
                rows.append(row)
            elif isinstance(value, (int, float, str)):
                rows.append({"team": team, "title_probability": value})
        if rows:
            return pd.DataFrame(rows)

    return pd.DataFrame()


def parse_probability_series(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.replace("%", "", regex=False)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().gt(1).any():
        numeric = numeric / 100
    return numeric


def normalize_score_series(series: pd.Series) -> pd.Series:
    """Accept strength scores expressed as 0-1, 0-100, or arbitrary ratings."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return numeric
    if valid.min() >= 0 and valid.max() <= 1:
        return numeric
    if valid.min() >= 0 and valid.max() <= 100:
        return numeric / 100
    return minmax(numeric)


def parse_team_stats(raw_dir: Path) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("wc2026_team_stats*.csv"))
    rows = []
    for path in paths:
        try:
            frame = normalize_columns(read_table(path))
        except Exception:
            continue
        team_col = first_present(frame, ["team", "country", "nation", "team_name", "name"])
        if team_col is None:
            continue
        wins_col = first_present(frame, ["wins", "win", "w"])
        draws_col = first_present(frame, ["draws", "draw", "d"])
        losses_col = first_present(frame, ["losses", "loss", "l"])
        gf_col = first_present(frame, ["goals_for", "gf", "goals_scored"])
        ga_col = first_present(frame, ["goals_against", "ga", "goals_conceded"])
        frame = to_numeric(frame, [col for col in [wins_col, draws_col, losses_col, gf_col, ga_col] if col])
        total = pd.Series(0, index=frame.index, dtype="float64")
        if wins_col:
            total = total + frame[wins_col].fillna(0) * 3
        if draws_col:
            total = total + frame[draws_col].fillna(0)
        if gf_col and ga_col:
            total = total + (frame[gf_col].fillna(0) - frame[ga_col].fillna(0)) * 0.35
        rows.append(pd.DataFrame({"team_key": frame[team_col].map(normalize_team), "team_form_score_raw": total}))

    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    return combined.groupby("team_key", as_index=False)["team_form_score_raw"].mean()


def attach_context(players: pd.DataFrame, fixtures: pd.DataFrame, strength: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    if not fixtures.empty:
        out = out.merge(fixtures, on="team_key", how="left")
    else:
        out["next_opponent"] = pd.NA
        out["opponent_key"] = pd.NA
        out["is_home"] = pd.NA
        out["next_fixture_date"] = pd.NaT

    if not strength.empty:
        team_strength = strength.add_prefix("team_").rename(columns={"team_team_key": "team_key"})
        out = out.merge(team_strength, on="team_key", how="left")
        opponent_strength = strength.add_prefix("opponent_").rename(columns={"opponent_team_key": "opponent_key"})
        out = out.merge(opponent_strength, on="opponent_key", how="left")

    for column in ["team_team_strength_score", "opponent_team_strength_score"]:
        if column not in out.columns:
            out[column] = 0.5
    if "team_elo" not in out.columns:
        out["team_elo"] = pd.NA
    if "opponent_elo" not in out.columns:
        out["opponent_elo"] = pd.NA

    team_elo = pd.to_numeric(out["team_elo"], errors="coerce")
    opponent_elo = pd.to_numeric(out["opponent_elo"], errors="coerce")
    has_fixture = out["next_opponent"].notna()
    has_elo = has_fixture & team_elo.notna() & opponent_elo.notna()
    elo_diff = team_elo - opponent_elo

    # Fixture score is Elo-driven when both teams have Elo. Positive Elo
    # difference means an easier fixture, negative means a harder fixture.
    out["fixture_ease_score"] = 0.5
    out.loc[has_elo, "fixture_ease_score"] = (0.5 + elo_diff.loc[has_elo] / 500).clip(0.05, 0.95)
    out["fixture_difficulty"] = 3
    out.loc[has_elo & elo_diff.ge(200), "fixture_difficulty"] = 1
    out.loc[has_elo & elo_diff.ge(75) & elo_diff.lt(200), "fixture_difficulty"] = 2
    out.loc[has_elo & elo_diff.gt(-75) & elo_diff.lt(75), "fixture_difficulty"] = 3
    out.loc[has_elo & elo_diff.gt(-200) & elo_diff.le(-75), "fixture_difficulty"] = 4
    out.loc[has_elo & elo_diff.le(-200), "fixture_difficulty"] = 5
    out.loc[out["next_opponent"].isna(), "fixture_difficulty"] = pd.NA
    out["team_strength_score"] = pd.to_numeric(out["team_team_strength_score"], errors="coerce").fillna(0.5)
    return out


def score_players(players: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    # First-pass BALLDONTLIE enrichment only: keep the signal bounded so it can
    # nudge the ranking without overpowering the open-data baseline.
    BALLDONTLIE_MAX_BONUS = 1.5
    BALLDONTLIE_MAX_PENALTY = 1.5
    BALLDONTLIE_OUT_PENALTY = 1.5
    BALLDONTLIE_ACTIVE_INJURY_PENALTY = 0.75

    numeric_defaults = {
        "goals_per90": 0,
        "assists_per90": 0,
        "tackles_per90": 0,
        "interceptions_per90": 0,
        "saves_per90": 0,
        "goals_conceded_per90": 0,
        "historical_start_rate": 0,
        "fantasy_price": pd.NA,
        "club_goals_4y": 0,
        "club_assists_4y": 0,
        "club_games_4y": 0,
        "nt_goals_4y": 0,
        "nt_assists_4y": 0,
        "nt_games_4y": 0,
        "hist_games": 0,
    }
    for column, default in numeric_defaults.items():
        if column not in out.columns:
            out[column] = default
        out[column] = pd.to_numeric(out[column], errors="coerce")

    fallback_games = (out["club_games_4y"].fillna(0) + out["nt_games_4y"].fillna(0)).replace(0, pd.NA)
    fallback_ga = (
        out["club_goals_4y"].fillna(0)
        + out["club_assists_4y"].fillna(0)
        + out["nt_goals_4y"].fillna(0)
        + out["nt_assists_4y"].fillna(0)
    ) / fallback_games
    out["goal_assist_per90"] = (out["goals_per90"].fillna(0) + out["assists_per90"].fillna(0)).where(
        (out["goals_per90"].fillna(0) + out["assists_per90"].fillna(0)) > 0,
        fallback_ga.fillna(0),
    )

    likelihood_col = first_present(out, ["starting_likelihood", "current_starting_likelihood", "starter_status"])
    if likelihood_col:
        out["starting_likelihood_used"] = out[likelihood_col].astype("string").fillna("Unknown")
    else:
        out["starting_likelihood_used"] = "Unknown"
    out["starting_likelihood_category"] = out["starting_likelihood_used"].map(classify_starting_likelihood)
    out["starting_likelihood_score"] = out["starting_likelihood_category"].map(
        {
            "Likely Starter": 1.0,
            "Maybe Starter": 0.55,
            "Unknown": 0.40,
            "Unlikely Starter": 0.20,
        }
    ).fillna(0.40)
    out["starting_penalty"] = out["starting_likelihood_category"].map(STARTING_PENALTIES).fillna(0.80)

    out["position"] = out["position"].astype("string").str.upper().fillna("UNK")
    out["production_raw"] = 0.0

    gk = out["position"].eq("GK")
    defender = out["position"].eq("DEF")
    midfielder = out["position"].eq("MID")
    forward = out["position"].eq("FWD")

    # Production is position-aware: goalkeepers get credit for saves, defenders
    # for defensive actions plus attacking returns, midfielders for a blend, and
    # forwards mostly for goals and assists.
    out.loc[gk, "production_raw"] = out.loc[gk, "saves_per90"].fillna(0) - 0.25 * out.loc[gk, "goals_conceded_per90"].fillna(0)
    out.loc[defender, "production_raw"] = (
        1.4 * out.loc[defender, "goals_per90"].fillna(0)
        + 1.0 * out.loc[defender, "assists_per90"].fillna(0)
        + 0.12 * out.loc[defender, "interceptions_per90"].fillna(0)
        + 0.05 * out.loc[defender, "tackles_per90"].fillna(0)
    )
    out.loc[midfielder, "production_raw"] = (
        1.5 * out.loc[midfielder, "goals_per90"].fillna(0)
        + 1.2 * out.loc[midfielder, "assists_per90"].fillna(0)
        + 0.08 * out.loc[midfielder, "interceptions_per90"].fillna(0)
        + 0.03 * out.loc[midfielder, "tackles_per90"].fillna(0)
    )
    out.loc[forward, "production_raw"] = (
        1.7 * out.loc[forward, "goals_per90"].fillna(0)
        + 1.2 * out.loc[forward, "assists_per90"].fillna(0)
    )
    out.loc[~(gk | defender | midfielder | forward), "production_raw"] = out.loc[
        ~(gk | defender | midfielder | forward), "goal_assist_per90"
    ].fillna(0)

    out["production_capped"] = cap_by_position(out, "production_raw", quantile=0.95)
    out["production_score"] = minmax_by_position(out, "production_capped").clip(0, 0.95)
    out["historical_score"] = (100 * out["production_score"]).round(2)

    # Minutes score combines official starter likelihood with historical starts.
    # This keeps a productive bench player below a comparable locked starter.
    # The model does not invent predicted lineups; without a lineup input file,
    # starting_likelihood and historical starts are only a minutes proxy.
    out["minutes_score"] = (
        0.65 * out["starting_likelihood_score"].fillna(0.5)
        + 0.35 * out["historical_start_rate"].clip(0, 1).fillna(0)
    ).clip(0, 1)

    # Fixture score is the inverse of next-opponent difficulty. Neutral is used
    # when the fixture or strength data is missing.
    out["fixture_score"] = pd.to_numeric(out.get("fixture_ease_score", 0.5), errors="coerce").fillna(0.5).clip(0, 1)

    # Team score is a transparent 15% component in the final score. If a custom
    # team-strength CSV is supplied, attackers and midfielders lean toward team
    # expected goals, while defenders and goalkeepers lean toward clean-sheet
    # probability. Missing optional fields fall back to the overall strength.
    base_team_score = pd.to_numeric(out.get("team_strength_score", 0.5), errors="coerce").fillna(0.5).clip(0, 1)
    custom_strength = optional_team_signal(out, "team_custom_strength_score", base_team_score, normalize_values=False)
    win_signal = optional_team_signal(out, "team_win_probability_next_match", base_team_score, normalize_values=False)
    attack_signal = optional_team_signal(out, "team_expected_goals", custom_strength, normalize_values=True)
    defense_signal = optional_team_signal(out, "team_clean_sheet_probability", custom_strength, normalize_values=False)
    out["team_attack_score"] = (0.70 * attack_signal + 0.20 * base_team_score + 0.10 * win_signal).clip(0, 1)
    out["team_defense_score"] = (0.70 * defense_signal + 0.20 * base_team_score + 0.10 * win_signal).clip(0, 1)
    out["team_score"] = base_team_score
    out.loc[midfielder | forward, "team_score"] = out.loc[midfielder | forward, "team_attack_score"]
    out.loc[gk | defender, "team_score"] = out.loc[gk | defender, "team_defense_score"]

    price = out["fantasy_price"].replace(0, pd.NA)
    position_median_price = out.groupby("position")["fantasy_price"].transform("median").replace(0, pd.NA)
    out["value_raw"] = (
        (0.70 * out["production_score"] + 0.30 * out["minutes_score"])
        / (price / position_median_price)
    )
    # Value score rewards cheaper players who still have a credible production
    # and minutes profile. Expensive players can still rank well via other parts.
    out["value_score"] = minmax_by_position(out, "value_raw").fillna(0.5)

    form_source = first_present(out, ["current_total_points", "form", "avg_points"])
    if form_source:
        out["current_form_raw"] = pd.to_numeric(out[form_source], errors="coerce")
    else:
        out["current_form_raw"] = 0
    out["current_form_score"] = minmax_by_position(out, "current_form_raw").fillna(0.5)

    bd_adjustment = pd.Series(0.0, index=out.index)
    has_balldontlie = False
    if "live_form_score" in out.columns:
        live_form = pd.to_numeric(out["live_form_score"], errors="coerce")
        out["bd_live_form_signal"] = minmax(live_form).fillna(0.5)
        has_balldontlie = True
    if "bd_live_form_signal" in out.columns:
        out["bd_live_form_bonus"] = ((out["bd_live_form_signal"] - 0.5) * 2.0).clip(-1, 1) * BALLDONTLIE_MAX_BONUS

    if "bd_injury_status" in out.columns:
        injury_text = out["bd_injury_status"].astype("string").str.strip().str.lower()
        out["bd_injury_penalty"] = 0.0
        has_balldontlie = True
    else:
        injury_text = pd.Series("", index=out.index, dtype="string")

    active_injury_mask = injury_text.notna() & ~injury_text.isin({"", "nan", "none", "available", "fit", "active", "healthy"})
    if has_balldontlie:
        if "bd_injury_penalty" not in out.columns:
            out["bd_injury_penalty"] = 0.0
        out.loc[active_injury_mask, "bd_injury_penalty"] = BALLDONTLIE_ACTIVE_INJURY_PENALTY
        out.loc[injury_text.str.contains("out", na=False), "bd_injury_penalty"] = BALLDONTLIE_OUT_PENALTY
        if "bd_live_form_bonus" not in out.columns:
            out["bd_live_form_bonus"] = 0.0
        out["bd_live_form_adjustment"] = (out["bd_live_form_bonus"] - out["bd_injury_penalty"]).clip(
            lower=-BALLDONTLIE_MAX_PENALTY,
            upper=BALLDONTLIE_MAX_BONUS,
        )
        bd_adjustment = out["bd_live_form_adjustment"].fillna(0)

    out["pre_penalty_score"] = (
        100
        * (
            0.25 * out["minutes_score"]
            + 0.25 * out["fixture_score"]
            + 0.20 * out["production_score"]
            + 0.15 * out["team_score"]
            + 0.10 * out["value_score"]
            + 0.05 * out["current_form_score"]
        )
    )
    out["pre_penalty_score"] = out["pre_penalty_score"] + bd_adjustment
    out["fantasy_shortlist_score"] = (out["pre_penalty_score"] * out["starting_penalty"]).round(2)
    out["final_score"] = (out["fantasy_shortlist_score"] + bd_adjustment).round(2)

    out["risk_notes"] = out.apply(build_risk_notes, axis=1)
    return out.sort_values(["final_score", "fantasy_shortlist_score", "fixture_score", "minutes_score"], ascending=False)


def classify_starting_likelihood(value: Any) -> str:
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "<na>", "unknown", "historical starts only"}:
        return "Unknown"
    if "unlikely" in text or "bench" in text:
        return "Unlikely Starter"
    if "maybe" in text or "rotation" in text:
        return "Maybe Starter"
    if "likely" in text or text == "starter":
        return "Likely Starter"
    return "Unknown"


def cap_by_position(df: pd.DataFrame, column: str, quantile: float) -> pd.Series:
    pieces = []
    for _, group in df.groupby("position", dropna=False):
        values = pd.to_numeric(group[column], errors="coerce").fillna(0)
        cap = values.quantile(quantile)
        if pd.isna(cap) or cap <= 0:
            pieces.append(values)
        else:
            pieces.append(values.clip(upper=cap))
    if not pieces:
        return pd.Series(dtype="float64")
    return pd.concat(pieces).sort_index()


def optional_team_signal(
    df: pd.DataFrame,
    column: str,
    fallback: pd.Series,
    normalize_values: bool,
) -> pd.Series:
    if column not in df.columns:
        return fallback.copy()
    values = pd.to_numeric(df[column], errors="coerce")
    if normalize_values:
        values = minmax(values)
    return values.fillna(fallback).clip(0, 1)


def build_risk_notes(row: pd.Series) -> str:
    risks: list[str] = []
    starter_category = row.get("starting_likelihood_category", "Unknown")
    if starter_category == "Maybe Starter":
        risks.append("maybe starter")
    elif starter_category == "Unknown":
        risks.append("unknown starter status")
    elif starter_category == "Unlikely Starter":
        risks.append("unlikely starter")
    if pd.to_numeric(row.get("hist_games"), errors="coerce") < 10:
        risks.append("sparse history")
    if pd.isna(row.get("next_opponent")):
        risks.append("no next fixture")
    elif pd.to_numeric(row.get("fixture_difficulty"), errors="coerce") >= 4:
        risks.append("hard fixture")
    if pd.to_numeric(row.get("current_total_points"), errors="coerce") < 0:
        risks.append("negative current fantasy points")
    if "current_export_match_method" in row.index and pd.isna(row.get("current_export_match_method")):
        risks.append("no current export match")
    if row.get("position") == "GK" and starter_category != "Likely Starter":
        risks.append("GK not likely starter")
    if pd.to_numeric(row.get("team_score"), errors="coerce") < 0.25:
        risks.append("weaker team")
    return "; ".join(risks)


def export_rankings(scored: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_columns = [
        "Player",
        "Position",
        "Team",
        "Opponent",
        "Round/Matchday",
        "Price",
        "Fixture difficulty",
        "Starting likelihood",
        "Historical score",
        "Final score",
        "Risk note",
        "display_name",
        "player_name_standardized",
        "position",
        "team",
        "team_standardized",
        "balldontlie_player_id",
        "live_form_score",
        "bd_minutes",
        "bd_starts",
        "bd_goals",
        "bd_assists",
        "bd_xg",
        "bd_xgot",
        "bd_injury_status",
        "bd_injury_description",
        "bd_lineup_rows",
        "bd_matches_in_stats",
        "bd_shots",
        "bd_shots_on_target",
        "bd_yellow_cards",
        "bd_red_cards",
        "bd_live_form_bonus",
        "bd_live_form_adjustment",
        "bd_injury_penalty",
        "fantasy_price",
        "fantasy_price_original",
        "fantasy_price_source",
        "percent_selected",
        "fantasy_shortlist_score",
        "production_score",
        "minutes_score",
        "team_score",
        "team_attack_score",
        "team_defense_score",
        "fixture_score",
        "value_score",
        "team_custom_strength_score",
        "team_win_probability_next_match",
        "team_expected_goals",
        "team_clean_sheet_probability",
        "team_strength_source",
        "next_opponent",
        "next_round",
        "is_home",
        "next_fixture_date",
        "fixture_difficulty",
        "starting_likelihood",
        "starting_likelihood_used",
        "starting_likelihood_category",
        "starting_penalty",
        "pre_penalty_score",
        "current_form_score",
        "historical_score",
        "final_score",
        "hist_games",
        "historical_start_rate",
        "goals_per90",
        "assists_per90",
        "interceptions_per90",
        "saves_per90",
        "hist_rating",
        "team_elo",
        "opponent_elo",
        "current_export_match_method",
        "current_fixture",
        "current_total_points",
        "risk_notes",
    ]
    export_frame = build_export_frame(scored)
    available = [col for col in export_columns if col in export_frame.columns]
    paths = {"all": output_dir / "player_rankings_all.csv"}
    export_frame[available].to_csv(paths["all"], index=False, encoding="utf-8-sig")

    for position in POSITION_ORDER:
        path = output_dir / f"player_rankings_{position}.csv"
        export_frame.loc[scored["position"].eq(position), available].to_csv(path, index=False, encoding="utf-8-sig")
        paths[position] = path
    return paths


def build_export_frame(scored: pd.DataFrame) -> pd.DataFrame:
    """Add user-facing output columns while preserving auditable raw columns."""
    aliases = {
        "Player": "display_name",
        "Position": "position",
        "Team": "team",
        "Opponent": "next_opponent",
        "Round/Matchday": "next_round",
        "Price": "fantasy_price",
        "Fixture difficulty": "fixture_difficulty",
        "Starting likelihood": "starting_likelihood_used",
        "Historical score": "historical_score",
        "Final score": "final_score",
        "Risk note": "risk_notes",
    }
    out = pd.DataFrame(index=scored.index)
    for label, source in aliases.items():
        if source in scored.columns:
            out[label] = scored[source]
        else:
            out[label] = pd.NA
    return pd.concat([out, scored], axis=1)


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int) -> str:
    subset = df.head(limit).copy()
    visible_columns: list[str] = []
    for col in columns:
        if col not in subset.columns or col in visible_columns:
            continue
        visible_columns.append(col)
    subset = subset[visible_columns]
    if subset.empty:
        return "_No rows available._"
    headers = list(subset.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in subset.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, pd.Series):
                non_null = value.dropna()
                value = non_null.iloc[0] if not non_null.empty else ""
            if isinstance(value, float):
                value = round(value, 2)
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    scored: pd.DataFrame,
    fixtures: pd.DataFrame,
    strength: pd.DataFrame,
    output_dir: Path,
    raw_dir: Path,
    fixture_csv: Path,
    as_of: pd.Timestamp,
    ranking_paths: dict[str, Path],
    current_export: Path | None = None,
    target_round: int | None = None,
    report_md: Path | None = None,
    alias_file: Path | None = None,
    team_strength_csv: Path | None = None,
    balldontlie_features_csv: Path | None = None,
) -> Path:
    report_path = report_md or (output_dir / "fantasy_shortlist_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()
    manifest_path = raw_dir / "source_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(read_text_with_fallback(manifest_path))
        except json.JSONDecodeError:
            manifest = {}

    report_frame = add_report_metrics(build_export_frame(scored))
    fixture_mode_note = (
        f"Target round/matchday: `{target_round}`"
        if target_round is not None
        else f"As-of for fixtures: `{as_of.isoformat()}`"
    )

    custom_strength_used = has_custom_team_strength(strength)
    source_lines = build_source_lines(
        raw_dir,
        fixture_csv,
        manifest,
        current_export,
        target_round,
        alias_file,
        team_strength_csv,
        custom_strength_used,
        balldontlie_features_csv,
    )
    warning_lines = build_team_match_warning_lines(report_frame, fixtures)
    strength_warning_lines = build_team_strength_warning_lines(report_frame, strength, team_strength_csv)
    bdl_used = balldontlie_features_csv is not None
    bdl_lines: list[str] = []
    bdl_positive = pd.DataFrame()
    bdl_negative = pd.DataFrame()
    bdl_injuries = pd.DataFrame()
    if bdl_used:
        matched_mask = report_frame["balldontlie_player_id"].notna() if "balldontlie_player_id" in report_frame.columns else pd.Series(False, index=report_frame.index)
        matched = int(matched_mask.sum())
        total = len(report_frame)
        unmatched = total - matched
        match_rate = (matched / total) if total else 0.0
        bdl_lines = [
            f"- Ranked players matched to BALLDONTLIE features: `{matched}` / `{total}` ({match_rate:.1%})",
            f"- Unmatched ranked players: `{unmatched}`",
            "- Optional player-level BALLDONTLIE enrichment is layered on top of the open/static baseline.",
            "- `live_form_score` is a first-pass signal and should not be treated as the final ranking model.",
        ]

        if "bd_live_form_adjustment" in report_frame.columns:
            bdl_positive = report_frame.loc[matched_mask].sort_values("bd_live_form_adjustment", ascending=False).head(10)
            bdl_negative = report_frame.loc[matched_mask].sort_values("bd_live_form_adjustment", ascending=True).head(10)
        if "bd_injury_status" in report_frame.columns:
            injury_text = report_frame["bd_injury_status"].astype("string").str.strip().str.lower()
            active_mask = matched_mask & injury_text.notna() & injury_text.ne("") & ~injury_text.isin({"nan", "none", "available", "fit", "active", "healthy"})
            bdl_injuries = report_frame.loc[active_mask].sort_values(
                ["bd_injury_penalty", "bd_live_form_adjustment"],
                ascending=[False, True],
            ).head(10)
    parts = [
        "# FIFA World Cup Fantasy Shortlist",
        "",
        f"Generated: `{generated}`",
        fixture_mode_note,
        f"Players ranked: `{len(report_frame)}`",
        "",
        "## Score Model",
        "",
        "Fantasy Shortlist Score = 25% starting/minutes + 25% fixture + 20% capped historical player score + 15% team strength + 10% price value + 5% current fantasy form, then a starting-likelihood penalty.",
        "",
        "- Production is normalized within each position and uses simple football actions: goals, assists, defensive actions, and goalkeeper saves.",
        "- Minutes combines official starting likelihood with historical start rate.",
        "- Team strength uses Elo, open title probabilities, optional team form, and optional custom team-strength CSV data when available.",
        "- Custom team strength is role-aware: expected goals helps MID/FWD, clean-sheet probability helps GK/DEF, and missing optional fields fall back to strength_score.",
        "- Fixture ease compares the player's team strength against the next opponent.",
        "- Value rewards cheaper players who still have credible production and minutes.",
        "",
        "## Data Freshness Notes",
        "",
        *source_lines,
        "",
        "## Team Match Warnings",
        "",
        *warning_lines,
        "",
        "## Team Strength Warnings",
        "",
        *strength_warning_lines,
    ]

    if bdl_used:
        parts.extend(
            [
                "",
                "## BALLDONTLIE Enrichment",
                "",
                *bdl_lines,
                "",
                "### Top Positive BALLDONTLIE Adjustments",
                "",
                markdown_table(bdl_positive, BDL_REPORT_COLUMNS, 10),
                "",
                "### Top Negative BALLDONTLIE Adjustments",
                "",
                markdown_table(bdl_negative, BDL_REPORT_COLUMNS, 10),
                "",
                "### Active BALLDONTLIE Injuries",
                "",
                markdown_table(bdl_injuries, BDL_REPORT_COLUMNS, 10),
            ]
        )

    parts.extend(
        [
            "",
            "## Top 10 Overall Players",
            "",
            markdown_table(report_frame, REPORT_COLUMNS, 10),
        ]
    )

    position_limits = {"GK": 5, "DEF": 10, "MID": 10, "FWD": 10}
    for position, limit in position_limits.items():
        position_rows = report_frame.loc[report_frame["Position"].eq(position)]
        parts.extend(
            [
                "",
                f"## Top {limit} {position}",
                "",
                markdown_table(position_rows, REPORT_COLUMNS, limit),
            ]
        )

    parts.extend(["", "## Best Value Players By Position"])
    for position in POSITION_ORDER:
        value_rows = (
            report_frame.loc[report_frame["Position"].eq(position)]
            .sort_values(["_report_value", "Final score"], ascending=False)
            .head(5)
        )
        parts.extend(["", f"### {position}", "", markdown_table(value_rows, REPORT_COLUMNS, 5)])

    captain_pool = report_frame[
        report_frame["Position"].isin(["MID", "FWD"])
        | (
            report_frame["Position"].eq("DEF")
            & pd.to_numeric(report_frame["Final score"], errors="coerce").ge(80)
            & report_frame["Risk note"].astype("string").fillna("").eq("")
        )
    ]
    captain_rows = captain_pool.sort_values(
        ["_captain_score", "Final score"], ascending=False
    ).head(10)

    risk_mask = (
        report_frame["Risk note"].astype("string").str.len().fillna(0).gt(0)
        | report_frame["_low_starting_likelihood"]
        | report_frame["_difficult_fixture"]
    )
    risk_rows = report_frame.loc[risk_mask].sort_values(
        ["_risk_score", "Final score"], ascending=[False, False]
    ).head(15)
    parts.extend(
        [
            "",
            "## Highest-Risk Players",
            "",
            markdown_table(risk_rows, REPORT_COLUMNS, 15),
            "",
            "## Best Captain Candidates",
            "",
            markdown_table(captain_rows, REPORT_COLUMNS, 10),
            "",
            "## Output Files",
            "",
            *[f"- `{label}`: `{path}`" for label, path in ranking_paths.items()],
        ]
    )

    report_path.write_text("\n".join(parts) + "\n", encoding="utf-8-sig")
    return report_path


def add_report_metrics(report_frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate report-only ranking helpers without changing CSV columns."""
    out = report_frame.copy()
    final_score = pd.to_numeric(out["Final score"], errors="coerce").fillna(0)
    price = pd.to_numeric(out["Price"], errors="coerce").replace(0, pd.NA)
    difficulty = pd.to_numeric(out["Fixture difficulty"], errors="coerce")
    starting_source = out["starting_likelihood_score"] if "starting_likelihood_score" in out.columns else pd.Series(0.5, index=out.index)
    starting = pd.to_numeric(starting_source, errors="coerce").fillna(0.5)

    # Best value is intentionally simple and transparent: final score per price.
    out["_report_value"] = (final_score / price).fillna(0)

    out["_low_starting_likelihood"] = starting.lt(0.55)
    out["_difficult_fixture"] = difficulty.ge(4).fillna(False)
    out["_risk_score"] = (
        out["Risk note"].astype("string").str.len().fillna(0).gt(0).astype(int) * 2
        + out["_low_starting_likelihood"].astype(int)
        + out["_difficult_fixture"].astype(int)
        + (1 - starting).clip(0, 1)
    )

    attacking_bonus = out["Position"].map({"FWD": 1.0, "MID": 0.95, "DEF": 0.20, "GK": 0.0}).fillna(0.4)
    fixture_ease = pd.to_numeric(out.get("fixture_score", pd.Series(0.5, index=out.index)), errors="coerce").fillna(0.5)
    # Captain candidates prioritize final score, a friendly fixture, reliable
    # minutes proxy, and attacking roles. No predicted lineups are invented.
    out["_captain_score"] = (
        0.55 * (final_score / 100).clip(0, 1)
        + 0.20 * fixture_ease.clip(0, 1)
        + 0.15 * starting.clip(0, 1)
        + 0.10 * attacking_bonus
    )
    return out


def build_balldontlie_feature_lines(features_csv: Path | None, scored: pd.DataFrame) -> list[str]:
    lines = [
        "- Optional player-level BALLDONTLIE enrichment is enabled, but the baseline still comes from open/static data.",
        "- `live_form_score` is a first-pass signal and should not be treated as the final ranking model.",
    ]
    if features_csv is not None:
        lines.insert(0, f"- BALLDONTLIE features file: `{features_csv}`")
        if features_csv.exists():
            modified = datetime.fromtimestamp(features_csv.stat().st_mtime, timezone.utc).isoformat()
            lines.append(f"- BALLDONTLIE features file modified UTC: `{modified}`")
        else:
            lines.append("- BALLDONTLIE features file was provided but not found.")

    if "live_form_score" in scored.columns:
        top_players = scored.loc[scored["live_form_score"].notna()].sort_values("live_form_score", ascending=False).head(5)
    else:
        top_players = pd.DataFrame()
    if not top_players.empty:
        lines.append("- Top enriched players are shown below for quick review.")
    else:
        lines.append("- No usable BALLDONTLIE enrichment rows were matched to ranked players.")
    return lines


def build_team_match_warning_lines(scored: pd.DataFrame, fixtures: pd.DataFrame) -> list[str]:
    """Report teams that still fail to connect after alias normalization."""
    lines: list[str] = []
    if "team_key" not in scored.columns:
        return ["_Team matching could not be checked because `team_key` is missing._"]

    player_teams = scored[["Team", "team_key", "Opponent"]].drop_duplicates()
    unmatched_players = player_teams[
        player_teams["team_key"].astype("string").ne("")
        & player_teams["Opponent"].isna()
    ].copy()
    if not unmatched_players.empty:
        lines.append("- Player teams with no matching fixture:")
        for _, row in unmatched_players.sort_values("Team").iterrows():
            lines.append(f"  - {row['Team']} -> `{row['team_key']}`")

    if not fixtures.empty and "team_key" in fixtures.columns:
        player_keys = set(scored["team_key"].dropna().astype(str))
        fixture_only = fixtures.loc[
            fixtures["team_key"].astype("string").ne("")
            & ~fixtures["team_key"].astype(str).isin(player_keys),
            ["team_key", "next_opponent"],
        ].drop_duplicates()
        if not fixture_only.empty:
            lines.append("- Fixture teams with no matching fantasy players:")
            for _, row in fixture_only.sort_values("team_key").iterrows():
                team_name = display_team_from_key(row["team_key"])
                lines.append(f"  - {team_name} -> `{row['team_key']}` vs {row['next_opponent']}")

    return lines or ["_No unmatched teams detected after alias normalization._"]


def has_custom_team_strength(strength: pd.DataFrame) -> bool:
    return (
        not strength.empty
        and "custom_strength_score" in strength.columns
        and strength["custom_strength_score"].notna().any()
    )


def build_team_strength_warning_lines(
    scored: pd.DataFrame,
    strength: pd.DataFrame,
    team_strength_csv: Path | None,
) -> list[str]:
    if team_strength_csv is None:
        return ["_No custom team-strength CSV was provided._"]
    if not team_strength_csv.exists():
        return [f"_Team-strength CSV was provided but not found: `{team_strength_csv}`._"]
    if not has_custom_team_strength(strength):
        return ["_Team-strength CSV was loaded but no usable `strength_score` rows were found._"]

    player_keys = set(scored["team_key"].dropna().astype(str)) if "team_key" in scored.columns else set()
    unmatched = strength.loc[
        strength["custom_strength_score"].notna()
        & strength["team_key"].astype("string").ne("")
        & ~strength["team_key"].astype(str).isin(player_keys),
        ["team_key"],
    ].drop_duplicates()
    if unmatched.empty:
        return ["_All custom team-strength teams matched at least one ranked player team._"]

    lines = ["- Team-strength teams with no matching fantasy players:"]
    for _, row in unmatched.sort_values("team_key").iterrows():
        lines.append(f"  - {display_team_from_key(row['team_key'])} -> `{row['team_key']}`")
    return lines


def build_source_lines(
    raw_dir: Path,
    fixture_csv: Path,
    manifest: dict[str, Any],
    current_export: Path | None,
    target_round: int | None = None,
    alias_file: Path | None = None,
    team_strength_csv: Path | None = None,
    custom_strength_used: bool = False,
    balldontlie_features_csv: Path | None = None,
) -> list[str]:
    lines = [f"- Fixture file: `{fixture_csv}`"]
    if fixture_csv.exists():
        modified = datetime.fromtimestamp(fixture_csv.stat().st_mtime, timezone.utc).isoformat()
        lines.append(f"- Fixture file modified UTC: `{modified}`")
    else:
        lines.append("- Fixture file was not found when the report was generated.")

    if target_round is not None:
        lines.append(f"- Fixture selection mode: target round/matchday `{target_round}`.")
    else:
        lines.append("- Fixture selection mode: next unplayed fixture from `--as-of`.")

    if alias_file:
        lines.append(f"- Team alias file: `{alias_file}`")
        if alias_file.exists():
            modified = datetime.fromtimestamp(alias_file.stat().st_mtime, timezone.utc).isoformat()
            lines.append(f"- Team alias file modified UTC: `{modified}`")
        else:
            lines.append("- Team alias file was not found; only built-in aliases were used.")

    if current_export:
        lines.append(f"- Current FIFA Fantasy export: `{current_export}`")
        if current_export.exists():
            modified = datetime.fromtimestamp(current_export.stat().st_mtime, timezone.utc).isoformat()
            lines.append(f"- Current export modified UTC: `{modified}`")
        else:
            lines.append("- Current export path was provided but not found.")

    if team_strength_csv:
        lines.append(f"- Custom team-strength CSV: `{team_strength_csv}`")
        if team_strength_csv.exists():
            modified = datetime.fromtimestamp(team_strength_csv.stat().st_mtime, timezone.utc).isoformat()
            used_text = "used" if custom_strength_used else "loaded but no usable strength rows found"
            lines.append(f"- Custom team-strength CSV modified UTC: `{modified}`")
            lines.append(f"- Custom team-strength status: `{used_text}`")
        else:
            lines.append("- Custom team-strength CSV was provided but not found.")
    else:
        lines.append("- Custom team-strength CSV: not provided; using built-in/open strength sources when available.")

    if balldontlie_features_csv:
        lines.append(f"- BALLDONTLIE features CSV: `{balldontlie_features_csv}`")
        if balldontlie_features_csv.exists():
            modified = datetime.fromtimestamp(balldontlie_features_csv.stat().st_mtime, timezone.utc).isoformat()
            lines.append(f"- BALLDONTLIE features CSV modified UTC: `{modified}`")
            lines.append("- BALLDONTLIE enrichment is optional player-level data layered on top of the open/static baseline.")
            lines.append("- `live_form_score` is a first-pass signal and should not be treated as the final ranking model.")
        else:
            lines.append("- BALLDONTLIE features CSV was provided but not found.")
    if manifest:
        lines.append(f"- Download manifest generated UTC: `{manifest.get('generated_at_utc', 'unknown')}`")
        for source in manifest.get("sources", []):
            lines.append(
                f"- {source.get('filename')}: {source.get('status')} from {source.get('url')}"
            )
    else:
        lines.append(f"- No source manifest found in `{raw_dir}`; data was loaded from local files only.")

    for required in ["wc2026_players.csv", "wc2026_player_stats_apifootball.csv"]:
        path = raw_dir / required
        if path.exists():
            modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            lines.append(f"- `{required}` modified UTC: `{modified}`")
        else:
            lines.append(f"- `{required}` missing.")

    return lines


def run_pipeline(
    fixture_csv: Path,
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("output"),
    current_export: Path | None = None,
    as_of: str | None = None,
    target_round: int | None = None,
    report_md: Path | None = None,
    alias_file: Path | None = Path("config/team_aliases.csv"),
    team_strength_csv: Path | None = None,
    balldontlie_features_csv: Path | None = None,
    download_missing: bool = True,
    overwrite_downloads: bool = False,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_paths = [raw_dir / "wc2026_players.csv", raw_dir / "wc2026_player_stats_apifootball.csv"]
    if download_missing and (overwrite_downloads or any(not path.exists() for path in required_paths)):
        download_all(raw_dir, overwrite=overwrite_downloads)

    if not fixture_csv.exists():
        raise FileNotFoundError(f"Fixture CSV not found: {fixture_csv}")

    missing_required = [path for path in required_paths if not path.exists()]
    if missing_required:
        missing_text = ", ".join(str(path) for path in missing_required)
        raise FileNotFoundError(f"Required raw data missing: {missing_text}. Run scripts/download_data.py first.")

    load_team_aliases(alias_file)

    as_of_ts = pd.Timestamp(as_of or datetime.now(timezone.utc).isoformat())
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    else:
        as_of_ts = as_of_ts.tz_convert("UTC")

    players = load_players(raw_dir, current_export=current_export)
    stats = aggregate_stats(raw_dir)
    players = merge_player_stats(players, stats)
    fixtures = load_fixtures(fixture_csv, as_of_ts, target_round=target_round)
    strength = load_strength(raw_dir, team_strength_csv=team_strength_csv)
    contextual = attach_context(players, fixtures, strength)
    contextual = attach_balldontlie_features(contextual, balldontlie_features_csv)
    scored = score_players(contextual)
    ranking_paths = export_rankings(scored, output_dir)
    report_path = write_report(
        scored,
        fixtures,
        strength,
        output_dir,
        raw_dir,
        fixture_csv,
        as_of_ts,
        ranking_paths,
        current_export,
        target_round,
        report_md,
        alias_file,
        team_strength_csv,
        balldontlie_features_csv,
    )

    return {
        "players": scored,
        "ranking_paths": ranking_paths,
        "report_path": report_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build FIFA World Cup Fantasy rankings.")
    parser.add_argument("--fixture-csv", type=Path, default=Path("world-cup_2026.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--current-export", type=Path)
    parser.add_argument("--report-md", type=Path, help="Markdown report path, for example outputs/matchday_2_report.md.")
    parser.add_argument("--team-aliases", type=Path, default=Path("config/team_aliases.csv"), help="CSV with alias,canonical columns for team-name normalization.")
    parser.add_argument("--team-strength-csv", type=Path, help="Optional CSV with team, strength_score, and optional match/team projection columns.")
    parser.add_argument("--balldontlie-features", type=Path, help="Optional CSV with BALLDONTLIE player_live_features.csv enrichment.")
    # --as-of is the rolling mode: matches before this timestamp are treated as
    # played, and each player gets the next unplayed fixture for their team.
    parser.add_argument("--as-of", help="Fixture cutoff timestamp, for example 2026-06-18 or 2026-06-18T12:00:00Z.")
    # --target-round is the slate mode: ignore played/unplayed status and attach
    # each player to the fixture in that exact round or matchday.
    parser.add_argument("--target-round", type=int, help="Attach players to fixtures from this round/matchday only, for example 2.")
    parser.add_argument("--no-download", action="store_true", help="Do not download missing raw data.")
    parser.add_argument("--overwrite-downloads", action="store_true", help="Re-download raw/open-data files before ranking.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_pipeline(
        fixture_csv=args.fixture_csv,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        current_export=args.current_export,
        as_of=args.as_of,
        target_round=args.target_round,
        report_md=args.report_md,
        alias_file=args.team_aliases,
        team_strength_csv=args.team_strength_csv,
        balldontlie_features_csv=args.balldontlie_features,
        download_missing=not args.no_download,
        overwrite_downloads=args.overwrite_downloads,
    )
    print(f"Wrote rankings to {result['ranking_paths']['all']}")
    print(f"Wrote report to {result['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
