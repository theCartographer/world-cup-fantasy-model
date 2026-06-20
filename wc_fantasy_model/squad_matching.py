from __future__ import annotations

import csv
import difflib
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from wc_fantasy_model.normalize import normalize_player_name, normalize_team


SAFE_INJURY_STATUSES = {"", "nan", "none", "available", "fit", "active", "healthy"}
PRICE_CANDIDATES = ["fantasy_price", "price", "fantasy_price_original"]


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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return normalize_columns(pd.read_csv(path, encoding="utf-8-sig"))


def first_present(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def get_series(frame: pd.DataFrame, column: str | None) -> pd.Series | None:
    if column is None or column not in frame.columns:
        return None
    value = frame.loc[:, column]
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return None
        value = value.iloc[:, 0]
    return value


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def normalize_position(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def surname_key(value: Any) -> str:
    key = normalize_player_name(value)
    if not key:
        return ""
    parts = key.split()
    return parts[-1] if parts else ""


def build_risk_note(row: pd.Series, low_minutes_threshold: float = 0.6) -> str:
    parts: list[str] = []
    match_quality = str(row.get("match_quality", "")).strip().lower()
    if match_quality == "unmatched":
        parts.append("unmatched")
    elif match_quality in {"surname", "fuzzy"}:
        parts.append("ambiguous")

    minutes = pd.to_numeric(row.get("expected_minutes_probability"), errors="coerce")
    if pd.notna(minutes) and minutes < low_minutes_threshold:
        parts.append(f"low expected minutes probability: {minutes:.2f}")

    injury_penalty = pd.to_numeric(row.get("bd_injury_penalty"), errors="coerce")
    injury_status = str(row.get("bd_injury_status", "")).strip().lower()
    if pd.notna(injury_penalty) and injury_penalty > 0:
        parts.append("injury penalty")
    elif injury_status and injury_status not in SAFE_INJURY_STATUSES:
        parts.append("injury penalty")

    fixture_difficulty = pd.to_numeric(row.get("fixture_difficulty"), errors="coerce")
    if pd.notna(fixture_difficulty) and fixture_difficulty >= 4:
        parts.append("hard fixture")

    deduped: list[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return "; ".join(deduped)


def build_rankings_view(frame: pd.DataFrame, low_minutes_threshold: float = 0.6) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    field_map = {
        "player": ["player", "display_name", "name"],
        "team": ["team", "team_standardized", "country_name"],
        "position": ["position"],
        "expected_fantasy_points": ["expected_fantasy_points"],
        "final_score": ["final_score", "fantasy_shortlist_score"],
        "opponent": ["opponent", "next_opponent"],
        "fixture_difficulty": ["fixture_difficulty"],
        "expected_minutes_probability": ["expected_minutes_probability"],
        "expected_playing_time_warning": ["expected_playing_time_warning"],
        "risk_note": ["risk_note", "risk_notes"],
        "bd_injury_penalty": ["bd_injury_penalty"],
        "bd_injury_status": ["bd_injury_status"],
        "bd_injury_description": ["bd_injury_description"],
    }
    price_col = first_present(frame, PRICE_CANDIDATES)
    for output_name, candidates in field_map.items():
        source_col = first_present(frame, candidates)
        series = get_series(frame, source_col)
        out[output_name] = series if series is not None else pd.NA
    if price_col is not None:
        out["price"] = pd.to_numeric(get_series(frame, price_col), errors="coerce")
    else:
        out["price"] = pd.NA
    out["player_key"] = out["player"].map(normalize_player_name)
    out["team_key"] = out["team"].map(normalize_team)
    out["position_key"] = out["position"].map(normalize_position)
    out["match_key"] = out["player_key"].fillna("") + "||" + out["team_key"].fillna("") + "||" + out["position_key"].fillna("")
    out["name_team_key"] = out["player_key"].fillna("") + "||" + out["team_key"].fillna("")
    out["surname_key"] = out["player"].map(surname_key)
    out["surname_team_position_key"] = out["surname_key"].fillna("") + "||" + out["team_key"].fillna("") + "||" + out["position_key"].fillna("")
    out["expected_fantasy_points"] = pd.to_numeric(out["expected_fantasy_points"], errors="coerce")
    out["final_score"] = pd.to_numeric(out["final_score"], errors="coerce")
    out["fixture_difficulty"] = pd.to_numeric(out["fixture_difficulty"], errors="coerce")
    out["bd_injury_penalty"] = pd.to_numeric(out["bd_injury_penalty"], errors="coerce").fillna(0)
    out["risk_note"] = out.apply(lambda row: build_risk_note(row, low_minutes_threshold), axis=1)
    return out


def selection_metric(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    expected = pd.to_numeric(get_series(out, "expected_fantasy_points"), errors="coerce") if "expected_fantasy_points" in out.columns else pd.Series(pd.NA, index=out.index)
    final = pd.to_numeric(get_series(out, "final_score"), errors="coerce") if "final_score" in out.columns else pd.Series(pd.NA, index=out.index)
    out["selection_expected_fantasy_points"] = expected
    out["selection_final_score"] = final
    if expected.notna().any():
        out["selection_score"] = expected.fillna(final)
    else:
        out["selection_score"] = final
    out["selection_score"] = pd.to_numeric(out["selection_score"], errors="coerce").fillna(0)
    out["selection_final_score"] = pd.to_numeric(out["selection_final_score"], errors="coerce").fillna(0)
    return out


def build_current_squad(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["player", "team", "position", "squad_role"]
    for column in required:
        if column not in frame.columns:
            raise ValueError(f"Current squad CSV must contain a {column} column.")

    out = pd.DataFrame(index=frame.index)
    out["player"] = get_series(frame, "player")
    out["team"] = get_series(frame, "team")
    out["position"] = get_series(frame, "position")
    out["squad_role"] = get_series(frame, "squad_role").astype("string").str.upper().fillna("XI")
    out["bench_order"] = pd.to_numeric(get_series(frame, "bench_order"), errors="coerce") if "bench_order" in frame.columns else pd.NA
    out["is_captain"] = get_series(frame, "is_captain").map(to_bool) if "is_captain" in frame.columns else False
    out["is_vice_captain"] = get_series(frame, "is_vice_captain").map(to_bool) if "is_vice_captain" in frame.columns else False
    out["locked"] = get_series(frame, "locked").map(to_bool) if "locked" in frame.columns else False
    out["notes"] = get_series(frame, "notes") if "notes" in frame.columns else ""
    out["player_key"] = out["player"].map(normalize_player_name)
    out["team_key"] = out["team"].map(normalize_team)
    out["position_key"] = out["position"].map(normalize_position)
    out["match_key"] = out["player_key"].fillna("") + "||" + out["team_key"].fillna("") + "||" + out["position_key"].fillna("")
    out["name_team_key"] = out["player_key"].fillna("") + "||" + out["team_key"].fillna("")
    return out


def load_player_aliases(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        return None
    aliases = read_csv(path)
    required = ["alias", "team", "position", "canonical_player"]
    missing = [column for column in required if column not in aliases.columns]
    if missing:
        raise ValueError("Player alias file must contain columns: alias, team, position, canonical_player")
    out = aliases[required].copy()
    out["alias_key"] = out["alias"].map(normalize_player_name)
    out["team_key"] = out["team"].map(normalize_team)
    out["position_key"] = out["position"].map(normalize_position)
    out["canonical_player_key"] = out["canonical_player"].map(normalize_player_name)
    out["alias_lookup_key"] = out["alias_key"].fillna("") + "||" + out["team_key"].fillna("") + "||" + out["position_key"].fillna("")
    out["canonical_lookup_key"] = out["canonical_player_key"].fillna("") + "||" + out["team_key"].fillna("") + "||" + out["position_key"].fillna("")
    return out.drop_duplicates(subset=["alias_lookup_key", "canonical_lookup_key"], keep="first")


def best_fuzzy_candidate(current_key: str, candidates: pd.DataFrame) -> pd.Series | None:
    if not current_key or candidates.empty:
        return None
    scored: list[tuple[float, int]] = []
    for idx, candidate in candidates.iterrows():
        candidate_key = str(candidate.get("player_key", ""))
        if not candidate_key:
            continue
        score = difflib.SequenceMatcher(None, current_key, candidate_key).ratio()
        scored.append((score, idx))
    if not scored:
        return None
    scored.sort(reverse=True)
    best_score, best_index = scored[0]
    if best_score < 0.92:
        return None
    if len(scored) > 1 and scored[1][0] >= best_score - 0.01:
        return None
    return candidates.loc[best_index]


def best_surname_candidate(current_row: pd.Series, rankings: pd.DataFrame) -> pd.Series | None:
    surname = surname_key(current_row.get("player"))
    team_key = str(current_row.get("team_key", ""))
    position_key = str(current_row.get("position_key", ""))
    if not surname or not team_key or not position_key:
        return None
    candidates = rankings.loc[
        rankings["surname_key"].fillna("").eq(surname)
        & rankings["team_key"].fillna("").eq(team_key)
        & rankings["position_key"].fillna("").eq(position_key)
    ]
    if len(candidates) != 1:
        return None
    return candidates.iloc[0]


def lookup_ranking_row(rankings: pd.DataFrame, key: str, column: str = "match_key") -> pd.Series | None:
    if not key or column not in rankings.columns:
        return None
    matches = rankings.loc[rankings[column].fillna("").eq(key)]
    if matches.empty:
        return None
    return matches.iloc[0]


def resolve_current_match(
    current_row: pd.Series,
    rankings: pd.DataFrame,
    aliases: pd.DataFrame | None = None,
) -> dict[str, Any]:
    current_team_key = str(current_row.get("team_key", ""))
    current_position_key = str(current_row.get("position_key", ""))
    current_match_key = str(current_row.get("match_key", ""))
    current_name_team_key = str(current_row.get("name_team_key", ""))
    current_player_key = str(current_row.get("player_key", ""))

    if aliases is not None and not aliases.empty:
        alias_key = current_player_key + "||" + current_team_key + "||" + current_position_key
        alias_rows = aliases.loc[aliases["alias_lookup_key"].fillna("").eq(alias_key)]
        if not alias_rows.empty:
            canonical_key = str(alias_rows.iloc[0].get("canonical_player_key", ""))
            canonical_lookup_key = canonical_key + "||" + current_team_key + "||" + current_position_key
            alias_match = lookup_ranking_row(rankings, canonical_lookup_key)
            if alias_match is not None:
                result = alias_match.to_dict()
                result["matched_player"] = alias_match.get("player", "")
                result["match_quality"] = "alias"
                result["match_method"] = "alias"
                result["match_key"] = alias_match.get("match_key", current_match_key)
                return result

    exact_match = lookup_ranking_row(rankings, current_match_key)
    if exact_match is not None:
        result = exact_match.to_dict()
        result["matched_player"] = exact_match.get("player", "")
        result["match_quality"] = "exact"
        result["match_method"] = "exact"
        result["match_key"] = exact_match.get("match_key", current_match_key)
        return result

    name_team_match = lookup_ranking_row(rankings, current_name_team_key, "name_team_key")
    if name_team_match is not None:
        result = name_team_match.to_dict()
        result["matched_player"] = name_team_match.get("player", "")
        result["match_quality"] = "exact"
        result["match_method"] = "exact"
        result["match_key"] = name_team_match.get("match_key", current_match_key)
        return result

    surname_match = best_surname_candidate(current_row, rankings)
    if surname_match is not None:
        result = surname_match.to_dict()
        result["matched_player"] = surname_match.get("player", "")
        result["match_quality"] = "surname"
        result["match_method"] = "surname"
        result["match_key"] = surname_match.get("match_key", current_match_key)
        return result

    same_team_position = rankings.loc[
        rankings["team_key"].fillna("").eq(current_team_key)
        & rankings["position_key"].fillna("").eq(current_position_key)
    ]
    fuzzy_match = best_fuzzy_candidate(current_player_key, same_team_position)
    if fuzzy_match is not None:
        result = fuzzy_match.to_dict()
        result["matched_player"] = fuzzy_match.get("player", "")
        result["match_quality"] = "fuzzy"
        result["match_method"] = "fuzzy"
        result["match_key"] = fuzzy_match.get("match_key", current_match_key)
        return result

    return {
        "matched_player": "",
        "match_quality": "unmatched",
        "match_method": "unmatched",
        "match_key": current_match_key,
    }


def is_active_injury(row: pd.Series) -> bool:
    penalty = pd.to_numeric(row.get("bd_injury_penalty"), errors="coerce")
    if pd.notna(penalty) and penalty > 0:
        return True
    status = str(row.get("bd_injury_status", "")).strip().lower()
    return bool(status and status not in SAFE_INJURY_STATUSES)


def match_current_to_rankings(
    current: pd.DataFrame,
    rankings: pd.DataFrame,
    aliases: pd.DataFrame | None = None,
    low_minutes_threshold: float = 0.6,
) -> pd.DataFrame:
    matched_rows: list[dict[str, Any]] = []
    for _, current_row in current.iterrows():
        resolved = resolve_current_match(current_row, rankings, aliases)
        merged = current_row.to_dict()
        merged["matched_player"] = resolved.get("matched_player", "")
        merged["match_quality"] = resolved.get("match_quality", "unmatched")
        merged["match_method"] = resolved.get("match_method", merged["match_quality"])
        merged["match_key"] = resolved.get("match_key", merged.get("match_key", ""))
        for column in [
            "team",
            "position",
            "expected_fantasy_points",
            "final_score",
            "price",
            "opponent",
            "fixture_difficulty",
            "expected_minutes_probability",
            "expected_playing_time_warning",
            "risk_note",
            "bd_injury_penalty",
            "bd_injury_status",
            "bd_injury_description",
        ]:
            if column in resolved and pd.notna(resolved.get(column)):
                merged[column] = resolved.get(column)
        matched_rows.append(merged)

    matched = pd.DataFrame(matched_rows)
    if matched.empty:
        return matched
    matched["match_method"] = matched.get("match_method", pd.Series("unmatched", index=matched.index)).fillna("unmatched")
    matched.loc[matched["match_method"].eq("unmatched"), "match_quality"] = "unmatched"
    matched["price"] = pd.to_numeric(matched["price"], errors="coerce")
    matched["expected_fantasy_points"] = pd.to_numeric(matched["expected_fantasy_points"], errors="coerce")
    matched["final_score"] = pd.to_numeric(matched["final_score"], errors="coerce")
    matched["fixture_difficulty"] = pd.to_numeric(matched["fixture_difficulty"], errors="coerce")
    matched["bd_injury_penalty"] = pd.to_numeric(matched["bd_injury_penalty"], errors="coerce").fillna(0)
    matched["risk_note"] = matched.apply(lambda row: build_risk_note(row, low_minutes_threshold), axis=1)
    return matched


def current_squad_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    xi = frame.loc[frame["squad_role"].eq("XI")].copy()
    bench = frame.loc[frame["squad_role"].eq("BENCH")].copy()
    xi = xi.sort_values(
        ["is_captain", "is_vice_captain", "selection_score", "selection_final_score"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    bench = bench.sort_values(
        ["bench_order", "selection_score", "selection_final_score"],
        ascending=[True, False, False],
        na_position="last",
    )
    return xi, bench
