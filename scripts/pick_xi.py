from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wc_fantasy_model.normalize import normalize_player_name, normalize_team


SQUAD_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
SAFE_INJURY_STATUSES = {"", "nan", "none", "available", "fit", "active", "healthy"}
PRICE_CANDIDATES = ["fantasy_price", "price", "fantasy_price_original"]
RANKING_DISPLAY_COLUMNS = [
    "player",
    "team",
    "position",
    "final_score",
    "price",
    "opponent",
    "fixture_difficulty",
    "bd_injury_penalty",
    "risk_note",
]
CURRENT_DISPLAY_COLUMNS = [
    "player",
    "matched_player",
    "match_quality",
    "team",
    "position",
    "price",
    "expected_fantasy_points",
    "final_score",
    "opponent",
    "fixture_difficulty",
    "expected_minutes_probability",
    "expected_playing_time_warning",
    "bd_injury_penalty",
    "risk_note",
    "squad_role",
    "bench_order",
    "is_captain",
    "is_vice_captain",
    "locked",
]
SUGGESTED_DISPLAY_COLUMNS = [
    "player",
    "matched_player",
    "match_quality",
    "team",
    "position",
    "price",
    "expected_fantasy_points",
    "final_score",
    "opponent",
    "fixture_difficulty",
    "expected_minutes_probability",
    "expected_playing_time_warning",
    "bd_injury_penalty",
    "risk_note",
    "squad_role",
    "bench_order",
]
ACTION_OUTPUT_COLUMNS = [
    "player",
    "matched_player",
    "match_quality",
    "team",
    "position",
    "price",
    "expected_fantasy_points",
    "final_score",
    "squad_role",
    "bench_order",
    "action",
    "risk_note",
]
TRANSFER_OUTPUT_COLUMNS = [
    "transfer_no",
    "action",
    "sell_player",
    "sell_team",
    "sell_position",
    "sell_price",
    "sell_reason",
    "buy_player",
    "buy_team",
    "buy_position",
    "buy_price",
    "buy_expected_fantasy_points",
    "buy_final_score",
    "buy_expected_minutes_probability",
    "buy_bd_injury_penalty",
    "buy_risk_note",
    "budget_after",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review a current World Cup squad and build a simple next-round plan.")
    parser.add_argument("--rankings-csv", type=Path, required=True)
    parser.add_argument("--current-squad-csv", type=Path)
    parser.add_argument("--player-aliases", type=Path, help="Optional CSV of alias,team,position,canonical_player mappings.")
    parser.add_argument(
        "--captain-min-expected-minutes-probability",
        type=float,
        default=0.75,
        help="Minimum expected minutes probability for safe captain candidates.",
    )
    parser.add_argument(
        "--planning-scope",
        choices=["free_transfers", "wildcard"],
        default="free_transfers",
        help="Plan either a small transfer set or a full wildcard rebuild.",
    )
    parser.add_argument(
        "--min-expected-minutes-probability",
        type=float,
        default=0.6,
        help="Minimum expected minutes probability for plan-mode player selection.",
    )
    parser.add_argument("--formation", default="3-4-3")
    parser.add_argument("--budget", type=float, default=100)
    parser.add_argument("--max-per-team", type=int, default=3)
    parser.add_argument("--free-transfers", type=int, default=2)
    parser.add_argument("--mode", choices=["review", "plan"], default="review")
    parser.add_argument("--allow-injury-risk", action="store_true")
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser


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


def unique_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    selected: list[str] = []
    for column in columns:
        if column in selected or column not in frame.columns:
            continue
        selected.append(column)
    return selected


def select_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    selected: dict[str, pd.Series] = {}
    for column in columns:
        if column in selected:
            continue
        values = get_series(frame, column)
        if values is None:
            continue
        selected[column] = values
    return pd.DataFrame(selected)


def scalarize(value: Any) -> Any:
    if isinstance(value, pd.Series):
        non_null = value.dropna()
        return non_null.iloc[0] if not non_null.empty else pd.NA
    return value


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


def format_value(value: Any) -> str:
    value = scalarize(value)
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        value = round(value, 2)
    return str(value).replace("|", "/")


def render_table(frame: pd.DataFrame, columns: Iterable[str], limit: int | None = None) -> str:
    subset = select_columns(frame, columns)
    if limit is not None:
        subset = subset.head(limit)
    if subset.empty:
        return "_No rows available._"
    headers = list(subset.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in subset.iterrows():
        lines.append("| " + " | ".join(format_value(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def append_section(parts: list[str], title: str, body: str) -> None:
    parts.extend(["", f"## {title}", "", body])


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def match_quality_rank(value: Any) -> int:
    quality = str(value).strip().lower()
    return {"unmatched": 0, "ambiguous": 1, "surname": 2, "fuzzy": 3, "exact": 4}.get(quality, 0)


def build_risk_note(row: pd.Series) -> str:
    parts: list[str] = []
    match_quality = str(row.get("match_quality", "")).strip().lower()
    if match_quality == "unmatched":
        parts.append("unmatched")
    elif match_quality in {"surname", "fuzzy"}:
        parts.append("ambiguous")

    minutes = pd.to_numeric(row.get("expected_minutes_probability"), errors="coerce")
    if pd.notna(minutes) and minutes < 0.6:
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


def build_rankings_view(frame: pd.DataFrame) -> pd.DataFrame:
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
    out["risk_note"] = out.apply(build_risk_note, axis=1)
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
        print("Warning: expected_fantasy_points is missing; falling back to final_score for squad selection.")
        out["selection_score"] = final
    out["selection_score"] = pd.to_numeric(out["selection_score"], errors="coerce").fillna(0)
    out["selection_final_score"] = pd.to_numeric(out["selection_final_score"], errors="coerce").fillna(0)
    return out


def normalize_formation(formation: str) -> tuple[int, int, int]:
    parts = formation.split("-")
    if len(parts) != 3:
        raise ValueError("--formation must look like 3-4-3.")
    try:
        defense, midfield, forwards = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--formation must contain three integers like 3-4-3.") from exc
    if defense < 0 or midfield < 0 or forwards < 0:
        raise ValueError("--formation numbers must be non-negative.")
    return defense, midfield, forwards


def is_active_injury(row: pd.Series) -> bool:
    penalty = pd.to_numeric(row.get("bd_injury_penalty"), errors="coerce")
    if pd.notna(penalty) and penalty > 0:
        return True
    status = str(row.get("bd_injury_status", "")).strip().lower()
    return bool(status and status not in SAFE_INJURY_STATUSES)


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
        raise FileNotFoundError(path)
    aliases = read_csv(path)
    required = ["alias", "team", "position", "canonical_player"]
    missing = [column for column in required if column not in aliases.columns]
    if missing:
        raise ValueError(
            f"Player alias file {path} must contain columns: alias, team, position, canonical_player"
        )
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
    current_player = current_row.get("player", "")
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


def match_current_to_rankings(current: pd.DataFrame, rankings: pd.DataFrame, aliases: pd.DataFrame | None = None) -> pd.DataFrame:
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
    matched["risk_note"] = matched.apply(build_risk_note, axis=1)
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


def has_prices(frame: pd.DataFrame) -> bool:
    price_source = get_series(frame, "price")
    price = pd.to_numeric(price_source, errors="coerce") if price_source is not None else pd.Series(dtype="float64")
    return price.notna().any()


def select_squad(
    rankings: pd.DataFrame,
    formation: tuple[int, int, int],
    budget: float,
    max_per_team: int,
    allow_injury_risk: bool,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if not has_prices(rankings):
        return pd.DataFrame(columns=list(rankings.columns)), ["No price column was available, so planned squad selection was skipped."]

    filtered = selection_metric(rankings)
    filtered["price"] = pd.to_numeric(filtered["price"], errors="coerce")
    filtered["selection_score"] = pd.to_numeric(filtered["selection_score"], errors="coerce").fillna(0)
    filtered["selection_final_score"] = pd.to_numeric(filtered["selection_final_score"], errors="coerce").fillna(0)
    filtered = filtered[filtered["price"].notna()].copy()
    skipped_price = len(rankings) - len(filtered)
    if skipped_price:
        warnings.append(f"Skipped {skipped_price} ranking rows with no price.")

    if not allow_injury_risk and "bd_injury_penalty" in filtered.columns:
        injury_mask = filtered.apply(is_active_injury, axis=1)
        skipped_injured = int(injury_mask.sum())
        if skipped_injured:
            warnings.append(f"Skipped {skipped_injured} injured ranking rows because --allow-injury-risk was not set.")
        filtered = filtered.loc[~injury_mask].copy()

    if "expected_minutes_probability" in filtered.columns:
        minute_signal = pd.to_numeric(filtered["expected_minutes_probability"], errors="coerce")
        low_minutes_mask = minute_signal.notna() & minute_signal.lt(0.15)
        skipped_low_minutes = int(low_minutes_mask.sum())
        if skipped_low_minutes and not allow_injury_risk:
            warnings.append(f"Skipped {skipped_low_minutes} rows with very low expected minutes.")
            filtered = filtered.loc[~low_minutes_mask].copy()

    quotas = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    selected_rows: list[pd.Series] = []
    team_counts: dict[str, int] = {}
    position_counts: dict[str, int] = {position: 0 for position in quotas}
    spent = 0.0

    ordered = filtered.sort_values(["selection_score", "selection_final_score", "price"], ascending=[False, False, True], na_position="last")
    for _, row in ordered.iterrows():
        position = str(row.get("position", "")).upper()
        if position not in quotas:
            continue
        if position_counts[position] >= quotas[position]:
            continue
        team_key = str(row.get("team_key", ""))
        if team_counts.get(team_key, 0) >= max_per_team:
            continue
        price = pd.to_numeric(row.get("price"), errors="coerce")
        if pd.isna(price) or spent + float(price) > budget:
            continue
        selected_rows.append(row)
        position_counts[position] += 1
        team_counts[team_key] = team_counts.get(team_key, 0) + 1
        spent += float(price)
        if all(position_counts[pos] >= quotas[pos] for pos in quotas):
            break

    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        return selected, warnings

    selected = selection_metric(selected)

    missing_positions = {position: quotas[position] - position_counts[position] for position in quotas if position_counts[position] < quotas[position]}
    if missing_positions:
        warnings.append(
            "Unable to fill all squad slots: "
            + ", ".join(f"{position} short {count}" for position, count in missing_positions.items())
        )

    return selected, warnings


def build_transfer_plan(
    current: pd.DataFrame,
    rankings: pd.DataFrame,
    budget: float,
    max_per_team: int,
    free_transfers: int,
    allow_injury_risk: bool,
    min_expected_minutes_probability: float,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if current is None or current.empty:
        return pd.DataFrame(), ["No current squad CSV was supplied, so free-transfer planning was skipped."]
    if free_transfers <= 0:
        return pd.DataFrame(), ["No free transfers were available."]

    current = selection_metric(current)
    rankings = selection_metric(rankings)

    current = current.copy()
    rankings = rankings.copy()
    current["price"] = pd.to_numeric(current["price"], errors="coerce")
    rankings["price"] = pd.to_numeric(rankings["price"], errors="coerce")
    current["expected_fantasy_points"] = pd.to_numeric(current["expected_fantasy_points"], errors="coerce")
    rankings["expected_fantasy_points"] = pd.to_numeric(rankings["expected_fantasy_points"], errors="coerce")
    current["final_score"] = pd.to_numeric(current["final_score"], errors="coerce")
    rankings["final_score"] = pd.to_numeric(rankings["final_score"], errors="coerce")
    current["expected_minutes_probability"] = pd.to_numeric(get_series(current, "expected_minutes_probability"), errors="coerce") if "expected_minutes_probability" in current.columns else pd.Series(index=current.index, dtype="float64")
    rankings["expected_minutes_probability"] = pd.to_numeric(get_series(rankings, "expected_minutes_probability"), errors="coerce") if "expected_minutes_probability" in rankings.columns else pd.Series(index=rankings.index, dtype="float64")
    current["bd_injury_penalty"] = pd.to_numeric(current.get("bd_injury_penalty"), errors="coerce").fillna(0) if "bd_injury_penalty" in current.columns else 0
    rankings["bd_injury_penalty"] = pd.to_numeric(rankings.get("bd_injury_penalty"), errors="coerce").fillna(0) if "bd_injury_penalty" in rankings.columns else 0

    current_counts = current["team"].dropna().astype(str).value_counts().to_dict()
    current_total = pd.to_numeric(current["price"], errors="coerce").fillna(0).sum()
    current_keys = set(current["match_key"].astype(str)) if "match_key" in current.columns else set()
    used_buy_keys: set[str] = set()
    rows: list[dict[str, Any]] = []

    safe_current = current.loc[
        (pd.to_numeric(current["expected_minutes_probability"], errors="coerce").fillna(0) < min_expected_minutes_probability)
        | (pd.to_numeric(current["bd_injury_penalty"], errors="coerce").fillna(0) > 0)
        | current["match_quality"].astype(str).str.lower().isin({"unmatched", "ambiguous", "surname", "fuzzy"})
    ].copy()
    if safe_current.empty:
        safe_current = current.copy()

    safe_current["sell_rank_quality"] = safe_current["match_quality"].map(match_quality_rank)
    safe_current["sell_minutes"] = pd.to_numeric(safe_current["expected_minutes_probability"], errors="coerce").fillna(0)
    safe_current["sell_injury"] = pd.to_numeric(safe_current["bd_injury_penalty"], errors="coerce").fillna(0)
    safe_current["sell_efp"] = pd.to_numeric(safe_current["expected_fantasy_points"], errors="coerce").fillna(0)
    safe_current = safe_current.sort_values(
        ["sell_rank_quality", "sell_minutes", "sell_injury", "sell_efp"],
        ascending=[True, True, False, True],
        na_position="last",
    )

    candidate_pool = rankings.loc[~rankings["match_key"].astype(str).isin(current_keys)].copy() if "match_key" in rankings.columns else rankings.copy()
    candidate_pool["candidate_minutes"] = pd.to_numeric(candidate_pool["expected_minutes_probability"], errors="coerce").fillna(0)
    candidate_pool["candidate_injury"] = pd.to_numeric(candidate_pool["bd_injury_penalty"], errors="coerce").fillna(0)
    candidate_pool["candidate_price"] = pd.to_numeric(candidate_pool["price"], errors="coerce")
    candidate_pool["candidate_efp"] = pd.to_numeric(candidate_pool["expected_fantasy_points"], errors="coerce").fillna(0)
    candidate_pool["candidate_final"] = pd.to_numeric(candidate_pool["final_score"], errors="coerce").fillna(0)

    def buy_pool_for_position(position: str, remaining_budget: float, team_counts: dict[str, int]) -> pd.DataFrame:
        pool = candidate_pool.loc[candidate_pool["position"].astype(str).str.upper().eq(position)].copy()
        if pool.empty:
            pool = candidate_pool.copy()
        pool = pool.loc[pool["candidate_minutes"].ge(min_expected_minutes_probability)]
        if not allow_injury_risk:
            pool = pool.loc[pool["candidate_injury"].le(0)]
        if pool.empty:
            return pool
        allowed_teams = [team for team, count in team_counts.items() if count < max_per_team]
        if allowed_teams:
            pool = pool.loc[pool["team"].astype(str).isin(allowed_teams) | ~pool["team"].astype(str).isin(team_counts.keys())]
        pool = pool.loc[pool["candidate_price"].notna() & pool["candidate_price"].le(remaining_budget)]
        return pool.sort_values(
            ["candidate_efp", "candidate_final", "candidate_minutes", "candidate_price"],
            ascending=[False, False, False, True],
            na_position="last",
        )

    for _, sell_row in safe_current.head(free_transfers).iterrows():
        if len(rows) >= free_transfers:
            break
        sell_price = pd.to_numeric(sell_row.get("price"), errors="coerce")
        sell_price = float(sell_price) if pd.notna(sell_price) else 0.0
        sell_team = str(sell_row.get("team", ""))
        sell_position = str(sell_row.get("position", "")).upper()
        sell_reason_parts: list[str] = []
        if str(sell_row.get("match_quality", "")).lower() in {"unmatched", "ambiguous", "surname", "fuzzy"}:
            sell_reason_parts.append(str(sell_row.get("match_quality", "")))
        sell_minutes = pd.to_numeric(sell_row.get("expected_minutes_probability"), errors="coerce")
        if pd.notna(sell_minutes) and sell_minutes < min_expected_minutes_probability:
            sell_reason_parts.append(f"low minutes {sell_minutes:.2f}")
        sell_injury = pd.to_numeric(sell_row.get("bd_injury_penalty"), errors="coerce")
        if pd.notna(sell_injury) and sell_injury > 0:
            sell_reason_parts.append("injury penalty")
        sell_efp = pd.to_numeric(sell_row.get("expected_fantasy_points"), errors="coerce")
        if pd.notna(sell_efp):
            sell_reason_parts.append(f"efp {sell_efp:.2f}")
        remaining_budget = budget - (current_total - sell_price)

        buy_choice = pd.Series(dtype="object")
        buy_pool = buy_pool_for_position(sell_position, remaining_budget, current_counts)
        if buy_pool.empty:
            warnings.append(f"No safe buy candidate found for {sell_row.get('player', '')} at position {sell_position}.")
            continue
        for _, candidate in buy_pool.iterrows():
            candidate_key = str(candidate.get("match_key", ""))
            if candidate_key in used_buy_keys:
                continue
            candidate_team = str(candidate.get("team", ""))
            candidate_price = pd.to_numeric(candidate.get("price"), errors="coerce")
            final_budget = current_total - sell_price + (float(candidate_price) if pd.notna(candidate_price) else 0.0)
            final_counts = current_counts.copy()
            if sell_team:
                final_counts[sell_team] = max(final_counts.get(sell_team, 0) - 1, 0)
            if candidate_team:
                final_counts[candidate_team] = final_counts.get(candidate_team, 0) + 1
            if final_budget > budget:
                continue
            if final_counts.get(candidate_team, 0) > max_per_team:
                continue
            buy_choice = candidate
            current_total = final_budget
            current_counts = final_counts
            used_buy_keys.add(candidate_key)
            break

        if buy_choice.empty:
            warnings.append(f"No budget- and team-compliant buy candidate found for {sell_row.get('player', '')}.")
            continue

        rows.append(
            {
                "transfer_no": len(rows) + 1,
                "action": "SELL/BUY",
                "sell_player": sell_row.get("player", ""),
                "sell_team": sell_row.get("team", ""),
                "sell_position": sell_row.get("position", ""),
                "sell_price": sell_price,
                "sell_reason": "; ".join(sell_reason_parts) if sell_reason_parts else str(sell_row.get("risk_note", "")),
                "buy_player": buy_choice.get("player", ""),
                "buy_team": buy_choice.get("team", ""),
                "buy_position": buy_choice.get("position", ""),
                "buy_price": buy_choice.get("price", ""),
                "buy_expected_fantasy_points": buy_choice.get("expected_fantasy_points", ""),
                "buy_final_score": buy_choice.get("final_score", ""),
                "buy_expected_minutes_probability": buy_choice.get("expected_minutes_probability", ""),
                "buy_bd_injury_penalty": buy_choice.get("bd_injury_penalty", 0),
                "buy_risk_note": buy_choice.get("risk_note", ""),
                "budget_after": current_total,
            }
        )

    if not rows:
        warnings.append("No transfer pair met the plan-mode filters.")

    plan = pd.DataFrame(rows)
    if not plan.empty:
        plan["sell_price"] = pd.to_numeric(plan["sell_price"], errors="coerce")
        plan["buy_price"] = pd.to_numeric(plan["buy_price"], errors="coerce")
        plan["buy_expected_fantasy_points"] = pd.to_numeric(plan["buy_expected_fantasy_points"], errors="coerce")
        plan["buy_final_score"] = pd.to_numeric(plan["buy_final_score"], errors="coerce")
        plan["buy_expected_minutes_probability"] = pd.to_numeric(plan["buy_expected_minutes_probability"], errors="coerce")
        plan["buy_bd_injury_penalty"] = pd.to_numeric(plan["buy_bd_injury_penalty"], errors="coerce").fillna(0)
    return plan, warnings


def choose_xi(selected: pd.DataFrame, formation: tuple[int, int, int]) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    needed = {"GK": 1, "DEF": formation[0], "MID": formation[1], "FWD": formation[2]}
    work = selected.copy()
    work = selection_metric(work)
    starters: list[pd.DataFrame] = []
    used_indexes: list[int] = []
    for position, count in needed.items():
        position_rows = work.loc[work["position"].astype(str).str.upper().eq(position)].sort_values(
            ["selection_score", "selection_final_score", "price"],
            ascending=[False, False, True],
            na_position="last",
        )
        chosen = position_rows.head(count)
        starters.append(chosen)
        used_indexes.extend(chosen.index.tolist())
    xi = pd.concat(starters) if starters else pd.DataFrame(columns=work.columns)
    xi = xi.copy()
    xi["squad_role"] = "XI"
    xi["bench_order"] = pd.NA
    bench = work.loc[~work.index.isin(used_indexes)].copy()
    bench = bench.sort_values(["selection_score", "selection_final_score", "price"], ascending=[False, False, True], na_position="last")
    bench["squad_role"] = "BENCH"
    bench["bench_order"] = range(1, len(bench) + 1)
    return pd.concat([xi, bench], ignore_index=True)


def make_action_table(current: pd.DataFrame | None, suggested: pd.DataFrame) -> pd.DataFrame:
    current_map = {}
    suggested_map = {}
    if current is not None and not current.empty:
        for _, row in current.iterrows():
            current_map[str(row.get("match_key", ""))] = row
    for _, row in suggested.iterrows():
        suggested_map[str(row.get("match_key", ""))] = row

    rows: list[dict[str, Any]] = []
    for match_key in sorted(set(current_map) | set(suggested_map)):
        current_row = current_map.get(match_key)
        suggested_row = suggested_map.get(match_key)
        if current_row is None:
            action = "BUY" if suggested_row is not None else ""
        elif suggested_row is not None:
            action = "KEEP"
        else:
            action = "SELL"
        source_row = suggested_row if suggested_row is not None else current_row
        player = ""
        matched_player = ""
        match_quality = ""
        if current_row is not None:
            player = current_row.get("player", "")
            matched_player = current_row.get("matched_player", current_row.get("player", ""))
            match_quality = current_row.get("match_quality", "unmatched")
        elif suggested_row is not None:
            player = suggested_row.get("player", "")
            matched_player = suggested_row.get("matched_player", suggested_row.get("player", ""))
            match_quality = suggested_row.get("match_quality", "exact")
        rows.append(
            {
                "player": player if player != "" else (source_row.get("player", "") if source_row is not None else ""),
                "matched_player": matched_player if matched_player != "" else (source_row.get("matched_player", source_row.get("player", "")) if source_row is not None else ""),
                "match_quality": match_quality if match_quality != "" else (source_row.get("match_quality", "exact") if source_row is not None else ""),
                "team": source_row.get("team", "") if source_row is not None else "",
                "position": source_row.get("position", "") if source_row is not None else "",
                "price": source_row.get("price", "") if source_row is not None else "",
                "expected_fantasy_points": source_row.get("expected_fantasy_points", "") if source_row is not None else "",
                "final_score": source_row.get("final_score", "") if source_row is not None else "",
                "squad_role": source_row.get("squad_role", "") if source_row is not None else "",
                "bench_order": source_row.get("bench_order", "") if source_row is not None else "",
                "action": action,
                "risk_note": source_row.get("risk_note", "") if source_row is not None else "",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["expected_fantasy_points"] = pd.to_numeric(out["expected_fantasy_points"], errors="coerce")
        out["final_score"] = pd.to_numeric(out["final_score"], errors="coerce")
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
    return out


def build_output_rows(mode: str, current: pd.DataFrame | None, suggested: pd.DataFrame | None) -> pd.DataFrame:
    base_columns = [column for column in ACTION_OUTPUT_COLUMNS if column != "action"]
    if mode == "review":
        if current is None or current.empty:
            return pd.DataFrame(columns=ACTION_OUTPUT_COLUMNS)
        out = select_columns(current, base_columns).copy()
        out["action"] = "CURRENT"
        if "matched_player" not in out.columns:
            out["matched_player"] = out.get("player", "")
        if "match_quality" not in out.columns:
            out["match_quality"] = "unmatched"
        out["expected_fantasy_points"] = pd.to_numeric(out["expected_fantasy_points"], errors="coerce")
        out["final_score"] = pd.to_numeric(out["final_score"], errors="coerce")
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
        return out

    if current is None or current.empty:
        if suggested is None:
            return pd.DataFrame(columns=ACTION_OUTPUT_COLUMNS)
        out = select_columns(suggested, base_columns).copy()
        out["action"] = "BUY"
        out["matched_player"] = out.get("player", "")
        out["match_quality"] = "exact"
        out["expected_fantasy_points"] = pd.to_numeric(out["expected_fantasy_points"], errors="coerce")
        out["final_score"] = pd.to_numeric(out["final_score"], errors="coerce")
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
        return out

    if suggested is None or suggested.empty:
        if current is None or current.empty:
            return pd.DataFrame(columns=ACTION_OUTPUT_COLUMNS)
        out = select_columns(current, base_columns).copy()
        out["action"] = "CURRENT"
        if "matched_player" not in out.columns:
            out["matched_player"] = out.get("player", "")
        if "match_quality" not in out.columns:
            out["match_quality"] = "unmatched"
        out["expected_fantasy_points"] = pd.to_numeric(out["expected_fantasy_points"], errors="coerce")
        out["final_score"] = pd.to_numeric(out["final_score"], errors="coerce")
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
        return out

    return make_action_table(current, suggested if current is not None else None)


def build_transfer_output_rows(transfer_plan: pd.DataFrame | None) -> pd.DataFrame:
    if transfer_plan is None or transfer_plan.empty:
        return pd.DataFrame(columns=TRANSFER_OUTPUT_COLUMNS)
    return select_columns(transfer_plan, TRANSFER_OUTPUT_COLUMNS).copy()


def build_budget_summary(current: pd.DataFrame | None, suggested: pd.DataFrame | None, budget: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current_total = pd.to_numeric(current["price"], errors="coerce").fillna(0).sum() if current is not None and not current.empty and "price" in current.columns else pd.NA
    suggested_total = pd.to_numeric(suggested["price"], errors="coerce").fillna(0).sum() if suggested is not None and not suggested.empty and "price" in suggested.columns else pd.NA
    rows.append({"scope": "budget", "current_total": current_total, "suggested_total": suggested_total, "budget": budget, "remaining": (budget - suggested_total) if pd.notna(suggested_total) else pd.NA})
    return pd.DataFrame(rows)


def build_team_counts(current: pd.DataFrame | None, suggested: pd.DataFrame | None, max_per_team: int) -> pd.DataFrame:
    teams = set()
    if current is not None and not current.empty:
        teams.update(current["team"].dropna().astype(str))
    if suggested is not None and not suggested.empty:
        teams.update(suggested["team"].dropna().astype(str))
    rows: list[dict[str, Any]] = []
    for team in sorted(teams):
        row = {"team": team, "max_per_team": max_per_team}
        row["current_count"] = int((current["team"].astype(str) == team).sum()) if current is not None and not current.empty else 0
        row["suggested_count"] = int((suggested["team"].astype(str) == team).sum()) if suggested is not None and not suggested.empty else 0
        rows.append(row)
    return pd.DataFrame(rows)


def build_risk_warnings(current: pd.DataFrame | None, suggested: pd.DataFrame | None, budget: float, max_per_team: int, missing_price_count: int) -> list[str]:
    warnings: list[str] = []
    if missing_price_count:
        warnings.append(f"Skipped {missing_price_count} candidates with no price.")

    for label, frame in [("current", current), ("suggested", suggested)]:
        if frame is None or frame.empty:
            continue
        over_team = frame.groupby("team").size()
        over_team = over_team[over_team > max_per_team]
        if not over_team.empty:
            warnings.append(f"{label.title()} squad exceeds max-per-team for: " + ", ".join(f"{team} ({count})" for team, count in over_team.items()))

        if "price" in frame.columns:
            total = pd.to_numeric(frame["price"], errors="coerce").fillna(0).sum()
            if pd.notna(total) and total > budget:
                warnings.append(f"{label.title()} squad budget is over limit by {total - budget:.2f}.")

        injured = frame.loc[frame.apply(is_active_injury, axis=1)] if "bd_injury_penalty" in frame.columns else pd.DataFrame()
        if not injured.empty:
            warnings.append(f"{label.title()} squad has {len(injured)} players with active injury warnings.")
    return warnings


def render_report(
    mode: str,
    current: pd.DataFrame | None,
    suggested: pd.DataFrame | None,
    transfer_plan: pd.DataFrame | None,
    budget: float,
    max_per_team: int,
    free_transfers: int,
    current_locked: bool,
    plan_warnings: list[str],
    allow_injury_risk: bool,
    captain_min_expected_minutes_probability: float,
    planning_scope: str,
    min_expected_minutes_probability: float,
) -> str:
    parts: list[str] = ["# Current Squad Planner"]

    if current is not None and not current.empty:
        current = selection_metric(current)
        xi, bench = current_squad_summary(current)
        append_section(parts, "Current Squad Review", "")
        if current_locked:
            parts.extend(["", "Current round is locked/live. Treat this report as review-only; no immediate transfers are recommended."])
        xi_minutes = pd.to_numeric(get_series(xi, "expected_minutes_probability"), errors="coerce") if "expected_minutes_probability" in xi.columns else pd.Series(index=xi.index, dtype="float64")
        bench_minutes = pd.to_numeric(get_series(bench, "expected_minutes_probability"), errors="coerce") if "expected_minutes_probability" in bench.columns else pd.Series(index=bench.index, dtype="float64")
        low_xi = xi.loc[xi_minutes.fillna(1).lt(0.6)]
        low_bench = bench.loc[bench_minutes.fillna(1).lt(0.6)]
        if not low_xi.empty:
            parts.extend(["", f"- Warning: {len(low_xi)} XI players have expected minutes probability below 0.6: " + ", ".join(low_xi["player"].astype(str).tolist())])
        if not low_bench.empty:
            parts.extend(["", f"- Warning: {len(low_bench)} bench players have expected minutes probability below 0.6: " + ", ".join(low_bench["player"].astype(str).tolist())])
        parts.extend([
            "",
            "### Starting XI",
            "",
            render_table(xi, CURRENT_DISPLAY_COLUMNS, 11),
            "",
            "### Bench",
            "",
            render_table(bench, CURRENT_DISPLAY_COLUMNS, 10),
        ])

        captain = current.loc[current["is_captain"]].copy()
        vice = current.loc[current["is_vice_captain"]].copy()
        candidate_pool = xi.copy()
        if not allow_injury_risk:
            candidate_minutes = pd.to_numeric(get_series(candidate_pool, "expected_minutes_probability"), errors="coerce")
            candidate_injury = pd.to_numeric(get_series(candidate_pool, "bd_injury_penalty"), errors="coerce").fillna(0)
            safe_mask = candidate_minutes.fillna(0).ge(captain_min_expected_minutes_probability) & candidate_injury.fillna(0).le(0)
            risky_candidates = candidate_pool.loc[~safe_mask].copy()
            candidate_pool = candidate_pool.loc[safe_mask].copy()
        else:
            risky_candidates = pd.DataFrame(columns=xi.columns)
        top_candidates = candidate_pool.sort_values(
            ["expected_fantasy_points", "expected_minutes_probability", "selection_score", "selection_final_score"],
            ascending=[False, False, False, False],
            na_position="last",
        )
        append_section(parts, "Captain Review", "")
        captain_warning = ""
        if not captain.empty and not top_candidates.empty:
            captain_match = captain.iloc[0].get("match_key", "")
            candidate_keys = top_candidates.head(3)["match_key"].astype(str).tolist()
            if str(captain_match) not in candidate_keys:
                captain_warning = "- Warning: current captain is not among the top 3 current XI captain candidates."
        elif not captain.empty and top_candidates.empty:
            captain_warning = "- Warning: no safe captain candidates were available."
        if not captain.empty:
            captain_minutes = pd.to_numeric(get_series(captain, "expected_minutes_probability"), errors="coerce").iloc[0]
            if pd.notna(captain_minutes) and captain_minutes < captain_min_expected_minutes_probability:
                captain_warning = (captain_warning + "\n" if captain_warning else "") + "- Warning: current captain is below the captain minutes threshold."
        if not vice.empty:
            vice_minutes = pd.to_numeric(get_series(vice, "expected_minutes_probability"), errors="coerce").iloc[0]
            if pd.notna(vice_minutes) and vice_minutes < captain_min_expected_minutes_probability:
                captain_warning = (captain_warning + "\n" if captain_warning else "") + "- Warning: current vice captain is below the captain minutes threshold."
        parts.extend(
            [
                "",
                "### Current Captain",
                "",
                render_table(captain, CURRENT_DISPLAY_COLUMNS, 5),
                "",
                "### Current Vice Captain",
                "",
                render_table(vice, CURRENT_DISPLAY_COLUMNS, 5),
                "",
                "### Best Captain Candidates in Current XI",
                "",
                render_table(top_candidates, CURRENT_DISPLAY_COLUMNS, 5),
                captain_warning,
            ]
        )
        if not risky_candidates.empty:
            risky_candidates = risky_candidates.sort_values(
                ["expected_fantasy_points", "expected_minutes_probability", "selection_score", "selection_final_score"],
                ascending=[False, False, False, False],
                na_position="last",
            )
            parts.extend([
                "",
                "### High Projection but Risky Captain Options",
                "",
                render_table(risky_candidates, CURRENT_DISPLAY_COLUMNS, 10),
            ])

    if mode == "plan":
        if planning_scope == "wildcard" and suggested is not None and not suggested.empty:
            suggested = selection_metric(suggested)
            suggested_xi = suggested.loc[suggested["squad_role"].eq("XI")].copy()
            suggested_bench = suggested.loc[suggested["squad_role"].eq("BENCH")].copy()
            append_section(parts, "Wildcard Suggested XI", "")
            parts.extend([
                "",
                render_table(suggested_xi, SUGGESTED_DISPLAY_COLUMNS, 11),
                "",
                "### Wildcard Suggested Bench",
                "",
                render_table(suggested_bench, SUGGESTED_DISPLAY_COLUMNS, 10),
            ])

            if current is not None and not current.empty:
                transfer = make_action_table(current, suggested)
                transfer_count = int((transfer["action"].isin(["BUY", "SELL"])).sum())
                transfer_lines = [f"- Transfer count: `{transfer_count}`", f"- Free transfers: `{free_transfers}`"]
                if transfer_count > free_transfers:
                    transfer_lines.append("- Warning: transfer count exceeds free transfers.")
                append_section(parts, "Wildcard Transfer Plan", "\n".join(transfer_lines))
                parts.extend(["", render_table(transfer, ACTION_OUTPUT_COLUMNS, 30)])
            else:
                append_section(parts, "Wildcard Transfer Plan", "No current squad CSV was supplied, so this is a standalone wildcard squad plan.")
        else:
            append_section(parts, "Recommended Transfers", "")
            if current is None or current.empty:
                parts.extend(["", "No current squad CSV was supplied, so free-transfer planning was skipped."])
            elif transfer_plan is None or transfer_plan.empty:
                parts.extend(["", "No transfer pairs met the plan-mode filters."])
            else:
                transfer_count = len(transfer_plan)
                transfer_lines = [
                    f"- Transfer count: `{transfer_count}`",
                    f"- Free transfers: `{free_transfers}`",
                    f"- Planning scope: `free_transfers`",
                    f"- Minimum expected minutes probability: `{min_expected_minutes_probability}`",
                ]
                if transfer_count > free_transfers:
                    transfer_lines.append("- Warning: transfer count exceeds free transfers.")
                if transfer_count < free_transfers:
                    transfer_lines.append("- Note: fewer than the maximum free transfers were recommended because of the available filters.")
                append_section(parts, "Transfer Summary", "\n".join(transfer_lines))
                parts.extend(["", render_table(transfer_plan, TRANSFER_OUTPUT_COLUMNS, 30)])

    budget_frame = build_budget_summary(current, suggested, budget)
    append_section(parts, "Budget Summary", render_table(budget_frame, ["scope", "current_total", "suggested_total", "budget", "remaining"], 5))

    team_counts = build_team_counts(current, suggested, max_per_team)
    append_section(parts, "Team Counts", render_table(team_counts, ["team", "current_count", "suggested_count", "max_per_team"], 30))

    warnings = plan_warnings[:]
    if current is not None and not current.empty:
        unmatched = current.loc[current["match_method"].eq("unmatched")]
        if not unmatched.empty:
            warnings.append("Unmatched current squad players: " + ", ".join(unmatched["player"].astype(str).tolist()))

        injury_rows = current.loc[pd.to_numeric(current["bd_injury_penalty"], errors="coerce").fillna(0) > 0]
        if not injury_rows.empty:
            warnings.append("Players with injury penalties: " + ", ".join(injury_rows["player"].astype(str).tolist()))

        team_issues = current.groupby("team").size()
        team_issues = team_issues[team_issues > max_per_team]
        if not team_issues.empty:
            warnings.append("Current team limit issues: " + ", ".join(f"{team} ({count})" for team, count in team_issues.items()))

        current_total = pd.to_numeric(current["price"], errors="coerce").fillna(0).sum() if "price" in current.columns else 0
        if pd.notna(current_total) and current_total > budget:
            warnings.append(f"Current squad is over budget by {current_total - budget:.2f}.")

    if suggested is not None and not suggested.empty:
        injured = suggested.loc[suggested.apply(is_active_injury, axis=1)]
        if not injured.empty:
            warnings.append("Suggested squad still includes injured players: " + ", ".join(injured["player"].astype(str).tolist()))

    append_section(parts, "Risk Warnings", "\n".join(f"- {line}" for line in warnings) if warnings else "_No additional warnings._")

    return "\n".join(part for part in parts if part is not None and part != "")


def main() -> int:
    args = build_parser().parse_args()

    rankings = build_rankings_view(read_csv(args.rankings_csv))
    current = build_current_squad(read_csv(args.current_squad_csv)) if args.current_squad_csv else None
    aliases = load_player_aliases(args.player_aliases) if args.player_aliases else None
    current_ranked = match_current_to_rankings(current, rankings, aliases) if current is not None else None

    formation = normalize_formation(args.formation)
    plan_warnings: list[str] = []
    suggested = None
    transfer_plan = None
    current_locked = bool(current_ranked is not None and not current_ranked.empty and current_ranked["locked"].map(to_bool).any())
    if args.mode == "plan":
        if args.planning_scope == "wildcard":
            suggested_raw, plan_warnings = select_squad(rankings, formation, args.budget, args.max_per_team, args.allow_injury_risk)
            if not suggested_raw.empty:
                suggested = choose_xi(suggested_raw, formation)
            else:
                suggested = suggested_raw
        else:
            transfer_plan, plan_warnings = build_transfer_plan(
                current_ranked if current_ranked is not None else pd.DataFrame(),
                rankings,
                args.budget,
                args.max_per_team,
                args.free_transfers,
                args.allow_injury_risk,
                args.min_expected_minutes_probability,
            )

    report = render_report(
        args.mode,
        current_ranked,
        suggested,
        transfer_plan,
        args.budget,
        args.max_per_team,
        args.free_transfers,
        current_locked,
        plan_warnings,
        args.allow_injury_risk,
        args.captain_min_expected_minutes_probability,
        args.planning_scope,
        args.min_expected_minutes_probability,
    )

    print(report)

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report + "\n", encoding="utf-8")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "plan" and args.planning_scope == "free_transfers":
            rows_frame = build_transfer_output_rows(transfer_plan)
        else:
            rows_frame = build_output_rows(args.mode, current_ranked, suggested)
        rows_frame.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
