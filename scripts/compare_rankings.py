from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wc_fantasy_model.normalize import normalize_player_name, normalize_team


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two World Cup Fantasy ranking CSV files.")
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--enriched-csv", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=30)
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


def score_column(frame: pd.DataFrame) -> str | None:
    return first_present(frame, ["final_score", "fantasy_shortlist_score"])


def get_series(frame: pd.DataFrame, column: str | None) -> pd.Series | None:
    if column is None or column not in frame.columns:
        return None
    value = frame.loc[:, column]
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return None
        value = value.iloc[:, 0]
    return value


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


def key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    player_col = first_present(out, ["player", "display_name", "name"])
    team_col = first_present(out, ["team", "team_standardized", "country_name"])
    position_col = first_present(out, ["position"])

    if player_col is None:
        raise ValueError("Ranking CSV must contain a player column such as Player or display_name.")

    player_values = get_series(out, player_col)
    team_values = get_series(out, team_col)
    position_values = get_series(out, position_col)

    out["compare_player_key"] = player_values.map(normalize_player_name) if player_values is not None else ""
    out["compare_team_key"] = team_values.map(normalize_team) if team_values is not None else ""
    out["compare_position_key"] = position_values.astype("string").str.upper().fillna("") if position_values is not None else ""
    out["compare_key"] = (
        out["compare_player_key"].fillna("")
        + "||"
        + out["compare_team_key"].fillna("")
        + "||"
        + out["compare_position_key"].fillna("")
    )
    return out


def top_players(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    score_col = score_column(frame)
    if score_col is None:
        raise ValueError("Ranking CSV must contain final_score or fantasy_shortlist_score.")
    work = frame.copy()
    score_values = get_series(work, score_col)
    if score_values is None:
        raise ValueError("Ranking CSV must contain final_score or fantasy_shortlist_score.")
    work["_compare_score"] = pd.to_numeric(score_values, errors="coerce").fillna(0)
    work = work.sort_values("_compare_score", ascending=False)
    display_cols = ["player", "team", "position", score_col, "bd_live_form_adjustment", "bd_injury_penalty"]
    return select_columns(work, display_cols).head(top_n)


def print_table(title: str, frame: pd.DataFrame) -> None:
    print(title)
    if frame.empty:
        print("<no rows>")
        return
    print(frame.to_string(index=False))
    print()


def main() -> int:
    args = build_parser().parse_args()

    baseline = key_frame(read_csv(args.baseline_csv))
    enriched = key_frame(read_csv(args.enriched_csv))

    baseline_score_col = score_column(baseline)
    enriched_score_col = score_column(enriched)
    if baseline_score_col is None:
        raise ValueError("Baseline CSV needs final_score or fantasy_shortlist_score.")
    if enriched_score_col is None:
        raise ValueError("Enriched CSV needs final_score or fantasy_shortlist_score.")

    baseline_scores = get_series(baseline, baseline_score_col)
    enriched_scores = get_series(enriched, enriched_score_col)
    if baseline_scores is None:
        raise ValueError("Baseline CSV needs final_score or fantasy_shortlist_score.")
    if enriched_scores is None:
        raise ValueError("Enriched CSV needs final_score or fantasy_shortlist_score.")

    baseline = baseline.copy()
    enriched = enriched.copy()
    baseline["_compare_score"] = pd.to_numeric(baseline_scores, errors="coerce").fillna(0)
    enriched["_compare_score"] = pd.to_numeric(enriched_scores, errors="coerce").fillna(0)

    print_table("Top baseline players", top_players(baseline, args.top_n))
    print_table("Top enriched players", top_players(enriched, args.top_n))

    baseline_top = baseline.sort_values("_compare_score", ascending=False).head(args.top_n)
    enriched_top = enriched.sort_values("_compare_score", ascending=False).head(args.top_n)

    baseline_keys = set(baseline_top["compare_key"].dropna().astype(str))
    enriched_keys = set(enriched_top["compare_key"].dropna().astype(str))

    entering = enriched_top.loc[~enriched_top["compare_key"].astype(str).isin(baseline_keys)]
    leaving = baseline_top.loc[~baseline_top["compare_key"].astype(str).isin(enriched_keys)]

    print_table("Players entering top N", select_columns(entering, ["player", "team", "position", enriched_score_col]))
    print_table("Players leaving top N", select_columns(leaving, ["player", "team", "position", baseline_score_col]))

    baseline_movement = select_columns(baseline, ["compare_key", "player", "team", "position", baseline_score_col]).rename(
        columns={
            "player": "player_baseline",
            "team": "team_baseline",
            "position": "position_baseline",
            baseline_score_col: "baseline_score",
        }
    )
    enriched_movement = select_columns(
        enriched,
        ["compare_key", enriched_score_col, "bd_live_form_adjustment", "bd_injury_penalty"],
    ).rename(columns={enriched_score_col: "enriched_score"})
    movement = baseline_movement.merge(
        enriched_movement,
        on="compare_key",
        how="inner",
    )
    movement["score_movement"] = movement["enriched_score"] - movement["baseline_score"]
    movement = movement.sort_values("score_movement", ascending=False)

    movement_cols = ["player_baseline", "team_baseline", "position_baseline", "baseline_score", "enriched_score", "score_movement", "bd_live_form_adjustment", "bd_injury_penalty"]
    movement_display = select_columns(movement, movement_cols)
    print_table("Largest positive final_score movement", movement_display.head(args.top_n))
    print_table("Largest negative final_score movement", movement_display.sort_values("score_movement", ascending=True).head(args.top_n))

    if get_series(enriched, "bd_live_form_adjustment") is not None or get_series(enriched, "bd_injury_penalty") is not None:
        print("BALLDONTLIE diagnostics available in enriched CSV.")
        diag_score = get_series(enriched, "bd_live_form_adjustment")
        if diag_score is not None:
            diag_frame = enriched.copy()
            diag_frame["_compare_diag_score"] = pd.to_numeric(diag_score, errors="coerce").fillna(0)
            diag_frame = diag_frame.sort_values("_compare_diag_score", ascending=False)
        else:
            diag_frame = enriched
        print_table("Enriched BALLDONTLIE diagnostics", select_columns(diag_frame, ["player", "team", "position", "bd_live_form_adjustment", "bd_injury_penalty"]).head(args.top_n))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
