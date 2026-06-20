from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wc_fantasy_model.squad_matching import (
    build_current_squad,
    build_rankings_view,
    current_squad_summary,
    get_series,
    load_player_aliases,
    match_current_to_rankings,
    read_csv,
    selection_metric,
)


DEFAULT_RANKINGS_CSV = Path("output/player_rankings_all.csv")
DEFAULT_CURRENT_SQUAD_CSV = Path("data/manual/current_squad.csv")
DEFAULT_PLAYER_ALIASES = Path("data/manual/player_aliases.csv")
DEFAULT_FIXTURE_CSV = Path("data/raw/world-cup_2026.csv")
DEFAULT_OUTPUT_HTML = Path("outputs/current_squad_dashboard.html")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a local HTML dashboard for the current fantasy squad.")
    parser.add_argument("--rankings-csv", type=Path, default=DEFAULT_RANKINGS_CSV)
    parser.add_argument("--current-squad-csv", type=Path, default=DEFAULT_CURRENT_SQUAD_CSV)
    parser.add_argument("--player-aliases", type=Path, default=DEFAULT_PLAYER_ALIASES)
    parser.add_argument("--fixture-csv", type=Path, default=DEFAULT_FIXTURE_CSV)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument(
        "--captain-min-expected-minutes-probability",
        type=float,
        default=0.75,
        help="Minimum expected minutes probability for safe captain candidates.",
    )
    parser.add_argument(
        "--low-minutes-threshold",
        type=float,
        default=0.6,
        help="Threshold used to flag players with low expected minutes.",
    )
    return parser


def to_number(value: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else 0.0


def esc(value: Any) -> str:
    if isinstance(value, pd.Series):
        value = value.dropna().iloc[0] if not value.dropna().empty else ""
    if pd.isna(value):
        value = ""
    return html.escape(str(value))


def format_number(value: Any, digits: int = 2) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return ""
    return f"{float(number):.{digits}f}"


def format_optional_number(value: Any, digits: int = 1) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "N/A"
    return f"{float(number):.{digits}f}"


def file_timestamp(path: Path | None) -> str:
    if path is None or not path.exists():
        return "N/A"
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def sum_numeric_series(frame: pd.DataFrame, column: str) -> float | None:
    series = get_series(frame, column)
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.sum())


def format_optional_integer(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "N/A"
    return str(int(number))


def text_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((candidate for candidate in candidates if candidate in frame.columns), None)


def clean_text_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series("", index=frame.index)
    series = get_series(frame, column)
    if series is None:
        return pd.Series("", index=frame.index)
    return series.map(lambda value: "" if pd.isna(value) else str(value).strip().lower())


def player_title(row: pd.Series) -> str:
    player = str(row.get("player", "")).strip()
    matched = str(row.get("matched_player", "")).strip()
    if matched and matched != player:
        return f"<div class='player-name'>{esc(player)}</div><div class='player-subtitle'>Matched: {esc(matched)}</div>"
    return f"<div class='player-name'>{esc(player)}</div>"


def badge(label: str, kind: str) -> str:
    return f"<span class='badge {kind}'>{esc(label)}</span>"


def row_badges(row: pd.Series, low_minutes_threshold: float, captain_keys: set[str], vice_keys: set[str]) -> str:
    badges: list[str] = []
    minutes = pd.to_numeric(row.get("expected_minutes_probability"), errors="coerce")
    injury_penalty = pd.to_numeric(row.get("bd_injury_penalty"), errors="coerce")
    fixture_difficulty = pd.to_numeric(row.get("fixture_difficulty"), errors="coerce")
    match_quality = str(row.get("match_quality", "")).strip().lower()

    if pd.notna(minutes) and pd.notna(injury_penalty) and minutes >= low_minutes_threshold and injury_penalty <= 0 and match_quality in {"exact", "alias"}:
        badges.append(badge("SAFE", "safe"))
    if pd.isna(minutes) or minutes < low_minutes_threshold:
        badges.append(badge("LOW MINUTES", "warning"))
    if pd.notna(injury_penalty) and injury_penalty > 0:
        badges.append(badge("INJURY RISK", "danger"))
    if pd.notna(fixture_difficulty) and fixture_difficulty >= 4:
        badges.append(badge("HARD FIXTURE", "neutral"))
    match_key = str(row.get("match_key", ""))
    if match_key in captain_keys:
        badges.append(badge("CAPTAIN", "captain"))
    if match_key in vice_keys:
        badges.append(badge("VICE CAPTAIN", "vice"))
    return "".join(badges)


def render_table(
    frame: pd.DataFrame,
    columns: list[str],
    low_minutes_threshold: float,
    captain_keys: set[str],
    vice_keys: set[str],
    bench: bool = False,
    show_tags: bool = True,
    header_labels: dict[str, str] | None = None,
) -> str:
    if frame is None or frame.empty:
        return "<div class='table-wrap'><div class='empty'>No rows available.</div></div>"

    headers = columns + (["tags"] if show_tags else [])
    lines = ["<div class='table-wrap'>", "<table class='dashboard-table'>", "<thead><tr>"]
    for header in headers:
        label = header_labels.get(header, header.replace('_', ' ').title()) if header_labels else header.replace('_', ' ').title()
        lines.append(f"<th>{esc(label)}</th>")
    lines.extend(["</tr></thead>", "<tbody>"])

    for _, row in frame.iterrows():
        classes: list[str] = []
        minutes = pd.to_numeric(row.get("expected_minutes_probability"), errors="coerce")
        injury_penalty = pd.to_numeric(row.get("bd_injury_penalty"), errors="coerce")
        fixture_difficulty = pd.to_numeric(row.get("fixture_difficulty"), errors="coerce")
        match_quality = str(row.get("match_quality", "")).strip().lower()
        if pd.isna(minutes) or minutes < low_minutes_threshold:
            classes.append("low-minutes")
        if pd.notna(injury_penalty) and injury_penalty > 0:
            classes.append("injury-risk")
        if pd.notna(fixture_difficulty) and fixture_difficulty >= 4:
            classes.append("hard-fixture")
        if match_quality in {"unmatched", "ambiguous", "surname", "fuzzy"}:
            classes.append("unmatched")

        lines.append(f"<tr class='{' '.join(classes)}'>")
        for column in columns:
            value = row.get(column, "")
            if column == "player":
                cell = player_title(row)
            elif column in {"price", "expected_fantasy_points", "expected_minutes_probability"}:
                cell = format_number(value)
            elif column == "bench_order" and bench:
                cell = esc(value)
            else:
                cell = esc(value)
            td_class = "num" if column in {"price", "expected_fantasy_points", "expected_minutes_probability", "bench_order"} else "wrap"
            if column == "player":
                td_class = "wrap player-cell"
            elif column == "risk_note":
                td_class = "wrap risk-note"
            lines.append(f"<td class='{td_class}'>{cell}</td>")
        if show_tags:
            lines.append(f"<td class='tags'>{row_badges(row, low_minutes_threshold, captain_keys, vice_keys)}</td>")
        lines.append("</tr>")

    lines.extend(["</tbody>", "</table>", "</div>"])
    return "\n".join(lines)


def render_list(items: list[str]) -> str:
    if not items:
        return "<li class='empty-item'>None</li>"
    return "\n".join(f"<li>{item}</li>" for item in items)


def short_player_label(row: pd.Series, fallback: str = "") -> str:
    player = str(row.get("player", "")).strip()
    team = str(row.get("team", "")).strip()
    points = format_number(row.get("expected_fantasy_points"), 1)
    if not player:
        player = fallback
    pieces = [player]
    if team:
        pieces.append(team)
    if points:
        pieces.append(points)
    return " - ".join(piece for piece in pieces if piece)


def safe_captain_pool(xi: pd.DataFrame, captain_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    minutes = pd.to_numeric(get_series(xi, "expected_minutes_probability"), errors="coerce")
    injury = pd.to_numeric(get_series(xi, "bd_injury_penalty"), errors="coerce").fillna(0)
    safe = xi.loc[minutes.fillna(0).ge(captain_threshold) & injury.le(0)].copy()
    risky = xi.loc[~xi.index.isin(safe.index)].copy()
    safe = safe.sort_values(
        ["expected_fantasy_points", "expected_minutes_probability", "selection_score", "selection_final_score"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    risky = risky.sort_values(
        ["expected_fantasy_points", "expected_minutes_probability", "selection_score", "selection_final_score"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    return safe, risky


def format_alert_items(current: pd.DataFrame, low_minutes_threshold: float, captain_threshold: float) -> dict[str, list[str]]:
    alerts: dict[str, list[str]] = {
        "xi_low_minutes": [],
        "bench_low_minutes": [],
        "injuries": [],
        "captain": [],
        "unmatched": [],
    }

    xi = current.loc[current["squad_role"].eq("XI")].copy()
    bench = current.loc[current["squad_role"].eq("BENCH")].copy()

    xi_low = xi.loc[pd.to_numeric(get_series(xi, "expected_minutes_probability"), errors="coerce").fillna(1).lt(low_minutes_threshold)]
    bench_low = bench.loc[pd.to_numeric(get_series(bench, "expected_minutes_probability"), errors="coerce").fillna(1).lt(low_minutes_threshold)]
    if not xi_low.empty:
        alerts["xi_low_minutes"] = [f"{row['player']} ({format_number(row.get('expected_minutes_probability'))})" for _, row in xi_low.iterrows()]
    if not bench_low.empty:
        alerts["bench_low_minutes"] = [f"{row['player']} ({format_number(row.get('expected_minutes_probability'))})" for _, row in bench_low.iterrows()]

    injured = current.loc[pd.to_numeric(get_series(current, "bd_injury_penalty"), errors="coerce").fillna(0) > 0]
    if not injured.empty:
        alerts["injuries"] = [f"{row['player']} ({row.get('bd_injury_penalty')})" for _, row in injured.iterrows()]

    captain = current.loc[current["is_captain"]].copy()
    vice = current.loc[current["is_vice_captain"]].copy()
    safe_pool, _ = safe_captain_pool(xi, captain_threshold)
    safe_keys = safe_pool.head(3)["match_key"].astype(str).tolist()
    if not captain.empty:
        captain_row = captain.iloc[0]
        captain_minutes = pd.to_numeric(captain_row.get("expected_minutes_probability"), errors="coerce")
        if pd.notna(captain_minutes) and captain_minutes < captain_threshold:
            alerts["captain"].append("Current captain is below the captain minutes threshold.")
        if str(captain_row.get("match_key", "")) not in safe_keys:
            alerts["captain"].append("Current captain is not among the top 3 safe captain candidates.")
    if not vice.empty:
        vice_row = vice.iloc[0]
        vice_minutes = pd.to_numeric(vice_row.get("expected_minutes_probability"), errors="coerce")
        if pd.notna(vice_minutes) and vice_minutes < captain_threshold:
            alerts["captain"].append("Current vice captain is below the captain minutes threshold.")

    unmatched = current.loc[current["match_quality"].astype(str).str.lower().isin({"unmatched", "ambiguous", "surname", "fuzzy"})]
    if not unmatched.empty:
        alerts["unmatched"] = [f"{row['player']} ({row.get('match_quality', '')})" for _, row in unmatched.iterrows()]

    return alerts


def build_takeaways_section(
    current: pd.DataFrame,
    xi: pd.DataFrame,
    bench: pd.DataFrame,
    safe_candidates: pd.DataFrame,
    low_minutes_threshold: float,
    captain_threshold: float,
) -> str:
    captain = current.loc[current["is_captain"]].copy()
    vice = current.loc[current["is_vice_captain"]].copy()
    safe_captain = safe_candidates.head(1)
    best_bench = bench.sort_values(
        ["expected_fantasy_points", "expected_minutes_probability", "selection_score", "selection_final_score"],
        ascending=[False, False, False, False],
        na_position="last",
    ).head(3)
    xi_minutes = pd.to_numeric(get_series(xi, "expected_minutes_probability"), errors="coerce").fillna(0)
    low_xi = xi.loc[xi_minutes.lt(low_minutes_threshold)]
    injury_xi = xi.loc[pd.to_numeric(get_series(xi, "bd_injury_penalty"), errors="coerce").fillna(0) > 0]
    risk_rank = xi.copy()
    risk_rank["risk_score"] = (
        (pd.to_numeric(get_series(risk_rank, "bd_injury_penalty"), errors="coerce").fillna(0) > 0).astype(int) * 3
        + (pd.to_numeric(get_series(risk_rank, "expected_minutes_probability"), errors="coerce").fillna(1) < low_minutes_threshold).astype(int) * 2
        + pd.to_numeric(get_series(risk_rank, "fixture_difficulty"), errors="coerce").fillna(0)
    )
    worst_xi = risk_rank.sort_values(
        ["risk_score", "expected_fantasy_points", "expected_minutes_probability"],
        ascending=[False, True, True],
        na_position="last",
    ).head(3)

    captain_label = "No captain assigned"
    if not captain.empty:
        captain_row = captain.iloc[0]
        captain_label = short_player_label(captain_row)
    vice_label = "No vice captain assigned"
    if not vice.empty:
        vice_label = short_player_label(vice.iloc[0])
    safe_label = "No safe captain candidates"
    if not safe_captain.empty:
        safe_label = short_player_label(safe_captain.iloc[0])

    captain_minutes = pd.to_numeric(get_series(captain, "expected_minutes_probability"), errors="coerce")
    captain_status = captain_label if captain.empty else f"{captain_label} ({format_optional_number(captain_minutes.iloc[0] if not captain_minutes.empty else pd.NA, 2)})"
    vice_minutes = pd.to_numeric(get_series(vice, "expected_minutes_probability"), errors="coerce")
    vice_status = vice_label if vice.empty else f"{vice_label} ({format_optional_number(vice_minutes.iloc[0] if not vice_minutes.empty else pd.NA, 2)})"
    body = "\n".join(
        [
            "<ul class='takeaway-list'>",
            f"<li><strong>Best safe captain:</strong> {esc(safe_label)}</li>",
            f"<li><strong>Current captain:</strong> {esc(captain_status)}</li>",
            f"<li><strong>Current vice captain:</strong> {esc(vice_status)}</li>",
            f"<li><strong>Low-minutes XI players:</strong> {esc(', '.join(low_xi['player'].astype(str).tolist()) if not low_xi.empty else 'None')}</li>",
            f"<li><strong>Injury-risk players:</strong> {esc(', '.join(injury_xi['player'].astype(str).tolist()) if not injury_xi.empty else 'None')}</li>",
            f"<li><strong>Best bench options:</strong> {esc(', '.join(best_bench['player'].astype(str).tolist()) if not best_bench.empty else 'None')}</li>",
            f"<li><strong>Worst current XI risk:</strong> {esc(', '.join(worst_xi['player'].astype(str).tolist()) if not worst_xi.empty else 'None')}</li>",
            "</ul>",
        ]
    )
    return render_section("Immediate Squad Takeaways", body)


def build_transfer_watchlist(
    current: pd.DataFrame,
    rankings: pd.DataFrame,
    low_minutes_threshold: float,
) -> str:
    if current is None or current.empty:
        return render_section("Transfer Watchlist", "<div class='empty'>No current squad available.</div>")

    current = selection_metric(current)
    rankings = selection_metric(rankings)
    current_keys = set(current["match_key"].astype(str)) if "match_key" in current.columns else set()

    sell_frame = current.copy()
    sell_frame["watch_minutes"] = pd.to_numeric(get_series(sell_frame, "expected_minutes_probability"), errors="coerce").fillna(0)
    sell_frame["watch_injury"] = pd.to_numeric(get_series(sell_frame, "bd_injury_penalty"), errors="coerce").fillna(0)
    sell_frame["watch_points"] = pd.to_numeric(get_series(sell_frame, "expected_fantasy_points"), errors="coerce").fillna(0)
    sell_frame["watch_fixture"] = pd.to_numeric(get_series(sell_frame, "fixture_difficulty"), errors="coerce").fillna(0)
    sell_frame["watch_score"] = (
        (sell_frame["squad_role"].eq("BENCH").astype(int) * 2)
        + (sell_frame["watch_minutes"].lt(low_minutes_threshold).astype(int) * 3)
        + (sell_frame["watch_injury"].gt(0).astype(int) * 4)
        + sell_frame["watch_fixture"].fillna(0)
        - sell_frame["watch_points"].fillna(0) / 10.0
    )
    sell_frame = sell_frame.sort_values(
        ["watch_score", "watch_minutes", "watch_points", "price"],
        ascending=[False, True, True, True],
        na_position="last",
    )
    sell_table = sell_frame.loc[:, [
        "player",
        "team",
        "position",
        "price",
        "expected_fantasy_points",
        "expected_minutes_probability",
        "fixture_difficulty",
        "bd_injury_penalty",
        "risk_note",
    ]].head(5).copy()

    sell_positions = sell_table["position"].astype(str).str.upper().tolist()
    buy_pool = rankings.loc[~rankings["match_key"].astype(str).isin(current_keys)].copy()
    same_position_pool = buy_pool.loc[buy_pool["position"].astype(str).str.upper().isin(sell_positions)].copy()
    if not same_position_pool.empty:
        buy_pool = same_position_pool
    buy_pool["buy_minutes"] = pd.to_numeric(get_series(buy_pool, "expected_minutes_probability"), errors="coerce").fillna(0)
    buy_pool["buy_injury"] = pd.to_numeric(get_series(buy_pool, "bd_injury_penalty"), errors="coerce").fillna(0)
    buy_pool["buy_points"] = pd.to_numeric(get_series(buy_pool, "expected_fantasy_points"), errors="coerce").fillna(0)
    buy_pool["buy_price"] = pd.to_numeric(get_series(buy_pool, "price"), errors="coerce")
    buy_pool = buy_pool.loc[buy_pool["buy_minutes"].ge(low_minutes_threshold)]
    buy_pool = buy_pool.sort_values(
        ["buy_points", "buy_minutes", "final_score", "buy_price"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    buy_table = buy_pool.loc[:, [
        "player",
        "team",
        "position",
        "price",
        "expected_fantasy_points",
        "expected_minutes_probability",
        "fixture_difficulty",
        "bd_injury_penalty",
        "risk_note",
    ]].head(8).copy()

    if sell_table.empty:
        sell_html = "<div class='empty'>No obvious sell candidates found.</div>"
    else:
        sell_html = render_table(
            sell_table,
            list(sell_table.columns),
            low_minutes_threshold,
            set(),
            set(),
            show_tags=False,
            header_labels={
                "expected_fantasy_points": "exp_pts",
                "expected_minutes_probability": "min_prob",
                "risk_note": "risk",
            },
        )
    if buy_table.empty:
        buy_html = "<div class='empty'>No obvious buy candidates found.</div>"
    else:
        buy_html = render_table(
            buy_table,
            list(buy_table.columns),
            low_minutes_threshold,
            set(),
            set(),
            show_tags=False,
            header_labels={
                "expected_fantasy_points": "exp_pts",
                "expected_minutes_probability": "min_prob",
                "risk_note": "risk",
            },
        )

    note = (
        "This watchlist is a directional shortlist, not a final transfer plan. "
        "When planning transfers, prefer same-position swaps and keep an eye on budget and max-per-team limits."
    )
    body = "\n".join(
        [
            f"<p>{esc(note)}</p>",
            "<h3>Sell Candidates</h3>",
            sell_html,
            "<h3>Buy Candidates</h3>",
            buy_html,
        ]
    )
    return render_section("Transfer Watchlist", body)


def build_planning_hint_section() -> str:
    plan_md = Path("outputs/transfer_plan.md")
    plan_csv = Path("output/transfer_plan.csv")
    lines = [
        "Transfer planning should be generated when the round is open.",
        "Suggested command:",
        "python scripts/plan_transfers.py --rankings-csv output/player_rankings_all.csv --current-squad-csv data/manual/current_squad.csv --player-aliases data/manual/player_aliases.csv --max-transfers 10 --output-md outputs/transfer_plan.md --output-csv output/transfer_plan.csv",
    ]
    if plan_md.exists() or plan_csv.exists():
        lines.append("")
        lines.append("Existing outputs detected:")
        if plan_md.exists():
            lines.append(f"- {plan_md} (last modified: {file_timestamp(plan_md)})")
        if plan_csv.exists():
            lines.append(f"- {plan_csv} (last modified: {file_timestamp(plan_csv)})")
    return render_section("Transfer Planning", "\n".join(f"<p>{esc(line)}</p>" for line in lines))


def render_section(title: str, body: str) -> str:
    return f"<section class='panel'><h2>{esc(title)}</h2>{body}</section>"


def parse_fixture_matches(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame is None or frame.empty:
        return pd.DataFrame(), {"completed_count": 0, "latest_completed": "N/A"}

    work = frame.copy()
    date_col = text_column(work, ["date", "kickoff_date", "match_date", "datetime", "played_at"])
    time_col = text_column(work, ["time", "kickoff_time", "utc_time", "match_time"])
    matchday_col = text_column(work, ["matchday", "round", "gameweek", "week"])
    home_col = text_column(work, ["home_team", "home", "team_home", "team1"])
    away_col = text_column(work, ["away_team", "away", "team_away", "team2"])
    status_col = text_column(work, ["status", "match_status", "state", "phase"])
    result_col = text_column(work, ["result", "score", "full_time_score", "final_score", "result_text"])

    if home_col is None or away_col is None:
        return pd.DataFrame(), {"completed_count": 0, "latest_completed": "N/A"}

    raw_date = get_series(work, date_col) if date_col else pd.Series("", index=work.index)
    raw_time = get_series(work, time_col) if time_col else pd.Series("", index=work.index)
    combined = raw_date.astype(str).fillna("") + (" " + raw_time.astype(str).fillna("") if time_col else "")
    sort_dt = pd.to_datetime(combined.str.strip(), errors="coerce")
    if sort_dt.isna().all() and date_col:
        sort_dt = pd.to_datetime(raw_date, errors="coerce")
    work["_sort_dt"] = sort_dt

    status_text = clean_text_series(work, status_col)
    result_text = clean_text_series(work, result_col)

    completed_mask = status_text.isin({"played", "complete", "completed", "finished", "full time", "ft"}) | result_text.ne("")

    def display_date(series: pd.Series) -> pd.Series:
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed.dt.strftime("%Y-%m-%d").fillna(series.astype(str))

    def display_time(series: pd.Series) -> pd.Series:
        return series.astype(str).replace({"nan": "", "NaT": ""})

    def build_display(frame_slice: pd.DataFrame) -> pd.DataFrame:
        if frame_slice.empty:
            return pd.DataFrame(columns=["date", "time", "matchday", "home_team", "away_team", "status", "result"])
        out = pd.DataFrame(index=frame_slice.index)
        out["date"] = display_date(get_series(frame_slice, date_col)) if date_col else ""
        out["time"] = display_time(get_series(frame_slice, time_col)) if time_col else ""
        out["matchday"] = get_series(frame_slice, matchday_col) if matchday_col else ""
        out["home_team"] = get_series(frame_slice, home_col)
        out["away_team"] = get_series(frame_slice, away_col)
        out["status"] = get_series(frame_slice, status_col) if status_col else ""
        if result_col:
            out["result"] = get_series(frame_slice, result_col)
        else:
            out["result"] = ""
        return out.reset_index(drop=True)

    completed_source = work.loc[completed_mask].copy()
    completed_source = completed_source.sort_values("_sort_dt", ascending=False)
    completed = build_display(completed_source.head(10))
    latest_completed = "N/A"
    if not completed_source.empty:
        latest_row = completed_source.iloc[0]
        latest_date = str(latest_row.get(date_col, "")).strip() if date_col else ""
        latest_time = str(latest_row.get(time_col, "")).strip() if time_col else ""
        latest_completed = " ".join(part for part in [latest_date, latest_time] if part).strip() or "N/A"
    meta = {"completed_count": int(len(completed_source)), "latest_completed": latest_completed}
    return completed, meta


def build_freshness_section(
    rankings_csv: Path,
    current_squad_csv: Path,
    player_aliases_csv: Path | None,
    fixture_csv: Path | None,
    generated: str,
    completed_matches: pd.DataFrame,
    fixture_meta: dict[str, Any],
) -> str:
    path_rows = [
        ("Rankings CSV", str(rankings_csv), file_timestamp(rankings_csv)),
        ("Current Squad CSV", str(current_squad_csv), file_timestamp(current_squad_csv)),
        ("Player Aliases CSV", str(player_aliases_csv) if player_aliases_csv else "N/A", file_timestamp(player_aliases_csv) if player_aliases_csv else "N/A"),
        ("Fixture CSV", str(fixture_csv) if fixture_csv else "N/A", file_timestamp(fixture_csv) if fixture_csv else "N/A"),
        ("Generated", generated, generated),
        ("Completed Match Count", str(fixture_meta.get("completed_count", "N/A"))),
        ("Latest Completed Fixture", str(fixture_meta.get("latest_completed", "N/A"))),
    ]
    paths_html = render_table(
        pd.DataFrame(path_rows, columns=["source", "path", "timestamp"]),
        ["source", "path", "timestamp"],
        0.6,
        set(),
        set(),
        show_tags=False,
    )

    completed_html = render_table(completed_matches, ["date", "time", "matchday", "home_team", "away_team", "status", "result"], 0.6, set(), set(), show_tags=False)

    command = (
        "python scripts/render_squad_report.py "
        f"--rankings-csv {rankings_csv} "
        f"--current-squad-csv {current_squad_csv} "
    )
    if player_aliases_csv:
        command += f"--player-aliases {player_aliases_csv} "
    if fixture_csv:
        command += f"--fixture-csv {fixture_csv} "
    command += "--output-html outputs/current_squad_dashboard.html"

    note = (
        "This is a static local dashboard. To refresh match results and projections, rebuild fixtures/rankings/features, then rerun this renderer."
    )

    body = "\n".join(
        [
            f"<p>{esc(note)}</p>",
            f"<div class='refresh-note'><strong>Regenerate:</strong><pre>{esc(command)}</pre></div>",
            "<h3>File Snapshot</h3>",
            paths_html,
            "<h3>Recent Matches</h3>",
            "<div class='recent-block'><h4>Last 10 completed/played matches</h4>"
            + (
                completed_html
                if not completed_matches.empty
                else "<div class='empty'>No completed matches found.</div>"
            )
            + "</div>",
        ]
    )
    return render_section("Data Freshness & Recent Results", body)


def build_html(
    current: pd.DataFrame,
    rankings: pd.DataFrame,
    rankings_csv: Path,
    current_squad_csv: Path,
    player_aliases_csv: Path | None,
    fixture_csv: Path | None,
    completed_matches: pd.DataFrame,
    fixture_meta: dict[str, Any],
    low_minutes_threshold: float,
    captain_threshold: float,
) -> str:
    current = selection_metric(current)
    xi, bench = current_squad_summary(current)
    safe_candidates, risky_candidates = safe_captain_pool(xi, captain_threshold)
    captain = current.loc[current["is_captain"]].copy()
    vice = current.loc[current["is_vice_captain"]].copy()

    matched_current = current.loc[current["match_quality"].astype(str).str.lower().ne("unmatched")].copy() if "match_quality" in current.columns else current
    total_value = sum_numeric_series(matched_current, "price")
    xi_points = sum_numeric_series(xi, "expected_fantasy_points")
    bench_points = sum_numeric_series(bench, "expected_fantasy_points")
    risk_count = int(current["risk_note"].astype("string").str.strip().ne("").sum()) if "risk_note" in current.columns else None
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    captain_keys = set(captain["match_key"].astype(str)) if not captain.empty else set()
    vice_keys = set(vice["match_key"].astype(str)) if not vice.empty else set()
    alerts = format_alert_items(current, low_minutes_threshold, captain_threshold)

    alert_html = f"""
    <h3>XI players below low minutes threshold</h3>
    <ul>{render_list(alerts['xi_low_minutes'])}</ul>
    <h3>Bench players below low minutes threshold</h3>
    <ul>{render_list(alerts['bench_low_minutes'])}</ul>
    <h3>Injury penalty players</h3>
    <ul>{render_list(alerts['injuries'])}</ul>
    <h3>Unmatched / ambiguous players</h3>
    <ul>{render_list(alerts['unmatched'])}</ul>
    <h3>Captain warning</h3>
    <ul>{render_list(alerts['captain'])}</ul>
    """

    xi_columns = [
        "player",
        "team",
        "position",
        "price",
        "expected_fantasy_points",
        "expected_minutes_probability",
        "opponent",
        "risk_note",
    ]
    bench_columns = xi_columns + ["bench_order"]
    captain_columns = [
        "player",
        "team",
        "position",
        "expected_fantasy_points",
        "expected_minutes_probability",
        "risk_note",
    ]
    takeaways_html = build_takeaways_section(current, xi, bench, safe_candidates, low_minutes_threshold, captain_threshold)
    watchlist_html = build_transfer_watchlist(current, rankings, low_minutes_threshold)
    planning_hint_html = build_planning_hint_section()

    summary_bits = []
    if total_value is not None:
        summary_bits.append(f"Squad value: {format_optional_number(total_value, 1)}")
    if xi_points is not None:
        summary_bits.append(f"XI expected points: {format_optional_number(xi_points, 1)}")
    if bench_points is not None:
        summary_bits.append(f"Bench expected points: {format_optional_number(bench_points, 1)}")
    if risk_count is not None:
        summary_bits.append(f"Squad risk count: {format_optional_integer(risk_count)}")
    summary_line = " | ".join(summary_bits)
    summary_html = f"<div class='summary-line'>{esc(summary_line)}</div>" if summary_bits else ""

    table_labels = {
        "expected_fantasy_points": "exp_pts",
        "expected_minutes_probability": "min_prob",
        "risk_note": "risk",
        "bench_order": "bench_order",
    }
    current_captain_html = render_table(captain, captain_columns, low_minutes_threshold, captain_keys, vice_keys, header_labels=table_labels)
    current_vice_html = render_table(vice, captain_columns, low_minutes_threshold, captain_keys, vice_keys, header_labels=table_labels)
    safe_html = render_table(safe_candidates.head(5), captain_columns, low_minutes_threshold, captain_keys, vice_keys, header_labels=table_labels)
    risky_html = render_table(risky_candidates.head(10), captain_columns, low_minutes_threshold, captain_keys, vice_keys, header_labels=table_labels)

    xi_html = render_table(xi, xi_columns, low_minutes_threshold, captain_keys, vice_keys, header_labels=table_labels)
    bench_html = render_table(bench, bench_columns, low_minutes_threshold, captain_keys, vice_keys, bench=True, header_labels=table_labels)
    captain_panel_html = render_section(
        "Captain Panel",
        "\n".join(
            [
                "<h3>Current Captain</h3>",
                current_captain_html,
                "<h3>Current Vice Captain</h3>",
                current_vice_html,
                "<h3>Top 5 Safe Captain Candidates</h3>",
                safe_html,
                "<h3>High Projection but Risky Captain Options</h3>",
                risky_html,
            ]
        ),
    )
    squad_summary_html = render_section(
        "Squad Summary",
        "\n".join(
            [
                "<h3>Starting XI</h3>",
                xi_html,
                "<h3>Bench</h3>",
                bench_html,
            ]
        ),
    )
    freshness_html = build_freshness_section(
        rankings_csv,
        current_squad_csv,
        player_aliases_csv,
        fixture_csv,
        generated,
        completed_matches,
        fixture_meta,
    )

    css = """
    :root {
      --bg: #f3f4f6;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #6b7280;
      --line: #e5e7eb;
      --safe: #dcfce7;
      --safe-ink: #166534;
      --warn: #fef3c7;
      --warn-ink: #92400e;
      --danger: #fee2e2;
      --danger-ink: #991b1b;
      --neutral: #e0e7ff;
      --neutral-ink: #3730a3;
      --captain: #dbeafe;
      --captain-ink: #1d4ed8;
      --vice: #ede9fe;
      --vice-ink: #6d28d9;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #f8fafc 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: "Segoe UI", Arial, sans-serif;
      line-height: 1.45;
    }
    .container { max-width: 1100px; margin: 0 auto; padding: 18px; }
    .hero {
      background: linear-gradient(135deg, #0f172a 0%, #1f2937 50%, #334155 100%);
      color: #fff;
      border-radius: 20px;
      padding: 24px;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
    }
    .hero h1 { margin: 0 0 8px 0; font-size: 2rem; }
    .hero .meta { color: rgba(255,255,255,0.8); font-size: 0.95rem; }
    .summary-line {
      margin-top: 12px;
      color: rgba(255,255,255,0.92);
      font-size: 0.98rem;
      font-weight: 600;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }
    .panel { margin-top: 18px; overflow: hidden; }
    .panel h2 {
      margin: 0;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      font-size: 1.05rem;
      background: linear-gradient(180deg, #fff 0%, #f8fafc 100%);
    }
    .panel .body { padding: 16px 18px; }
    .panel h3 { margin: 16px 0 8px 0; font-size: 0.95rem; }
    .panel p { margin: 0 0 10px 0; }
    .panel ul { margin: 0 0 12px 18px; padding: 0; color: var(--ink); }
    .panel li { margin-bottom: 6px; }
    .empty, .empty-item { color: var(--muted); font-style: italic; }
    .table-wrap {
      max-width: 100%;
      overflow-x: auto;
      overflow-y: hidden;
      margin: 10px 0 14px 0;
    }
    table.dashboard-table { width: 100%; min-width: 760px; border-collapse: collapse; background: #fff; border: 1px solid var(--line); }
    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
      font-size: 0.85rem;
    }
    th { font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); background: #fafafa; position: sticky; top: 0; z-index: 1; }
    td.num, th.num { text-align: right; white-space: nowrap; }
    td.wrap, th.wrap { white-space: nowrap; }
    td.player-cell, td.risk-note { white-space: normal; min-width: 160px; }
    td.risk-note { min-width: 180px; }
    tbody tr:hover { background: #f8fafc; }
    tr.low-minutes { background: #fffaf0; }
    tr.injury-risk { background: #fef2f2; }
    tr.hard-fixture { background: #f8fafc; }
    tr.unmatched { background: #fefce8; }
    .player-name { font-weight: 700; }
    .player-subtitle { color: var(--muted); font-size: 0.82rem; margin-top: 2px; }
    .badge {
      display: inline-block;
      margin: 0 6px 6px 0;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 700;
      white-space: nowrap;
    }
    .badge.safe { background: var(--safe); color: var(--safe-ink); }
    .badge.warning { background: var(--warn); color: var(--warn-ink); }
    .badge.danger { background: var(--danger); color: var(--danger-ink); }
    .badge.neutral { background: var(--neutral); color: var(--neutral-ink); }
    .badge.captain { background: var(--captain); color: var(--captain-ink); }
    .badge.vice { background: var(--vice); color: var(--vice-ink); }
    .takeaway-list { margin: 0; padding-left: 18px; }
    .takeaway-list li { margin-bottom: 8px; }
    .refresh-note {
      margin-top: 12px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #f8fafc;
    }
    .refresh-note pre {
      margin: 10px 0 0 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.82rem;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
    }
    .recent-block { margin-top: 14px; }
    .recent-block h4 { margin: 0 0 10px 0; font-size: 0.95rem; }
    @media (max-width: 700px) {
      .container { padding: 12px; }
      .hero { padding: 18px; border-radius: 16px; }
      .hero h1 { font-size: 1.55rem; }
      table.dashboard-table { min-width: 620px; }
    }
    """

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Current Squad Dashboard</title>
      <style>{css}</style>
    </head>
    <body>
      <div class="container">
        <div class="hero">
          <h1>Current Squad Dashboard</h1>
          <div class="meta">Generated {esc(generated)}</div>
          {summary_html}
        </div>

        {render_section("Key Alerts", alert_html)}
        {takeaways_html}
        {captain_panel_html}
        {squad_summary_html}
        {watchlist_html}
        {freshness_html}
        {planning_hint_html}
      </div>
    </body>
    </html>
    """


def main() -> int:
    args = build_parser().parse_args()

    rankings = build_rankings_view(read_csv(args.rankings_csv), low_minutes_threshold=args.low_minutes_threshold)
    current = build_current_squad(read_csv(args.current_squad_csv))
    aliases = load_player_aliases(args.player_aliases)
    if args.player_aliases and not args.player_aliases.exists():
        print(f"Warning: player alias file not found at {args.player_aliases}; continuing without aliases.")
    current = match_current_to_rankings(current, rankings, aliases, low_minutes_threshold=args.low_minutes_threshold)
    current = selection_metric(current)
    fixture_frame = read_csv(args.fixture_csv) if args.fixture_csv and args.fixture_csv.exists() else pd.DataFrame()
    completed_matches, fixture_meta = parse_fixture_matches(fixture_frame)

    html_text = build_html(
        current,
        rankings,
        args.rankings_csv,
        args.current_squad_csv,
        args.player_aliases if args.player_aliases and args.player_aliases.exists() else None,
        args.fixture_csv if args.fixture_csv and args.fixture_csv.exists() else None,
        completed_matches,
        fixture_meta,
        args.low_minutes_threshold,
        args.captain_min_expected_minutes_probability,
    )
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html_text, encoding="utf-8")
    print(args.output_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
