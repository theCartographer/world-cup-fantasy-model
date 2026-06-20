from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pick_xi as px
from wc_fantasy_model.normalize import normalize_player_name, normalize_team
from wc_fantasy_model.squad_matching import (
    build_current_squad,
    build_rankings_view,
    current_squad_summary,
    load_player_aliases,
    match_current_to_rankings,
    read_csv,
    selection_metric,
)


DEFAULT_RANKINGS_CSV = Path("output/player_rankings_all.csv")
DEFAULT_CURRENT_SQUAD_CSV = Path("data/manual/current_squad.csv")
DEFAULT_PLAYER_ALIASES = Path("data/manual/player_aliases.csv")
DEFAULT_OUTPUT_MD = Path("outputs/transfer_plan.md")
DEFAULT_OUTPUT_CSV = Path("output/transfer_plan.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build transfer plans from the matched current squad.")
    parser.add_argument("--rankings-csv", type=Path, default=DEFAULT_RANKINGS_CSV)
    parser.add_argument("--current-squad-csv", type=Path, default=DEFAULT_CURRENT_SQUAD_CSV)
    parser.add_argument("--player-aliases", type=Path, default=DEFAULT_PLAYER_ALIASES)
    parser.add_argument("--budget", type=float, default=100)
    parser.add_argument("--max-per-team", type=int, default=3)
    parser.add_argument("--formation", default="4-3-3")
    parser.add_argument("--max-transfers", type=int, default=10)
    parser.add_argument("--min-expected-minutes-probability", type=float, default=0.6)
    parser.add_argument(
        "--captain-min-expected-minutes-probability",
        type=float,
        default=0.75,
        help="Minimum expected minutes probability for safe captain candidates.",
    )
    parser.add_argument("--allow-injury-risk", action="store_true")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser


def format_optional_number(value: Any, digits: int = 1) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "N/A"
    return f"{float(number):.{digits}f}"


def safe_captain_pool(xi: pd.DataFrame, captain_threshold: float, allow_injury_risk: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if xi.empty:
        return xi.copy(), xi.copy()
    minutes = pd.to_numeric(px.get_series(xi, "expected_minutes_probability"), errors="coerce")
    injury = pd.to_numeric(px.get_series(xi, "bd_injury_penalty"), errors="coerce").fillna(0)
    if allow_injury_risk:
        safe = xi.loc[minutes.fillna(0).ge(captain_threshold)].copy()
    else:
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


def captain_recommendation(
    xi: pd.DataFrame,
    captain_threshold: float,
    allow_injury_risk: bool,
) -> tuple[str, str]:
    safe, risky = safe_captain_pool(xi, captain_threshold, allow_injury_risk)
    pool = safe if not safe.empty else xi.copy()
    if pool.empty:
        return "N/A", "N/A"
    captain = pool.head(1).iloc[0]
    vice = pool.iloc[1] if len(pool) > 1 else captain
    captain_text = f"{captain.get('player', 'N/A')} ({captain.get('team', '')})"
    vice_text = f"{vice.get('player', 'N/A')} ({vice.get('team', '')})"
    return captain_text, vice_text


def make_match_key(player: Any, team: Any, position: Any) -> str:
    return (
        normalize_player_name(player)
        + "||"
        + normalize_team(team)
        + "||"
        + px.normalize_position(position)
    )


def apply_transfer_plan(current: pd.DataFrame, rankings: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    if current.empty:
        return current.copy()

    work = current.copy()
    ranking_lookup = {}
    if not rankings.empty and "match_key" in rankings.columns:
        ranking_lookup = {str(row["match_key"]): row for _, row in rankings.drop_duplicates("match_key").iterrows()}

    for _, row in plan.iterrows():
        sell_key = make_match_key(row.get("sell_player"), row.get("sell_team"), row.get("sell_position"))
        buy_key = make_match_key(row.get("buy_player"), row.get("buy_team"), row.get("buy_position"))
        if "match_key" in work.columns:
            work = work.loc[work["match_key"].astype(str).ne(sell_key)].copy()
        buy_row = ranking_lookup.get(buy_key)
        if buy_row is None:
            continue
        work = pd.concat([work, pd.DataFrame([buy_row])], ignore_index=True, sort=False)
    return work


def sum_points(frame: pd.DataFrame) -> float | None:
    if frame is None or frame.empty:
        return None
    series = pd.to_numeric(px.get_series(frame, "expected_fantasy_points"), errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.sum())


def sum_price(frame: pd.DataFrame) -> float | None:
    if frame is None or frame.empty:
        return None
    series = pd.to_numeric(px.get_series(frame, "price"), errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.sum())


def render_plan_section(
    plan_name: str,
    plan_limit: int,
    current: pd.DataFrame,
    rankings: pd.DataFrame,
    plan: pd.DataFrame,
    warnings: list[str],
    formation: tuple[int, int, int],
    budget: float,
    max_per_team: int,
    captain_threshold: float,
    allow_injury_risk: bool,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    suggested_current = apply_transfer_plan(current, rankings, plan) if not plan.empty else current.copy()
    suggested = px.choose_xi(suggested_current, formation) if not suggested_current.empty else suggested_current.copy()
    current_xi, current_bench = current_squad_summary(current)
    suggested_xi, suggested_bench = current_squad_summary(suggested)

    current_xi_points = sum_points(current_xi)
    suggested_xi_points = sum_points(suggested_xi)
    current_total = sum_price(current)
    suggested_total = sum_price(suggested_current)
    captain_text, vice_text = captain_recommendation(suggested_xi if not suggested_xi.empty else current_xi, captain_threshold, allow_injury_risk)

    summary = pd.DataFrame(
        [
            {
                "plan_name": plan_name,
                "target_transfers": plan_limit,
                "actual_transfers": int(len(plan)),
                "current_squad_value": current_total,
                "suggested_squad_value": suggested_total,
                "net_budget_change": (suggested_total - current_total) if pd.notna(current_total) and pd.notna(suggested_total) else pd.NA,
                "current_xi_expected_points": current_xi_points,
                "suggested_xi_expected_points": suggested_xi_points,
                "captain_recommendation": captain_text,
                "vice_captain_recommendation": vice_text,
            }
        ]
    )

    if not plan.empty:
        plan = plan.copy()
        plan["plan_name"] = plan_name
        plan["target_transfers"] = plan_limit
        plan["actual_transfers"] = int(len(plan))
        plan["current_squad_value"] = current_total
        plan["suggested_squad_value"] = suggested_total
        plan["net_budget_change"] = (suggested_total - current_total) if pd.notna(current_total) and pd.notna(suggested_total) else pd.NA
        plan["current_xi_expected_points"] = current_xi_points
        plan["suggested_xi_expected_points"] = suggested_xi_points
        plan["captain_recommendation"] = captain_text
        plan["vice_captain_recommendation"] = vice_text
    else:
        plan = pd.DataFrame(columns=list(px.TRANSFER_OUTPUT_COLUMNS) + [
            "plan_name",
            "target_transfers",
            "actual_transfers",
            "current_squad_value",
            "suggested_squad_value",
            "net_budget_change",
            "current_xi_expected_points",
            "suggested_xi_expected_points",
            "captain_recommendation",
            "vice_captain_recommendation",
        ])

    parts = [f"## {plan_name} Plan", ""]
    if warnings:
        parts.append("Warnings:")
        parts.extend(f"- {warning}" for warning in warnings)
        parts.append("")
    parts.extend(
        [
            f"- Transfer limit: `{plan_limit}`",
            f"- Actual transfers: `{len(plan)}`",
            f"- Current squad value: `{format_optional_number(current_total, 1)}`",
            f"- Suggested squad value: `{format_optional_number(suggested_total, 1)}`",
            f"- Net budget change: `{format_optional_number((suggested_total - current_total) if pd.notna(current_total) and pd.notna(suggested_total) else pd.NA, 1)}`",
            f"- Projected XI expected points before: `{format_optional_number(current_xi_points, 1)}`",
            f"- Projected XI expected points after: `{format_optional_number(suggested_xi_points, 1)}`",
            f"- Captain recommendation: `{captain_text}`",
            f"- Vice captain recommendation: `{vice_text}`",
            "",
            "### Transfer Pairs",
            "",
            px.render_table(plan, list(px.TRANSFER_OUTPUT_COLUMNS), 30) if not plan.empty else "_No transfer pairs were generated._",
            "",
            "### Possible Post-Transfer XI",
            "",
            px.render_table(suggested_xi, px.SUGGESTED_DISPLAY_COLUMNS, 11) if not suggested_xi.empty else "_No XI could be built._",
            "",
            "### Possible Post-Transfer Bench",
            "",
            px.render_table(suggested_bench, px.SUGGESTED_DISPLAY_COLUMNS, 10) if not suggested_bench.empty else "_No bench could be built._",
        ]
    )
    return "\n".join(parts), summary, plan


def normalize_transfer_targets(max_transfers: int) -> list[tuple[str, int]]:
    targets = [
        ("Conservative", min(2, max_transfers)),
        ("Balanced", min(5, max_transfers)),
        ("Aggressive", max_transfers),
    ]
    seen: set[int] = set()
    result: list[tuple[str, int]] = []
    for name, limit in targets:
        if limit <= 0 or limit in seen:
            continue
        seen.add(limit)
        result.append((name, limit))
    return result


def build_report(
    rankings_csv: Path,
    current_squad_csv: Path,
    player_aliases_csv: Path | None,
    budget: float,
    max_per_team: int,
    formation: tuple[int, int, int],
    max_transfers: int,
    captain_threshold: float,
    min_expected_minutes_probability: float,
    allow_injury_risk: bool,
    current: pd.DataFrame,
    rankings: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    current = selection_metric(current)
    rankings = selection_metric(rankings)
    current_xi, current_bench = current_squad_summary(current)
    current_xi_points = sum_points(current_xi)
    current_total = sum_price(current)
    current_locked = bool(current["locked"].map(px.to_bool).any()) if "locked" in current.columns else False
    current_captain = current.loc[current["is_captain"]].copy()
    current_captain_text = "N/A"
    if not current_captain.empty:
        row = current_captain.iloc[0]
        current_captain_text = f"{row.get('player', 'N/A')} ({row.get('team', '')})"

    targets = normalize_transfer_targets(max_transfers)
    summary_rows: list[pd.DataFrame] = []
    sections: list[str] = []

    sections.append("# Transfer Plan")
    sections.append("")
    sections.append("Use this when transfers open or at the end of the round. Current round may be locked.")
    sections.append("")
    sections.append(f"- Current squad value: `{format_optional_number(current_total, 1)}`")
    sections.append(f"- Current XI expected points: `{format_optional_number(current_xi_points, 1)}`")
    sections.append(f"- Current captain: `{current_captain_text}`")
    sections.append(f"- Current round locked: `{current_locked}`")
    sections.append("")

    for plan_name, limit in targets:
        transfer_plan, warnings = px.build_transfer_plan(
            current,
            rankings,
            budget,
            max_per_team,
            limit,
            allow_injury_risk,
            min_expected_minutes_probability,
        )
        section_text, summary_df, plan_df = render_plan_section(
            plan_name,
            limit,
            current,
            rankings,
            transfer_plan,
            warnings,
            formation,
            budget,
            max_per_team,
            captain_threshold,
            allow_injury_risk,
        )
        sections.append(section_text)
        sections.append("")
        summary_rows.append(summary_df)

    sections.append("## Risk Notes")
    sections.append("")
    risk_notes = px.build_risk_warnings(current, None, budget, max_per_team, 0)
    if risk_notes:
        sections.extend(f"- {line}" for line in risk_notes)
    else:
        sections.append("- No additional current-squad warnings.")

    combined_summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    combined_plan_rows = []
    for plan_name, limit in targets:
        transfer_plan, _ = px.build_transfer_plan(
            current,
            rankings,
            budget,
            max_per_team,
            limit,
            allow_injury_risk,
            min_expected_minutes_probability,
        )
        if transfer_plan.empty:
            continue
        transfer_plan = transfer_plan.copy()
        transfer_plan["plan_name"] = plan_name
        transfer_plan["target_transfers"] = limit
        combined_plan_rows.append(transfer_plan)
    combined_plan = pd.concat(combined_plan_rows, ignore_index=True, sort=False) if combined_plan_rows else pd.DataFrame()

    if current is not None and not current.empty:
        injury_series = px.get_series(current, "bd_injury_penalty")
        if injury_series is not None:
            current_risk_rows = current.loc[pd.to_numeric(injury_series, errors="coerce").fillna(0) > 0]
            if not current_risk_rows.empty:
                sections.append("")
                sections.append("## Current Squad Injury Warnings")
                sections.append("")
                sections.append(px.render_table(current_risk_rows, px.CURRENT_DISPLAY_COLUMNS, 10))

    if not combined_summary.empty:
        sections.append("")
        sections.append("## Plan Summary")
        sections.append("")
        sections.append(px.render_table(combined_summary, [
            "plan_name",
            "target_transfers",
            "actual_transfers",
            "current_squad_value",
            "suggested_squad_value",
            "net_budget_change",
            "current_xi_expected_points",
            "suggested_xi_expected_points",
            "captain_recommendation",
            "vice_captain_recommendation",
        ], 10))

    return "\n".join(section for section in sections if section is not None), combined_plan


def main() -> int:
    args = build_parser().parse_args()
    if args.max_transfers < 1 or args.max_transfers > 15:
        raise ValueError("--max-transfers must be between 1 and 15.")

    rankings = build_rankings_view(read_csv(args.rankings_csv))
    current = build_current_squad(read_csv(args.current_squad_csv))
    aliases = load_player_aliases(args.player_aliases)
    if args.player_aliases and not args.player_aliases.exists():
        print(f"Warning: player alias file not found at {args.player_aliases}; continuing without aliases.")
    current_ranked = match_current_to_rankings(current, rankings, aliases, low_minutes_threshold=args.min_expected_minutes_probability)
    current_ranked = selection_metric(current_ranked)

    formation = px.normalize_formation(args.formation)
    report, combined_plan = build_report(
        args.rankings_csv,
        args.current_squad_csv,
        args.player_aliases if args.player_aliases and args.player_aliases.exists() else None,
        args.budget,
        args.max_per_team,
        formation,
        args.max_transfers,
        args.captain_min_expected_minutes_probability,
        args.min_expected_minutes_probability,
        args.allow_injury_risk,
        current_ranked,
        rankings,
    )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report + "\n", encoding="utf-8")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_plan.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(args.output_md)
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
