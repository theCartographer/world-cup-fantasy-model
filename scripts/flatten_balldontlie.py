from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_ENDPOINTS = [
    "players",
    "rosters",
    "player_injuries",
    "matches",
    "match_lineups",
    "match_events",
    "player_match_stats",
    "team_match_stats",
    "match_shots",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flatten BALLDONTLIE raw JSON files into CSVs.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/balldontlie"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/balldontlie"))
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=DEFAULT_ENDPOINTS,
        help="BALLDONTLIE endpoints to flatten.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing flattened CSV files.",
    )
    return parser


def flatten_endpoint(input_dir: Path, output_dir: Path, endpoint: str, overwrite: bool) -> None:
    input_path = input_dir / f"{endpoint}.json"
    output_path = output_dir / f"{endpoint}.csv"

    if not input_path.exists():
        print(f"Warning: missing {input_path}")
        return

    if output_path.exists() and not overwrite:
        print(f"Skipping existing {output_path}")
        return

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object with a data array")

        data = payload.get("data", [])
        if data is None:
            data = []
        if not isinstance(data, list):
            raise ValueError("expected payload['data'] to be a list")

        frame = pd.json_normalize(data)
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print(f"{endpoint}: wrote {len(frame)} rows to {output_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to flatten {input_path}: {exc}") from exc


def main() -> int:
    args = build_parser().parse_args()
    had_error = False

    for endpoint in args.endpoints:
        try:
            flatten_endpoint(args.input_dir, args.output_dir, endpoint, args.overwrite)
        except Exception as exc:
            had_error = True
            print(str(exc))

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
