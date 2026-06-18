"""Download raw/open-data inputs for the fantasy pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests


RAW_GITHUB_BASE = "https://raw.githubusercontent.com"
WORLDCUP_API_GAMES_URLS = (
    "https://worldcup26.ir/get/games",
    "http://worldcup26.ir/get/games",
)

WORLDCUP_API_STADIUMS_URLS = (
    "https://worldcup26.ir/get/stadiums",
    "http://worldcup26.ir/get/stadiums",
)

DYNAMIC_FIXTURE_FILENAME = "world-cup_2026.csv"

@dataclass(frozen=True)
class DataSource:
    name: str
    url: str
    filename: str
    required: bool = False


SOURCES: tuple[DataSource, ...] = (
    DataSource(
        name="fantasy_players",
        url=f"{RAW_GITHUB_BASE}/jlbgouveia/fifa-wc2026-fantasy-analytics/main/wc2026_players.csv",
        filename="wc2026_players.csv",
        required=True,
    ),
    DataSource(
        name="player_stats_apifootball",
        url=f"{RAW_GITHUB_BASE}/jlbgouveia/fifa-wc2026-fantasy-analytics/main/wc2026_player_stats_apifootball.csv",
        filename="wc2026_player_stats_apifootball.csv",
        required=True,
    ),
    DataSource(
        name="team_stats",
        url=f"{RAW_GITHUB_BASE}/jlbgouveia/fifa-wc2026-fantasy-analytics/main/wc2026_team_stats.csv",
        filename="wc2026_team_stats.csv",
    ),
    DataSource(
        name="team_stats_by_season",
        url=f"{RAW_GITHUB_BASE}/jlbgouveia/fifa-wc2026-fantasy-analytics/main/wc2026_team_stats_by_season.csv",
        filename="wc2026_team_stats_by_season.csv",
    ),
    DataSource(
        name="elo_calibrated",
        url=f"{RAW_GITHUB_BASE}/Hicruben/world-cup-2026-prediction-model/main/data/elo-calibrated.json",
        filename="elo-calibrated.json",
    ),
    DataSource(
        name="cup26_probabilities_csv",
        url="https://cup26matches.com/probabilities.csv",
        filename="probabilities.csv",
    ),
    DataSource(
        name="cup26_probabilities_json",
        url="https://cup26matches.com/probabilities.json",
        filename="probabilities.json",
    ),
    DataSource(
        name="rezarahiminia_worldcup_games",
        url=f"{RAW_GITHUB_BASE}/rezarahiminia/worldcup2026/main/worldcup2026.games.csv",
        filename="worldcup2026.games.csv",
    ),
    DataSource(
        name="rezarahiminia_worldcup_teams",
        url=f"{RAW_GITHUB_BASE}/rezarahiminia/worldcup2026/main/worldcup2026.teams.csv",
        filename="worldcup2026.teams.csv",
    ),
    DataSource(
        name="rezarahiminia_worldcup_stadia",
        url=f"{RAW_GITHUB_BASE}/rezarahiminia/worldcup2026/main/worldcup2026.stadia.csv",
        filename="worldcup2026.stadia.csv",
    ),
)


def download_all(
    raw_dir: Path,
    sources: Iterable[DataSource] = SOURCES,
    overwrite: bool = False,
    timeout: int = 45,
) -> dict:
    """Download known raw/open-data files and write a manifest.

    The source list intentionally uses direct raw GitHub URLs plus the
    cup26matches.com open-data endpoints linked by the prediction-model README.
    It does not crawl or scrape HTML pages.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_note": "Only direct raw GitHub and explicitly linked open-data URLs are fetched; no HTML scraping.",
        "sources": [],
    }

    session = requests.Session()
    session.headers.update({"User-Agent": "wc-fantasy-pandas-pipeline/1.0"})

    for source in sources:
        destination = raw_dir / source.filename
        record = {
            "name": source.name,
            "url": source.url,
            "filename": source.filename,
            "required": source.required,
            "status": "skipped",
            "bytes": destination.stat().st_size if destination.exists() else 0,
            "fetched_at_utc": None,
            "error": None,
        }

        if destination.exists() and not overwrite:
            record["status"] = "exists"
            manifest["sources"].append(record)
            continue

        try:
            response = session.get(source.url, timeout=timeout)
            record["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
            if response.status_code == 200 and response.content:
                destination.write_bytes(response.content)
                record["status"] = "downloaded"
                record["bytes"] = len(response.content)
            else:
                record["status"] = "failed"
                record["error"] = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            record["status"] = "failed"
            record["error"] = str(exc)

        manifest["sources"].append(record)

    (raw_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest

def fetch_json(session: requests.Session, url: str, timeout: int) -> dict | list:
    """Fetch JSON from a direct API endpoint."""
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_json_with_fallbacks(
    session: requests.Session,
    urls: tuple[str, ...],
    timeout: int,
) -> tuple[dict | list, str]:
    """Try multiple API URLs and return the first successful JSON payload."""
    errors: list[str] = []

    for url in urls:
        try:
            return fetch_json(session, url, timeout), url
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError("All API URLs failed: " + " | ".join(errors))


def extract_records(payload: dict | list, possible_keys: tuple[str, ...]) -> list[dict]:
    """Extract list records from either a list response or a dict wrapper."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        for key in possible_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]

    return []


def truthy(value: object) -> bool:
    """Normalize API booleans that may arrive as strings."""
    return str(value).strip().lower() in {"true", "1", "yes", "y", "finished", "complete"}


def parse_local_date_time(value: object) -> tuple[str, str]:
    """Return date and time strings from the API local_date field."""
    text = str(value or "").strip()
    if not text:
        return "", ""

    formats = (
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    )

    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.date().isoformat(), parsed.strftime("%H:%M")
        except ValueError:
            continue

    # Fallback: keep the date-like part if parsing fails.
    parts = text.split()
    if len(parts) >= 2:
        return parts[0], parts[1]
    return text, ""


def build_stadium_lookup(stadium_records: list[dict]) -> dict[str, dict[str, str]]:
    """Build stadium_id -> stadium/city lookup from the stadium API."""
    lookup: dict[str, dict[str, str]] = {}

    for stadium in stadium_records:
        stadium_id = str(stadium.get("id", "")).strip()
        if not stadium_id:
            continue

        lookup[stadium_id] = {
            "stadium": str(
                stadium.get("name_en")
                or stadium.get("fifa_name")
                or stadium.get("name")
                or ""
            ).strip(),
            "city": str(
                stadium.get("city_en")
                or stadium.get("city")
                or stadium.get("country_en")
                or ""
            ).strip(),
        }

    return lookup


def api_match_status(game: dict) -> str:
    """Map API match fields to the local fixture status format."""
    if truthy(game.get("finished")):
        return "Complete"

    time_elapsed = str(game.get("time_elapsed", "")).strip().lower()
    if time_elapsed and time_elapsed not in {"notstarted", "not started", "0", "none", "null"}:
        return "Live"

    return "Scheduled"


def api_match_result(game: dict, status: str) -> str:
    """Return result only for live/complete matches."""
    if status not in {"Live", "Complete"}:
        return ""

    home_score = game.get("home_score")
    away_score = game.get("away_score")

    if home_score is None or away_score is None:
        return ""

    home_score_text = str(home_score).strip()
    away_score_text = str(away_score).strip()

    if not home_score_text or not away_score_text:
        return ""

    return f"{home_score_text}-{away_score_text}"


def convert_games_to_fixture_rows(
    game_records: list[dict],
    stadium_lookup: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Convert World Cup API games to the CSV format expected by the pipeline."""
    rows: list[dict[str, str]] = []

    for game in game_records:
        date_text, time_text = parse_local_date_time(game.get("local_date"))
        stadium_id = str(game.get("stadium_id", "")).strip()
        stadium_info = stadium_lookup.get(stadium_id, {})

        home_team = str(
            game.get("home_team_name_en")
            or game.get("home_team_label")
            or ""
        ).strip()
        away_team = str(
            game.get("away_team_name_en")
            or game.get("away_team_label")
            or ""
        ).strip()

        # Skip knockout placeholders where both teams are still unknown.
        if not home_team and not away_team:
            continue

        status = api_match_status(game)

        rows.append(
            {
                "date": date_text,
                "time": time_text,
                "matchday": str(game.get("matchday") or game.get("group") or game.get("type") or "").strip(),
                "home_team": home_team,
                "away_team": away_team,
                "city": stadium_info.get("city", ""),
                "stadium": stadium_info.get("stadium", ""),
                "status": status,
                "result": api_match_result(game, status),
            }
        )

    return rows


def write_fixture_csv(rows: list[dict[str, str]], destination: Path) -> None:
    """Write dynamic fixtures to the same CSV layout used by the ranking pipeline."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "date",
        "time",
        "matchday",
        "home_team",
        "away_team",
        "city",
        "stadium",
        "status",
        "result",
    ]

    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def download_dynamic_fixtures(
    raw_dir: Path,
    games_urls: tuple[str, ...] = WORLDCUP_API_GAMES_URLS,
    stadiums_urls: tuple[str, ...] = WORLDCUP_API_STADIUMS_URLS,
    timeout: int = 45,
) -> dict:
    """Download live World Cup fixtures and save them as world-cup_2026.csv.

    The existing local fixture CSV is overwritten only after the API response
    is successfully converted.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / DYNAMIC_FIXTURE_FILENAME

    record = {
        "name": "dynamic_worldcup_fixtures",
        "url": ", ".join(games_urls),
        "filename": DYNAMIC_FIXTURE_FILENAME,
        "required": False,
        "status": "failed",
        "rows": 0,
        "fetched_at_utc": None,
        "error": None,
    }

    session = requests.Session()
    session.headers.update({"User-Agent": "wc-fantasy-pandas-pipeline/1.0"})

    try:
        games_payload, used_games_url = fetch_json_with_fallbacks(
            session,
            games_urls,
            timeout=timeout,
        )
        stadiums_payload, used_stadiums_url = fetch_json_with_fallbacks(
            session,
            stadiums_urls,
            timeout=timeout,
        )

        game_records = extract_records(games_payload, ("games", "data", "rows"))
        stadium_records = extract_records(stadiums_payload, ("stadiums", "data", "rows"))


        game_records = extract_records(games_payload, ("games", "data", "rows"))
        stadium_records = extract_records(stadiums_payload, ("stadiums", "data", "rows"))

        stadium_lookup = build_stadium_lookup(stadium_records)
        rows = convert_games_to_fixture_rows(game_records, stadium_lookup)

        if not rows:
            raise ValueError("World Cup API returned no usable fixture rows.")

        write_fixture_csv(rows, destination)

        record["status"] = "downloaded"
        record["rows"] = len(rows)
        record["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
        record["error"] = None
        record["used_games_url"] = used_games_url
        record["used_stadiums_url"] = used_stadiums_url

    except Exception as exc:
        record["error"] = str(exc)

    return record

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download World Cup fantasy model inputs.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing files.")
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="Download dynamic World Cup fixtures/live scores into data/raw/world-cup_2026.csv.",
    )
    parser.add_argument(
        "--convert-fixtures",
        action="store_true",
        help="Convert downloaded rezarahiminia fixture CSV files into data/raw/world-cup_2026.csv.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    manifest = download_all(args.raw_dir, overwrite=args.overwrite, timeout=args.timeout)

    if args.fixtures:
        fixture_record = download_dynamic_fixtures(args.raw_dir, timeout=args.timeout)
        manifest["dynamic_fixtures"] = fixture_record

        manifest_path = args.raw_dir / "source_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    downloaded = sum(1 for source in manifest["sources"] if source["status"] == "downloaded")
    failed = [source for source in manifest["sources"] if source["status"] == "failed"]

    print(f"Downloaded {downloaded} files into {args.raw_dir}.")

    if args.fixtures:
        fixture_record = manifest["dynamic_fixtures"]
        if fixture_record["status"] == "downloaded":
            print(
                f"Downloaded dynamic fixtures into "
                f"{args.raw_dir / DYNAMIC_FIXTURE_FILENAME} "
                f"({fixture_record['rows']} rows)."
            )
        else:
            print("Dynamic fixtures could not be downloaded:")
            print(f"- {fixture_record['filename']}: {fixture_record['error']}")
            return 1

    if failed:
        print("Some optional or required files could not be downloaded:")
        for source in failed:
            print(f"- {source['filename']}: {source['error']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
