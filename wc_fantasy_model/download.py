"""Download raw/open-data inputs for the fantasy pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests


RAW_GITHUB_BASE = "https://raw.githubusercontent.com"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download World Cup fantasy model inputs.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing files.")
    parser.add_argument("--timeout", type=int, default=45)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = download_all(args.raw_dir, overwrite=args.overwrite, timeout=args.timeout)
    downloaded = sum(1 for source in manifest["sources"] if source["status"] == "downloaded")
    failed = [source for source in manifest["sources"] if source["status"] == "failed"]
    print(f"Downloaded {downloaded} files into {args.raw_dir}.")
    if failed:
        print("Some optional or required files could not be downloaded:")
        for source in failed:
            print(f"- {source['filename']}: {source['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
