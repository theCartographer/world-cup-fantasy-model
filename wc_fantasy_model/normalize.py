"""Name normalization helpers for teams and players."""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import Any


TEAM_ABBREVIATIONS = {
    "alg": "algeria",
    "arg": "argentina",
    "aus": "australia",
    "aut": "austria",
    "bel": "belgium",
    "bih": "bosnia and herzegovina",
    "bra": "brazil",
    "can": "canada",
    "civ": "ivory coast",
    "col": "colombia",
    "crc": "costa rica",
    "cro": "croatia",
    "cuw": "curacao",
    "cze": "czechia",
    "den": "denmark",
    "ecu": "ecuador",
    "egy": "egypt",
    "eng": "england",
    "esp": "spain",
    "fra": "france",
    "ger": "germany",
    "gha": "ghana",
    "hai": "haiti",
    "irn": "iran",
    "irq": "iraq",
    "ita": "italy",
    "jor": "jordan",
    "jpn": "japan",
    "kor": "south korea",
    "mar": "morocco",
    "mex": "mexico",
    "nld": "netherlands",
    "nor": "norway",
    "nzl": "new zealand",
    "pan": "panama",
    "par": "paraguay",
    "por": "portugal",
    "qat": "qatar",
    "rsa": "south africa",
    "sau": "saudi arabia",
    "sco": "scotland",
    "sen": "senegal",
    "sui": "switzerland",
    "swe": "sweden",
    "tun": "tunisia",
    "tur": "turkey",
    "uru": "uruguay",
    "usa": "united states",
    "uzb": "uzbekistan",
}


TEAM_ALIASES = {
    "bosnia herzegovina": "bosnia and herzegovina",
    "bosnia herz": "bosnia and herzegovina",
    "bosnia": "bosnia and herzegovina",
    "cabo verde": "cape verde",
    "cape verde": "cape verde",
    "congo dr": "dr congo",
    "democratic republic of congo": "dr congo",
    "drc": "dr congo",
    "dr congo": "dr congo",
    "cote d ivoire": "ivory coast",
    "cote divoire": "ivory coast",
    "ivory coast": "ivory coast",
    "curacao": "curacao",
    "czech republic": "czechia",
    "czechia": "czechia",
    "england": "england",
    "ir iran": "iran",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "south korea": "south korea",
    "saudi": "saudi arabia",
    "turkiye": "turkey",
    "turkey": "turkey",
    "u s a": "united states",
    "united states of america": "united states",
    "united states": "united states",
    "usa": "united states",
    "usmnt": "united states",
}

EXTERNAL_TEAM_ALIASES: dict[str, str] = {}


def strip_accents(value: str) -> str:
    """Return an ASCII approximation without changing word order."""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def basic_key(value: Any) -> str:
    """Lowercase, accent-free key used for fuzzy-ish exact joins."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    text = strip_accents(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"['`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_team(value: Any) -> str:
    """Normalize country/team names and common FIFA abbreviations."""
    key = basic_key(value)
    if not key:
        return ""
    if key in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[key]
    if key in EXTERNAL_TEAM_ALIASES:
        return EXTERNAL_TEAM_ALIASES[key]
    return TEAM_ALIASES.get(key, key)


def canonical_team_key(value: Any) -> str:
    """Normalize a configured canonical name without depending on external aliases."""
    key = basic_key(value)
    if not key:
        return ""
    if key in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[key]
    return TEAM_ALIASES.get(key, key)


def load_team_aliases(path: Path | str | None) -> dict[str, str]:
    """Load user-editable aliases from a simple alias,canonical CSV file."""
    if path is None:
        EXTERNAL_TEAM_ALIASES.clear()
        return {}

    alias_path = Path(path)
    if not alias_path.exists():
        EXTERNAL_TEAM_ALIASES.clear()
        return {}

    aliases: dict[str, str] = {}
    with alias_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"alias", "canonical"}
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Team alias file {alias_path} must contain columns: alias, canonical"
            )

        for row in reader:
            alias_key = basic_key(row.get("alias"))
            canonical_key = canonical_team_key(row.get("canonical"))
            if alias_key and canonical_key:
                aliases[alias_key] = canonical_key

    EXTERNAL_TEAM_ALIASES.clear()
    EXTERNAL_TEAM_ALIASES.update(aliases)
    return dict(EXTERNAL_TEAM_ALIASES)


def normalize_player_name(value: Any) -> str:
    """Normalize player names for deterministic joins across exports."""
    key = basic_key(value)
    key = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", key)
    return re.sub(r"\s+", " ", key).strip()


def display_team_from_key(team_key: str) -> str:
    """Convert a normalized team key back to a readable label."""
    if not team_key:
        return ""
    return " ".join(part.capitalize() for part in team_key.split())
