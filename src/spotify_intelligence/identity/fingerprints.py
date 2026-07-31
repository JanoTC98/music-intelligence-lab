from __future__ import annotations

import hashlib
from typing import Any

EXACT_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "track_name_normalized",
    "artists_normalized",
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
)


def build_exact_fingerprint(fields: dict[str, Any]) -> str:
    """Stable UTF-8 serialization of the exact fingerprint fields.

    Field order is fixed and values are ``repr``-serialized so the result is
    deterministic across runs and platforms.
    """
    parts: list[str] = []
    for key in EXACT_FINGERPRINT_FIELDS:
        if key not in fields:
            raise ValueError(f"Missing fingerprint field: {key}")
        parts.append(f"{key}={fields[key]!r}")
    return "\x1e".join(parts)


def fingerprint_to_recording_group_id(fingerprint: str) -> str:
    """SHA-256 of the canonical fingerprint -> 64 hex characters."""
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
