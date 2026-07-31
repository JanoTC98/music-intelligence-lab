from __future__ import annotations

import re
import unicodedata

import pandas as pd

from spotify_intelligence.data.contracts import load_rules_config

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(
    value: str,
    casefold: bool = True,
    trim: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Conservative text normalization for exact identity grouping.

    NFKC -> optional casefold -> optional trim -> optional whitespace collapse.
    Punctuation and internal word order are preserved.
    """
    result = unicodedata.normalize("NFKC", value)
    if casefold:
        result = result.casefold()
    if trim:
        result = result.strip()
    if collapse_whitespace:
        result = _WHITESPACE_RE.sub(" ", result)
    return result


def normalize_identity_fields(
    df: pd.DataFrame,
    text_columns: tuple[str, ...] = ("track_name", "artists"),
    config_path: str = "configs/data_rules.yaml",
) -> pd.DataFrame:
    """Add ``<column>_normalized`` columns using the versioned identity rules."""
    config = load_rules_config(config_path)
    identity = config.get("identity", {})

    result = df.copy()
    for column in text_columns:
        if column not in result.columns:
            continue
        normalized_column = f"{column}_normalized"
        result[normalized_column] = result[column].map(
            lambda value: (
                normalize_text(
                    str(value),
                    casefold=bool(identity.get("casefold", True)),
                    trim=bool(identity.get("trim", True)),
                    collapse_whitespace=bool(identity.get("collapse_whitespace", True)),
                )
                if pd.notna(value)
                else value
            )
        )
    return result
