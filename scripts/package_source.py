"""Create a clean source distribution ZIP under dist/.

Excludes Git metadata, virtual environments, caches, raw data,
processed data, models, reports, and secret files.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_PATTERNS = [
    ".git/",
    ".venv/",
    ".opencode/node_modules/",
    ".idea/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    "__pycache__/",
    "*.pyc",
    "data/raw/",
    "data/interim/",
    "data/quarantine/",
    "data/processed/",
    "models/",
    "reports/figures/",
    "dist/",
    "docs/archive/",
]

SECRET_PATTERNS = [
    re.compile(r"password\s*=\s*", re.IGNORECASE),
    re.compile(r"api[_-]?key\s*=\s*", re.IGNORECASE),
    re.compile(r"secret\s*=\s*", re.IGNORECASE),
    re.compile(r"token\s*=\s*", re.IGNORECASE),
]

INCLUDE_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".md",
    ".txt",
    ".json",
    ".lock",
    ".example",
    ".csv",
    ".parquet",
    ".joblib",
    ".npy",
    ".npz",
    ".pkl",
}

INCLUDE_FILES = {
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".env.example",
}


def _should_exclude(path: Path, root: Path) -> bool:
    rel = path.resolve().relative_to(root.resolve())
    rel_str = str(rel).replace("\\", "/")

    for pattern in EXCLUDE_PATTERNS:
        if pattern.endswith("/"):
            prefix = pattern.rstrip("/")
            if rel_str == prefix or rel_str.startswith(prefix + "/"):
                return True
        elif "*" in pattern:
            regex = pattern.replace(".", r"\.").replace("*", ".*")
            if re.fullmatch(regex, rel_str):
                return True
        else:
            if rel_str == pattern:
                return True

    return bool(path.suffix and path.suffix not in INCLUDE_EXTENSIONS)


def _has_secrets(path: Path) -> bool:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return False
    return any(pattern.search(content) for pattern in SECRET_PATTERNS)


def package_source(dry_run: bool = False, version: str | None = None) -> Path:
    root = PROJECT_ROOT
    dist_dir = root / "dist"
    dist_dir.mkdir(exist_ok=True)

    if version:
        name = f"spotify-music-intelligence-{version}"
    else:
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        name = f"spotify-music-intelligence-{date_str}"

    zip_path = dist_dir / f"{name}.zip"

    files_to_include: list[Path] = []
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        if _should_exclude(item, root):
            continue
        if item.name in (".env", ".gitignore"):
            continue
        if (
            item.name == ".env.example"
            or item.suffix in INCLUDE_EXTENSIONS
            or item.name in INCLUDE_FILES
        ):
            if _has_secrets(item):
                print(f"SECRET DETECTED: {item.relative_to(root)}")
                sys.exit(1)
            files_to_include.append(item)

    if dry_run:
        print(f"DRY RUN: would create {zip_path}")
        print(f"Files to include: {len(files_to_include)}")
        for f in files_to_include:
            print(f"  {f.relative_to(root)}")
        return zip_path

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_include:
            arcname = f.relative_to(root)
            zf.write(f, arcname)

    size = zip_path.stat().st_size
    print(f"Created: {zip_path}")
    print(f"Files: {len(files_to_include)}")
    print(f"Size: {size:,} bytes")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clean source distribution ZIP.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without creating the ZIP",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version string for the archive name",
    )
    args = parser.parse_args()
    package_source(dry_run=args.dry_run, version=args.version)


if __name__ == "__main__":
    main()
