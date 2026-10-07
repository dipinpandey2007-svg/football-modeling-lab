"""
Checksum manifest for the raw data behind the reported results.

data/MANIFEST.sha256 lists a SHA-256 for every raw file in data/raw/ as it was
when the reported results were produced. It is tracked in git (the data itself
is not), so anyone who rebuilds the data can see whether their copy matches.

CSV checksums are computed with carriage returns removed, so the same
football-data content hashes identically on Windows and Linux (the cached
files on Windows have \\r\\r\\n line endings; pandas parses both the same way).

Expected differences:
    *_2026.*  2026/27 is in progress, so rebuilding later adds matches. The
              analyses only use matches up to the analysis cutoff
              (config.ANALYSIS_CUTOFF), so the reported numbers are unaffected
              provided the earlier matches are unchanged.
    others    Understat or football-data may revise historical data; a mismatch
              on a completed season means the provider's data has changed.

Usage:
    python -m src.ingest.manifest            # verify data/raw against the manifest
    python -m src.ingest.manifest --write    # (re)write the manifest from data/raw
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .config import DATA_DIR, RAW_DIR

MANIFEST_PATH = DATA_DIR / "MANIFEST.sha256"


def file_digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix == ".csv":
        data = data.replace(b"\r", b"")
    return hashlib.sha256(data).hexdigest()


def current_digests() -> dict[str, str]:
    return {p.name: file_digest(p) for p in sorted(RAW_DIR.glob("*")) if p.is_file()}


def read_manifest() -> dict[str, str]:
    out = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def write_manifest() -> None:
    lines = ["# SHA-256 of data/raw/* used for the reported results (CSV: \\r removed).",
             "# Verify with: python -m src.ingest.manifest"]
    lines += [f"{digest}  {name}" for name, digest in current_digests().items()]
    MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def verify() -> dict[str, str]:
    """Status per manifest entry: 'ok', 'differs' or 'missing'."""
    expected, actual = read_manifest(), current_digests()
    return {name: ("missing" if name not in actual else "ok" if actual[name] == digest else "differs")
            for name, digest in expected.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="rewrite the manifest from data/raw")
    if ap.parse_args().write:
        write_manifest()
        print(f"Wrote {MANIFEST_PATH} ({len(current_digests())} files)")
    else:
        status = verify()
        for name, s in status.items():
            if s != "ok":
                print(f"  {s:8s} {name}")
        n_ok = sum(s == "ok" for s in status.values())
        print(f"{n_ok}/{len(status)} raw files match {MANIFEST_PATH.name}")
