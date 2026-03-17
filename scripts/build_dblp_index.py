"""
scripts/build_dblp_index.py

Downloads the DBLP XML dump and builds a local SQLite FTS5 database
for offline CS conference paper verification.

Usage:
    # First time (downloads everything)
    uv run python scripts/build_dblp_index.py

    # XML already downloaded, skip re-download
    uv run python scripts/build_dblp_index.py --xml-path data/dblp/dblp.xml.gz

    # Monthly refresh (re-downloads and rebuilds)
    uv run python scripts/build_dblp_index.py --force-download

    # Check if database is stale
    uv run python scripts/build_dblp_index.py --check

Size:   ~1GB download (compressed), ~2GB SQLite database
Time:   20-40 minutes depending on hardware
Output: $DBLP_DB_PATH (default: ./data/dblp/dblp.db)

Requires: lxml  (uv add lxml)
"""

import argparse
import gzip
import os
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DBLP_XML_URL = "https://dblp.org/xml/dblp.xml.gz"
DBLP_DTD_URL = "https://dblp.org/xml/dblp.dtd"

PUBLICATION_TYPES = {
    "article",
    "inproceedings",
    "proceedings",
    "incollection",
    "book",
    "phdthesis",
    "mastersthesis",
}

BATCH_SIZE = 10_000
PROGRESS_EVERY = 100_000
STALENESS_DAYS = 30


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS papers USING fts5(
    dblp_key,
    title,
    authors,
    venue,
    year UNINDEXED,
    url UNINDEXED,
    tokenize = "unicode61 remove_diacritics 2"
);
"""

CREATE_METADATA_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute(CREATE_TABLE_SQL)
    conn.execute(CREATE_METADATA_SQL)
    conn.commit()
    return conn


def get_metadata(db_path: Path) -> dict:
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT key, value FROM metadata").fetchall()
        conn.close()
        return dict(rows)
    except sqlite3.Error:
        return {}


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------


def check_staleness(db_path: Path) -> None:
    meta = get_metadata(db_path)
    if not meta:
        print(f"❌ No database found at {db_path}")
        print("   Run: uv run python scripts/build_dblp_index.py")
        sys.exit(1)

    built_at = meta.get("built_at", "unknown")
    total = meta.get("total_records", "unknown")

    print(f"Database: {db_path}")
    print(f"Built at: {built_at}")
    print(f"Records:  {total}")

    if built_at != "unknown":
        built_dt = datetime.fromisoformat(built_at.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - built_dt).days
        print(f"Age:      {age_days} days")

        if age_days > STALENESS_DAYS:
            print(
                f"\n⚠️  Database is {age_days} days old (>{STALENESS_DAYS} days threshold)."
            )
            print(
                "   Refresh with: uv run python scripts/build_dblp_index.py --force-download"
            )
        else:
            print(f"\n✅ Database is fresh ({age_days}/{STALENESS_DAYS} days).")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _progress_hook(block_num, block_size, total_size):
    if total_size > 0:
        downloaded = min(block_num * block_size, total_size)
        pct = downloaded * 100 / total_size
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {pct:.1f}% ({mb:.0f} / {total_mb:.0f} MB)", end="", flush=True)


def download(url: str, dest: Path, force: bool = False) -> bool:
    """
    Download url to dest. Skips if dest exists and force=False.
    Returns True if downloaded, False if skipped.
    """
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1_048_576
        print(f"  Already exists: {dest} ({size_mb:.0f} MB) — skipping")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    print(f"  → {dest}")
    urllib.request.urlretrieve(url, dest, _progress_hook)
    print()
    return True


# ---------------------------------------------------------------------------
# XML parsing — lxml handles DTD entities automatically
# ---------------------------------------------------------------------------


def parse_element(elem) -> dict | None:
    dblp_key = elem.get("key", "")
    if not dblp_key:
        return None

    title, venue, year, url = "", "", "", ""
    authors = []

    for child in elem:
        text = (child.text or "").strip()
        if not text:
            continue
        if child.tag == "title":
            title = text.rstrip(".")
        elif child.tag in ("author", "editor"):
            authors.append(text)
        elif child.tag in ("journal", "booktitle") and not venue:
            venue = text
        elif child.tag == "year":
            year = text
        elif child.tag == "ee" and not url:
            url = text

    return (
        {
            "dblp_key": dblp_key,
            "title": title,
            "authors": " | ".join(authors),
            "venue": venue,
            "year": year,
            "url": url,
        }
        if title
        else None
    )


def stream_xml(xml_gz_path: Path, dtd_path: Path):
    """
    Stream-parse gzipped DBLP XML using lxml.
    lxml resolves DTD entities (uuml, auml, etc.) automatically when
    the DTD file is in the same directory as the XML.
    """
    if not dtd_path.exists():
        raise FileNotFoundError(
            f"DTD not found: {dtd_path}\n"
            "Download it with: wget https://dblp.org/xml/dblp.dtd"
        )

    count = 0
    t0 = time.time()

    # lxml resolves the DTD via relative path — chdir to the XML directory
    original_cwd = Path.cwd()
    xml_gz_path = xml_gz_path.resolve()
    dtd_path = dtd_path.resolve()
    os.chdir(dtd_path.parent)

    try:
        with gzip.open(xml_gz_path, "rb") as f:
            parser = etree.iterparse(
                f,
                events=("start", "end"),
                load_dtd=True,
                resolve_entities=True,
                huge_tree=True,
            )

            current = None
            for event, elem in parser:
                if event == "start" and elem.tag in PUBLICATION_TYPES:
                    current = elem.tag
                elif event == "end" and elem.tag in PUBLICATION_TYPES:
                    if current == elem.tag:
                        pub = parse_element(elem)
                        if pub:
                            count += 1
                            yield pub
                            if count % PROGRESS_EVERY == 0:
                                elapsed = time.time() - t0
                                print(
                                    f"  {count:,} records ({count/elapsed:.0f}/sec) [{elapsed:.0f}s]"
                                )
                    current = None
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

    finally:
        os.chdir(original_cwd)

    print(f"\n  Total: {count:,} records in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------


def build_index(db_path: Path, xml_gz_path: Path, dtd_path: Path) -> None:
    print(f"\nBuilding index → {db_path}")

    conn = open_db(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM papers")
    conn.commit()

    batch, total = [], 0
    t0 = time.time()

    for pub in stream_xml(xml_gz_path, dtd_path):
        batch.append(
            (
                pub["dblp_key"],
                pub["title"],
                pub["authors"],
                pub["venue"],
                pub["year"],
                pub["url"],
            )
        )
        if len(batch) >= BATCH_SIZE:
            cur.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?)", batch)
            conn.commit()
            total += len(batch)
            batch.clear()

    if batch:
        cur.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?)", batch)
        conn.commit()
        total += len(batch)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cur.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", ("built_at", now))
    cur.execute(
        "INSERT OR REPLACE INTO metadata VALUES (?,?)", ("total_records", str(total))
    )
    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    db_mb = db_path.stat().st_size / 1_048_576

    print(f"\n✅ Done — {total:,} records | {db_mb:.0f} MB | {elapsed:.0f}s")
    print(f"   Built at: {now}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Build DBLP SQLite index for CiteGuard")
    ap.add_argument(
        "--db-path",
        type=Path,
        default=Path(os.getenv("DBLP_DB_PATH", "./data/dblp/dblp.db")),
    )
    ap.add_argument(
        "--xml-path",
        type=Path,
        default=None,
        help="Use existing dblp.xml.gz (skips download if file exists)",
    )
    ap.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if files exist (for monthly refresh)",
    )
    ap.add_argument(
        "--keep-xml",
        action="store_true",
        help="Keep XML after build (saves re-download time, costs ~1GB)",
    )
    ap.add_argument("--check", action="store_true", help="Check database age and exit")
    args = ap.parse_args()

    if args.check:
        check_staleness(args.db_path)

    xml_path = args.xml_path or args.db_path.parent / "dblp.xml.gz"
    dtd_path = xml_path.parent / "dblp.dtd"

    print("=" * 60)
    print("  CiteGuard — DBLP Index Builder")
    print("=" * 60)
    print(f"  Database : {args.db_path}")
    print(f"  XML      : {xml_path}")
    print(f"  DTD      : {dtd_path}")

    # DTD is small — always download if missing or force
    print("\n[DTD]")
    download(DBLP_DTD_URL, dtd_path, force=args.force_download)

    # XML is large — skip if already present unless forced
    print("\n[XML dump]")
    download(DBLP_XML_URL, xml_path, force=args.force_download)

    print("\n[Building index]")
    build_index(args.db_path, xml_path, dtd_path)

    if not args.keep_xml and xml_path.exists():
        xml_path.unlink()
        print(f"  Removed XML (use --keep-xml to retain it)")

    print(f"\n  Staleness check: uv run python scripts/build_dblp_index.py --check")
    print(
        f"  Monthly refresh: uv run python scripts/build_dblp_index.py --force-download"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
