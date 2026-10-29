"""Stream the source ArXiv CSV into normalized JSONL for downstream indexing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

REQUIRED_COLUMNS = {
    "id",
    "title",
    "category",
    "category_code",
    "published_date",
    "updated_date",
    "authors",
    "first_author",
    "summary",
    "summary_word_count",
}


def parse_date(value: str) -> str | None:
    """Convert the source's M/D/YY dates to ISO-8601 dates."""
    if not value.strip():
        return None
    return datetime.strptime(value.strip(), "%m/%d/%y").date().isoformat()


def normalize_row(row: dict[str, str]) -> dict[str, object]:
    summary = " ".join(row["summary"].split())
    paper_id = row["id"].strip()
    return {
        "id": paper_id,
        "source": "arxiv",
        "title": " ".join(row["title"].split()),
        "category": row["category"].strip(),
        "category_code": row["category_code"].strip(),
        "published_date": parse_date(row["published_date"]),
        "updated_date": parse_date(row["updated_date"]),
        "authors_raw": row["authors"].strip(),
        "first_author": row["first_author"].strip(),
        "summary": summary,
        "summary_word_count": len(summary.split()),
        "content_hash": hashlib.sha256(summary.encode("utf-8")).hexdigest(),
    }


def read_rows(input_path: Path) -> Iterator[dict[str, object]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV columns: {names}")

        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Unexpected extra CSV columns at line {line_number}")
            if any(not (row.get(column) or "").strip() for column in REQUIRED_COLUMNS):
                raise ValueError(f"Missing value at line {line_number}")
            yield normalize_row(row)


def ingest(
    input_path: Path, output_path: Path, *, overwrite: bool = False
) -> tuple[int, int]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped_duplicates = 0
    seen_ids: set[str] = set()
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for record in read_rows(input_path):
                if record["id"] in seen_ids:
                    skipped_duplicates += 1
                    continue
                seen_ids.add(str(record["id"]))
                output.write(json.dumps(record, ensure_ascii=True) + "\n")
                count += 1

        if overwrite:
            os.replace(temporary_path, output_path)
        else:
            try:
                os.link(temporary_path, output_path)
            except FileExistsError:
                raise FileExistsError(
                    f"Output already exists: {output_path}. Pass --overwrite to replace it."
                ) from None
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()

    return count, skipped_duplicates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the source CSV, usually under data/raw/corpus/",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path for normalized JSONL, usually under data/processed/",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file after successful ingestion",
    )
    args = parser.parse_args()

    try:
        count, duplicates = ingest(args.input, args.output, overwrite=args.overwrite)
    except (OSError, ValueError, csv.Error) as error:
        print(f"Ingestion failed: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {count:,} normalized records to {args.output}")
    print(f"Skipped duplicate IDs: {duplicates:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
