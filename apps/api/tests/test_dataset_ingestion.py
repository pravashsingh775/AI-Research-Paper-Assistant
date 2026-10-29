from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.ingest_arxiv_csv import REQUIRED_COLUMNS, ingest


def _write_source(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _row(paper_id: str) -> dict[str, str]:
    return {
        "id": paper_id,
        "title": f"Paper {paper_id}",
        "category": "Computer Science",
        "category_code": "cs.AI",
        "published_date": "1/2/24",
        "updated_date": "1/2/24",
        "authors": "['Researcher']",
        "first_author": "'Researcher'",
        "summary": "A useful paper abstract.",
        "summary_word_count": "4",
    }


def test_ingest_refuses_to_replace_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "papers.jsonl"
    output.write_text("keep existing data\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        ingest(tmp_path / "missing.csv", output)

    assert output.read_text(encoding="utf-8") == "keep existing data\n"


def test_ingest_creates_output_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "papers.jsonl"
    _write_source(source, [_row("paper-1")])

    count, duplicates = ingest(source, output)

    assert (count, duplicates) == (1, 0)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["id"] for record in records] == ["paper-1"]
    assert list(tmp_path.glob(".papers.jsonl.*.tmp")) == []


def test_ingest_explicit_overwrite_replaces_only_after_success(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "papers.jsonl"
    output.write_text("prior output\n", encoding="utf-8")
    _write_source(source, [_row("paper-1"), _row("paper-1"), _row("paper-2")])

    count, duplicates = ingest(source, output, overwrite=True)

    assert (count, duplicates) == (2, 1)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["id"] for record in records] == ["paper-1", "paper-2"]


def test_failed_overwrite_preserves_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "papers.jsonl"
    output.write_text("prior output\n", encoding="utf-8")
    _write_source(source, [{**_row("paper-1"), "published_date": "not-a-date"}])

    with pytest.raises(ValueError):
        ingest(source, output, overwrite=True)

    assert output.read_text(encoding="utf-8") == "prior output\n"
    assert list(tmp_path.glob(".papers.jsonl.*.tmp")) == []
