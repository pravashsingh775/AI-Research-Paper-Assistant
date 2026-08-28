from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROCESSED_DIR / "logs"

for directory in (PROCESSED_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger("dataset_cleaning")

HTML_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"\s+")
BAD_REPLACEMENT_RE = re.compile(r"\ufffd")


def configure_logging() -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        LOG_DIR / "cleaning.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


def clean_text_field(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = HTML_RE.sub(" ", text)
    text = BAD_REPLACEMENT_RE.sub("", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def clean_dataset(input_path: Path, output_path: Path) -> None:
    LOGGER.info("Reading raw dataset from %s", input_path)

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path, low_memory=False, on_bad_lines="warn")
    elif suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(input_path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(input_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    initial_rows = len(df)
    LOGGER.info("Initial row count: %d", initial_rows)

    cols = {str(c).lower(): c for c in df.columns}

    title_col = None
    for k, v in cols.items():
        if any(h in k for h in ["title", "paper_title"]):
            title_col = v
            break

    abstract_col = None
    for k, v in cols.items():
        if any(h in k for h in ["abstract", "summary", "description"]):
            abstract_col = v
            break

    date_col = None
    for k, v in cols.items():
        if any(h in k for h in ["date", "published", "year", "created"]):
            date_col = v
            break

    LOGGER.info("Identified title column: %s", title_col)
    LOGGER.info("Identified abstract column: %s", abstract_col)
    LOGGER.info("Identified date column: %s", date_col)

    # Step 1: Clean text fields
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for col in text_columns:
        df[col] = df[col].apply(clean_text_field)

    # Step 2: Remove rows with empty canonical text
    if title_col and abstract_col:
        valid_mask = (df[title_col] != "") | (df[abstract_col] != "")
        df = df[valid_mask]
        LOGGER.info("Rows after removing empty title and abstract: %d", len(df))
    elif title_col:
        df = df[df[title_col] != ""]
        LOGGER.info("Rows after removing empty title: %d", len(df))

    # Step 3: Remove exact duplicates
    if title_col and abstract_col:
        df = df.drop_duplicates(subset=[title_col, abstract_col], keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    LOGGER.info("Rows after duplicate removal: %d", len(df))

    # Step 4: Standardize dates
    if date_col:
        parsed_dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        df["standardized_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
        LOGGER.info("Standardized date column created.")

    # Step 5: Save processed dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_suffix = output_path.suffix.lower()
    if out_suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    LOGGER.info("Cleaned dataset successfully saved to %s", output_path)
    LOGGER.info("Final row count: %d (removed %d rows)", len(df), initial_rows - len(df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean research paper dataset based on audit findings.")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIR / "arxiv_scientific_dataset.csv",
        help="Path to raw dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "cleaned_arxiv_dataset.csv",
        help="Path to save processed dataset.",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    try:
        clean_dataset(args.input, args.output)
        return 0
    except Exception:
        LOGGER.exception("Dataset cleaning failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())