from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "id",
    "title",
    "category",
    "category_code",
    "published_date",
    "updated_date",
    "authors",
    "first_author",
    "summary",
)


class DatasetLoaderError(Exception):
    """Base exception for dataset loading failures."""


class MissingColumnsError(DatasetLoaderError):
    """Raised when required dataset columns are missing."""


class DatasetFormatError(DatasetLoaderError):
    """Raised when the dataset format is invalid."""


def _validate_file(csv_path: Path) -> None:
    """Validate that the input path points to a readable CSV file."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {csv_path}"
        )

    if not csv_path.is_file():
        raise DatasetLoaderError(
            f"Dataset path is not a file: {csv_path}"
        )

    if csv_path.suffix.lower() != ".csv":
        raise DatasetFormatError(
            f"Expected a CSV file, received: {csv_path.suffix}"
        )

    try:
        with csv_path.open("rb") as file:
            file.read(1)
    except OSError as exc:
        raise DatasetLoaderError(
            f"Dataset is not readable: {csv_path}"
        ) from exc


def _validate_schema(df: pd.DataFrame) -> None:
    """Validate that all required columns exist."""

    if not isinstance(df, pd.DataFrame):
        raise DatasetFormatError(
            "Pandas did not return a valid DataFrame."
        )

    # Remove accidental whitespace from column names.
    df.columns = df.columns.astype(str).str.strip()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise MissingColumnsError(
            "Missing required columns: "
            f"{missing_columns}. "
            f"Expected columns: {list(REQUIRED_COLUMNS)}"
        )


def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load and perform basic schema validation on the raw research-paper CSV.

    This function intentionally does NOT:
    - clean scientific text
    - remove duplicates
    - modify values
    - create search_text
    - normalize dates
    - create embeddings
    - build vector indexes

    Those responsibilities belong to later pipeline stages.
    """

    csv_path = Path(csv_path)

    _validate_file(csv_path)

    LOGGER.info("Loading raw dataset: %s", csv_path)

    try:
        df = pd.read_csv(
            csv_path,
            encoding="utf-8-sig",
            sep=",",
            low_memory=False,
            on_bad_lines="error",
        )

    except pd.errors.EmptyDataError as exc:
        raise DatasetFormatError(
            f"Dataset is empty: {csv_path}"
        ) from exc

    except pd.errors.ParserError as exc:
        raise DatasetFormatError(
            f"CSV parsing failed for {csv_path}: {exc}"
        ) from exc

    except UnicodeDecodeError as exc:
        raise DatasetFormatError(
            f"Dataset encoding is not valid UTF-8-SIG: {csv_path}"
        ) from exc

    except OSError as exc:
        raise DatasetLoaderError(
            f"Unable to read dataset: {csv_path}"
        ) from exc

    except Exception as exc:
        raise DatasetLoaderError(
            f"Unexpected error while loading dataset: {exc}"
        ) from exc

    _validate_schema(df)

    LOGGER.info(
        "Dataset loaded successfully: %d rows × %d columns",
        len(df),
        len(df.columns),
    )

    LOGGER.info(
        "Dataset memory usage: %.2f MB",
        df.memory_usage(deep=True).sum() / (1024**2),
    )

    LOGGER.info(
        "Required columns verified successfully."
    )

    return df