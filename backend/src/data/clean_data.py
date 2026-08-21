import os
import pathlib
import re
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Path Configurations
RAW_DATA_PATH = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "dataset/raw/arXiv_scientific_dataset.csv")
PROCESSED_DIR = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "dataset/processed")
OUTPUT_FILE_PATH = os.path.join(PROCESSED_DIR, "arXiv_cleaned_dataset.csv")
REPORTS_DIR = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "outputs/reports/Cleaning")


def setup_directories() -> None:
    """Creates directory structure for processed data and cleaning metrics logs."""
    pathlib.Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)


def load_raw_dataset(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads raw dataset using high-throughput parsing."""
    path = pathlib.Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset path does not exist: {file_path}")

    print(f"Reading raw data from: {file_path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError("Loaded raw dataset is empty.")
    return df


def sanitize_text(text: str) -> str:
    """Normalizes string inputs by stripping HTML, URLs, control chars, and multi-spaces."""
    if not isinstance(text, str):
        return ""

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove unprintable ASCII control characters
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    # Replace tabs and carriage returns with space
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Collapse redundant white spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def sanitize_category(category: str) -> str:
    """Standardizes subject domain string entries."""
    if not isinstance(category, str):
        return "Unspecified"
    cat_clean = sanitize_text(category)
    return cat_clean if len(cat_clean) > 0 else "Unspecified"


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parses date strings into standardized ISO 8601 YYYY-MM-DD representations."""
    date_columns = ["published_date", "updated_date"]

    for col in date_columns:
        if col in df.columns:
            # Parse to datetime object; coercing invalid inputs to NaT
            parsed_dates = pd.to_datetime(df[col], errors="coerce")
            df[col] = parsed_dates.dt.strftime("%Y-%m-%d")
            # Fill missing/invalid parsing with default fallback
            df[col] = df[col].fillna("1970-01-01")

    return df


def execute_cleaning_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Executes data cleaning, validation guardrails, and feature engineering."""
    initial_rows = len(df)
    stats = {"initial_rows": initial_rows}

    # 1. Deduplication on Primary Key ('id')
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    dedup_rows = len(df)
    stats["duplicates_removed"] = initial_rows - dedup_rows

    # 2. Text Normalization
    text_cols = ["title", "summary", "authors", "first_author"]
    for col in text_cols:
        if col in df.columns:
            print(f"Sanitizing text field: {col}...")
            df[col] = df[col].astype(str).apply(sanitize_text)

    if "category" in df.columns:
        df["category"] = df["category"].astype(str).apply(sanitize_category)

    # 3. Date Normalization
    df = clean_dates(df)

    # 4. Filter Quality Guardrails (Drop records missing essential core fields)
    df = df[(df["title"].str.len() >= 3) & (df["summary"].str.len() >= 15)].reset_index(drop=True)
    stats["invalid_rows_dropped"] = dedup_rows - len(df)

    # 5. Feature Engineering
    df["summary_word_count"] = df["summary"].apply(lambda text: len(text.split()))
    df["summary_char_len"] = df["summary"].str.len()
    df["title_char_len"] = df["title"].str.len()

    # Construct primary text context field optimized for RAG / Embedding models
    df["search_text"] = "Title: " + df["title"] + " | Abstract: " + df["summary"]

    stats["final_clean_rows"] = len(df)
    return df, stats


def save_cleaning_report(stats: dict) -> None:
    """Outputs cleaning performance metrics to CSV log."""
    report_df = pd.DataFrame([stats])
    report_path = os.path.join(REPORTS_DIR, "cleaning_summary.csv")
    report_df.to_csv(report_path, index=False)


def main() -> None:
    """Main execution function."""
    setup_directories()

    start_time = time.time()
    df_raw = load_raw_dataset(RAW_DATA_PATH)

    print("Executing dataset cleaning and validation pipeline...")
    df_clean, stats = execute_cleaning_pipeline(df_raw)

    print(f"Saving cleaned dataset to: {OUTPUT_FILE_PATH}")
    df_clean.to_csv(OUTPUT_FILE_PATH, index=False, encoding="utf-8")

    save_cleaning_report(stats)

    elapsed_time = round(time.time() - start_time, 2)

    print("\n========== DATASET CLEANING COMPLETE ==========")
    print(f"Initial Row Count       : {stats['initial_rows']:,}")
    print(f"Duplicates Removed      : {stats['duplicates_removed']:,}")
    print(f"Invalid Rows Dropped    : {stats['invalid_rows_dropped']:,}")
    print(f"Final Cleaned Row Count : {stats['final_clean_rows']:,}")
    print(f"Execution Time          : {elapsed_time} seconds")
    print(f"Clean Data File Path    : {OUTPUT_FILE_PATH}")
    print("================================================\n")


if __name__ == "__main__":
    main()