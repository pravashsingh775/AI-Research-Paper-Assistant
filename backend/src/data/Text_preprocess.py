"""Scientific Text Preprocessing Pipeline for AI Research Paper Assistant.

This module provides a modular, domain-specific text preprocessing pipeline designed 
for arXiv scientific papers. It enforces strict normalization while preserving 
technical terminology, mathematical notation, model names, equations, code syntax, 
and academic identifiers necessary for downstream Semantic Search and RAG.

Location: src/data/preprocess.py
"""

import collections
import html
import os
import pathlib
import re
import time
import unicodedata
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration and Path Definitions
# -----------------------------------------------------------------------------
INPUT_FILE_PATH = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "dataset/processed/arXiv_cleaned_dataset.csv")
OUTPUT_FILE_PATH = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "dataset/processed/arXiv_preprocessed_dataset.csv")

REPORTS_DIR = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "outputs/reports/preprocessing")

TEXT_COLUMNS = ["title", "summary", "authors", "first_author"]


# -----------------------------------------------------------------------------
# Directory Setup & Dataset Loader
# -----------------------------------------------------------------------------
def setup_directories() -> None:
    """Creates directory structures for output preprocessed datasets and reports."""
    pathlib.Path(os.path.dirname(OUTPUT_FILE_PATH)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads CSV dataset with fallback encoding mechanisms.

    Args:
        file_path: Path to the input CSV file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If file is empty or cannot be parsed.
    """
    path = pathlib.Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset file not found: {file_path}")

    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            if df.empty:
                raise ValueError(f"Dataset at '{file_path}' is empty.")
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    raise ValueError(f"Failed to load dataset at '{file_path}' with supported encodings.")


# -----------------------------------------------------------------------------
# Preprocessing Pipeline Units
# -----------------------------------------------------------------------------
def normalize_unicode(text: str) -> str:
    """Applies NFKC Unicode normalization.

    Args:
        text: Input string.

    Returns:
        NFKC normalized string.
    """
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def remove_invisible_characters(text: str) -> str:
    """Strips zero-width spaces, unprintable control characters, and Unicode garbage.

    Args:
        text: Input string.

    Returns:
        String stripped of invisible control characters.
    """
    if not isinstance(text, str):
        return ""
    # Strip zero-width characters and ASCII control chars (excluding space)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    return text


def decode_html_entities(text: str) -> str:
    """Decodes HTML entities (e.g., &amp;, &lt;, &gt;).

    Args:
        text: Input string.

    Returns:
        Decoded text string.
    """
    if not isinstance(text, str):
        return ""
    return html.unescape(text)


def remove_html_tags(text: str) -> str:
    """Strips structural HTML tags while preserving body content.

    Args:
        text: Input string.

    Returns:
        String with HTML markup removed.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"<[^>]+>", " ", text)


def normalize_quotation_marks(text: str) -> str:
    """Converts directional/curly quotes to standard straight quotes.

    Args:
        text: Input string.

    Returns:
        Text with normalized double quotation marks.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\u201C\u201D\u201E\u201F\u2033\u2036]", '"', text)


def normalize_apostrophes(text: str) -> str:
    """Converts curly single quotes and backticks to standard apostrophes.

    Args:
        text: Input string.

    Returns:
        Text with normalized single quotation marks and apostrophes.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\u2018\u2019\u201A\u201B\u2032\u2035`]", "'", text)


def normalize_dashes(text: str) -> str:
    """Normalizes em-dashes, en-dashes, and figure dashes to standard hyphens.

    Args:
        text: Input string.

    Returns:
        Text with standardized dash characters.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\u2012\u2013\u2014\u2015\u2212]", "-", text)


def normalize_bullet_characters(text: str) -> str:
    """Removes or standardizes bullet point symbols.

    Args:
        text: Input string.

    Returns:
        Text with normalized list bullet symbols.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\u2022\u2023\u2043\u204C\u204D\u2219\u25CB\u25CF\u25E6\u25A0]", " ", text)


def normalize_scientific_punctuation(text: str) -> str:
    """Standardizes mathematical and scientific punctuation marks.

    Args:
        text: Input string.

    Returns:
        Text with normalized scientific notation symbols.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\u2026", "...", text)
    text = re.sub(r"\u00A0", " ", text)
    return text


def sanitize_urls_and_links(text: str) -> str:
    """Removes malformed URLs while preserving DOI, GitHub, and arXiv links.

    Args:
        text: Input string.

    Returns:
        Text with malformed URLs cleaned and academic links intact.
    """
    if not isinstance(text, str):
        return ""

    def _preserve_or_strip(match: re.Match) -> str:
        url = match.group(0)
        # Preserve academic URLs, DOIs, GitHub repositories, and arXiv permalinks
        if re.search(r"(doi\.org|github\.com|arxiv\.org|gitlab\.com|bitbucket\.org)", url, re.IGNORECASE):
            return url
        # Remove broken or general tracking URLs
        if re.search(r"http[s]?://\S*(?:undefined|null|void|javascript|%00)", url, re.IGNORECASE):
            return " "
        return url

    return re.sub(r"https?://\S+|www\.\S+", _preserve_or_strip, text)


def sanitize_email_addresses(text: str) -> str:
    """Removes malformed or corrupted email strings.

    Args:
        text: Input string.

    Returns:
        Text with invalid emails removed.
    """
    if not isinstance(text, str):
        return ""

    def _filter_email(match: re.Match) -> str:
        email = match.group(0)
        if re.search(r"(undefined|null|invalid|@localhost)", email, re.IGNORECASE):
            return " "
        return email

    return re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", _filter_email, text)


def normalize_whitespace(text: str) -> str:
    """Collapses spaces, tabs, and linebreaks into single spaces and trims bounds.

    Args:
        text: Input string.

    Returns:
        Cleaned, single-spaced string.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_author_names(text: str) -> str:
    """Normalizes author formatting and spacing without altering name casing/spelling.

    Args:
        text: Author string.

    Returns:
        Normalized author string.
    """
    if not isinstance(text, str):
        return ""
    text = decode_html_entities(text)
    text = normalize_unicode(text)
    text = remove_invisible_characters(text)
    text = normalize_apostrophes(text)
    text = normalize_dashes(text)
    text = normalize_whitespace(text)
    return text


def preprocess_scientific_text(text: str) -> str:
    """Applies complete non-destructive scientific preprocessing pipeline to a string.

    Preserves mathematical syntax (+, -, *, /, =, ^, (), [], {}), decimal numbers, 
    years, software/model version numbers (e.g., Python 3.11, YOLOv8, GPT-4), 
    chemical/biological terms, code constructs, and technical abbreviations.

    Args:
        text: Input scientific text string.

    Returns:
        Fully preprocessed string.
    """
    if not isinstance(text, str):
        return ""

    text = normalize_unicode(text)
    text = remove_invisible_characters(text)
    text = decode_html_entities(text)
    text = remove_html_tags(text)
    text = normalize_quotation_marks(text)
    text = normalize_apostrophes(text)
    text = normalize_dashes(text)
    text = normalize_bullet_characters(text)
    text = normalize_scientific_punctuation(text)
    text = sanitize_urls_and_links(text)
    text = sanitize_email_addresses(text)
    text = normalize_whitespace(text)

    return text


# -----------------------------------------------------------------------------
# DataFrame Execution Engine
# -----------------------------------------------------------------------------
def process_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Applies preprocessing functions across designated DataFrame columns.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Tuple containing preprocessed DataFrame and operational statistics dictionary.
    """
    initial_rows = len(df)
    stats = {"initial_rows": initial_rows}

    # Process Title Column
    if "title" in df.columns:
        print("Preprocessing column: 'title'...")
        df["title"] = df["title"].astype(str).apply(preprocess_scientific_text)

    # Process Summary/Abstract Column
    if "summary" in df.columns:
        print("Preprocessing column: 'summary'...")
        df["summary"] = df["summary"].astype(str).apply(preprocess_scientific_text)

    # Process Author Columns
    if "authors" in df.columns:
        print("Preprocessing column: 'authors'...")
        df["authors"] = df["authors"].astype(str).apply(normalize_author_names)

    if "first_author" in df.columns:
        print("Preprocessing column: 'first_author'...")
        df["first_author"] = df["first_author"].astype(str).apply(normalize_author_names)

    # Recalculate word counts and character length metrics
    if "summary" in df.columns:
        df["summary_word_count"] = df["summary"].apply(lambda s: len(s.split()))

    # Re-construct unified search field for RAG vectorization
    if "title" in df.columns and "summary" in df.columns:
        df["search_text"] = "Title: " + df["title"] + " | Abstract: " + df["summary"]

    # Filter out records failing minimal quality threshold
    print("Executing post-preprocessing validation guardrails...")
    df = df[(df["title"].str.len() >= 3) & (df["summary"].str.len() >= 15)].reset_index(drop=True)

    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    stats["final_rows"] = len(df)
    stats["removed_rows"] = initial_rows - len(df)

    return df, stats


# -----------------------------------------------------------------------------
# Validation and Reporting
# -----------------------------------------------------------------------------
def validate_preprocessed_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validates data integrity post-preprocessing.

    Args:
        df: Preprocessed DataFrame.

    Returns:
        Validation check metrics as a DataFrame.
    """
    checks = []

    null_count = int(df.isnull().sum().sum())
    empty_titles = int((df["title"].str.strip() == "").sum()) if "title" in df.columns else 0
    empty_summaries = int((df["summary"].str.strip() == "").sum()) if "summary" in df.columns else 0
    duplicate_ids = int(df.duplicated(subset=["id"]).sum()) if "id" in df.columns else 0

    checks.append({"check_name": "total_null_values", "value": null_count, "status": "PASS" if null_count == 0 else "FAIL"})
    checks.append({"check_name": "empty_title_count", "value": empty_titles, "status": "PASS" if empty_titles == 0 else "FAIL"})
    checks.append({"check_name": "empty_summary_count", "value": empty_summaries, "status": "PASS" if empty_summaries == 0 else "FAIL"})
    checks.append({"check_name": "duplicate_id_count", "value": duplicate_ids, "status": "PASS" if duplicate_ids == 0 else "FAIL"})

    return pd.DataFrame(checks)


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates statistical distributions of word and character metrics.

    Args:
        df: Preprocessed DataFrame.

    Returns:
        Summary statistics DataFrame.
    """
    stats_list = []

    for col in ["title", "summary", "search_text"]:
        if col in df.columns:
            char_lens = df[col].str.len()
            word_counts = df[col].apply(lambda s: len(s.split()))

            stats_list.append({
                "column_name": col,
                "avg_char_length": round(float(char_lens.mean()), 2),
                "median_char_length": float(char_lens.median()),
                "p95_char_length": float(np.percentile(char_lens, 95)),
                "avg_word_count": round(float(word_counts.mean()), 2),
                "median_word_count": float(word_counts.median()),
                "p95_word_count": float(np.percentile(word_counts, 95)),
            })

    return pd.DataFrame(stats_list)


def save_reports(df: pd.DataFrame, stats: Dict[str, int], val_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Saves validation and statistical reports to target output paths.

    Args:
        df: Preprocessed DataFrame.
        stats: Pipeline execution statistics.
        val_df: Validation DataFrame.
        summary_df: Summary statistics DataFrame.
    """
    # 1. Pipeline execution report
    prep_report = pd.DataFrame([stats])
    prep_report.to_csv(os.path.join(REPORTS_DIR, "preprocessing_report.csv"), index=False)

    # 2. Validation report
    val_df.to_csv(os.path.join(REPORTS_DIR, "validation_report.csv"), index=False)

    # 3. Summary statistics
    summary_df.to_csv(os.path.join(REPORTS_DIR, "summary_statistics.csv"), index=False)


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    """Main function executing the scientific text preprocessing pipeline."""
    setup_directories()

    start_time = time.time()
    print(f"Loading cleaned dataset from: {INPUT_FILE_PATH}")
    df_raw = load_dataset(INPUT_FILE_PATH)

    print("Executing scientific text preprocessing pipeline...")
    df_processed, stats = process_dataframe(df_raw)

    print(f"Saving preprocessed dataset to: {OUTPUT_FILE_PATH}")
    df_processed.to_csv(OUTPUT_FILE_PATH, index=False, encoding="utf-8")

    print("Validating preprocessed dataset integrity...")
    val_df = validate_preprocessed_data(df_processed)
    summary_df = generate_summary_statistics(df_processed)

    save_reports(df_processed, stats, val_df, summary_df)

    elapsed_time = round(time.time() - start_time, 2)

    print("\n========== PREPROCESSING PIPELINE COMPLETE ==========")
    print(f"Rows Before Preprocessing : {stats['initial_rows']:,}")
    print(f"Rows After Preprocessing  : {stats['final_rows']:,}")
    print(f"Invalid Rows Removed      : {stats['removed_rows']:,}")
    print(f"Execution Time            : {elapsed_time} seconds")
    print(f"Columns Processed         : {', '.join(TEXT_COLUMNS)}")
    print(f"Output Dataset Saved To   : {OUTPUT_FILE_PATH}")
    print(f"Reports Directory         : {REPORTS_DIR}")
    print("=====================================================\n")


if __name__ == "__main__":
    main()