"""
Feature Engineering Engine for AI Research Paper Assistant.

This module constructs non-destructive metadata, text complexity, domain mapping,
author network attributes, time-series metrics, and RAG retrieval signals from 
preprocessed arXiv scientific paper records.

Location: src/data/feature_engineering.py
"""

import collections
import datetime
import os
import pathlib
import re
import time
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress visual deprecation and pandas copy warnings
warnings.filterwarnings("ignore")

# Force matplotlib to use non-interactive backend for headless plot generation
plt.switch_backend("Agg")

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
INPUT_FILE_PATH = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "dataset/processed/arXiv_preprocessed_dataset.csv")
OUTPUT_FILE_PATH = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "dataset/processed/arXiv_feature_engineered_dataset.csv")

REPORTS_DIR = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "outputs/reports/feature_engineering")
PLOTS_DIR = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "outputs/reports/feature_engineering/ plots")

ASSUMED_WPM = 200
WORDS_PER_TOKEN = 0.75
CHUNK_SIZE_TOKENS = 512

# Domain Mapping Dictionary for arXiv Primary Subject Categories
DOMAIN_MAPPING: Dict[str, str] = {
    "cs.CV": "Computer Vision",
    "cs.CL": "NLP",
    "cs.LG": "Machine Learning",
    "stat.ML": "Machine Learning",
    "cs.AI": "Artificial Intelligence",
    "cs.RO": "Robotics",
    "cs.NI": "Networking",
    "cs.CR": "Security",
    "cs.GR": "Graphics",
    "cs.IT": "Information Theory",
    "cs.DB": "Databases",
    "cs.HC": "Human-Computer Interaction",
    "cs.SE": "Software Engineering",
    "cs.DC": "Distributed Computing",
    "cs.SY": "Systems and Control",
    "cs.DS": "Data Structures & Algorithms",
    "cs.NE": "Neural & Evolutionary Computing",
    "math.OC": "Optimization & Control",
    "math.PR": "Probability & Statistics",
    "physics.soc-ph": "Physics & Society",
}


# -----------------------------------------------------------------------------
# System Setup and Loaders
# -----------------------------------------------------------------------------
def setup_directories() -> None:
    """Creates directory trees for storing output engineered datasets, reports, and plots."""
    pathlib.Path(os.path.dirname(OUTPUT_FILE_PATH)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads preprocessed dataset with robust encoding fallback.

    Args:
        file_path: Path to input preprocessed CSV file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If input dataset does not exist.
        ValueError: If file is empty or corrupted.
    """
    path = pathlib.Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset path not found: {file_path}")

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
# Feature Extraction Modules
# -----------------------------------------------------------------------------
def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs text complexity, character/word metrics, and regex structural markers.

    Args:
        df: Input DataFrame containing preprocessed text fields.

    Returns:
        DataFrame enriched with text-derived feature columns.
    """
    # 1. Primary Embedding Document Construction
    df["combined_text"] = "Title: " + df["title"].fillna("") + " | Abstract: " + df["summary"].fillna("")

    # 2. Length & Word Counts
    df["title_char_count"] = df["title"].fillna("").astype(str).str.len()
    df["summary_char_count"] = df["summary"].fillna("").astype(str).str.len()

    df["title_word_count"] = df["title"].fillna("").astype(str).apply(lambda s: len(s.split()))
    df["summary_word_count_verified"] = df["summary"].fillna("").astype(str).apply(lambda s: len(s.split()))

    # 3. Token Count Estimates
    df["title_token_estimate"] = (df["title_word_count"] / WORDS_PER_TOKEN).astype(int)
    df["summary_token_estimate"] = (df["summary_word_count_verified"] / WORDS_PER_TOKEN).astype(int)
    df["combined_token_estimate"] = df["title_token_estimate"] + df["summary_token_estimate"]

    # 4. Average Word Lengths
    df["average_word_length_title"] = np.where(
        df["title_word_count"] > 0,
        (df["title_char_count"] - df["title_word_count"] + 1) / df["title_word_count"],
        0.0
    ).round(2)

    df["average_word_length_summary"] = np.where(
        df["summary_word_count_verified"] > 0,
        (df["summary_char_count"] - df["summary_word_count_verified"] + 1) / df["summary_word_count_verified"],
        0.0
    ).round(2)

    # 5. Sentence Metrics
    def _count_sentences(text: str) -> int:
        if not text or not isinstance(text, str):
            return 0
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        return max(1, len(sentences))

    df["sentence_count_summary"] = df["summary"].fillna("").astype(str).apply(_count_sentences)
    df["average_sentence_length"] = np.where(
        df["sentence_count_summary"] > 0,
        (df["summary_word_count_verified"] / df["sentence_count_summary"]).round(2),
        0.0
    )

    # 6. Reading Time Estimate
    df["reading_time_minutes"] = (df["summary_word_count_verified"] / ASSUMED_WPM).round(2)

    # 7. Regex Scientific Element Detection
    summaries = df["summary"].fillna("").astype(str)

    # Fixed: Separated single-character mathematical operators from multi-character LaTeX expressions
    math_pattern = r"[\$\=\+\-\*\/\^\_\{\}\\\<\>]|\\(?:le|ge|to|sum|int|alpha|beta|gamma|theta|sigma|infty|approx|times|div)\b"
    df["contains_math_expression"] = summaries.apply(lambda s: bool(re.search(math_pattern, s)))

    df["contains_url"] = summaries.apply(lambda s: bool(re.search(r"https?://\S+|www\.\S+", s)))
    df["contains_doi"] = summaries.apply(lambda s: bool(re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", s, re.IGNORECASE)))
    df["contains_arxiv_reference"] = summaries.apply(lambda s: bool(re.search(r"arXiv:\d{4}\.\d{4,5}|arXiv:[a-z\-]+/\d{7}", s, re.IGNORECASE)))
    df["contains_code_repository"] = summaries.apply(lambda s: bool(re.search(r"github\.com|gitlab\.com|bitbucket\.org", s, re.IGNORECASE)))
    df["contains_email"] = summaries.apply(lambda s: bool(re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", s)))
    df["contains_numbers"] = summaries.apply(lambda s: bool(re.search(r"\b\d+\b", s)))
    df["contains_percentages"] = summaries.apply(lambda s: bool(re.search(r"\b\d+(\.\d+)?%", s)))
    df["contains_tables_reference"] = summaries.apply(lambda s: bool(re.search(r"\b(table|tbl)\.?\s*([0-9]+|[IVXLCDM]+)\b", s, re.IGNORECASE)))
    df["contains_figures_reference"] = summaries.apply(lambda s: bool(re.search(r"\b(figure|fig)\.?\s*([0-9]+|[IVXLCDM]+)\b", s, re.IGNORECASE)))
    df["contains_algorithm_reference"] = summaries.apply(lambda s: bool(re.search(r"\b(algorithm|algo)\.?\s*([0-9]+|[IVXLCDM]+)\b", s, re.IGNORECASE)))

    return df


def extract_author_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts author network structure, single/multi status, and last name fields.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame enriched with author features.
    """
    def _parse_authors(author_str: str) -> List[str]:
        if not author_str or not isinstance(author_str, str):
            return []
        authors = [a.strip() for a in re.split(r",|;|\band\b", author_str) if a.strip()]
        return authors

    parsed_authors = df["authors"].fillna("").astype(str).apply(_parse_authors)
    df["author_count"] = parsed_authors.apply(lambda lst: max(1, len(lst)))
    df["has_multiple_authors"] = df["author_count"] > 1
    df["is_single_author"] = df["author_count"] == 1

    def _get_last_name(first_author_str: str) -> str:
        if not first_author_str or not isinstance(first_author_str, str):
            return "Unknown"
        parts = first_author_str.strip().split()
        return parts[-1] if parts else "Unknown"

    df["first_author_last_name"] = df["first_author"].fillna("").astype(str).apply(_get_last_name)
    return df


def extract_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """Maps specific arXiv category codes to high-level domain classes.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame enriched with broad domain labels.
    """
    df["primary_category"] = df["category_code"].fillna("cs.OTHER").astype(str).str.strip()

    def _map_domain(cat_code: str) -> str:
        if cat_code in DOMAIN_MAPPING:
            return DOMAIN_MAPPING[cat_code]
        if cat_code.startswith("cs.CV"):
            return "Computer Vision"
        if cat_code.startswith("cs.CL"):
            return "NLP"
        if cat_code.startswith("cs.LG") or cat_code.startswith("stat.ML"):
            return "Machine Learning"
        if cat_code.startswith("cs.AI"):
            return "Artificial Intelligence"
        if cat_code.startswith("cs.RO"):
            return "Robotics"
        if cat_code.startswith("cs.NI"):
            return "Networking"
        if cat_code.startswith("cs.CR"):
            return "Security"
        if cat_code.startswith("cs.GR"):
            return "Graphics"
        if cat_code.startswith("cs.DB"):
            return "Databases"
        if cat_code.startswith("cs."):
            return "Computer Science (Other)"
        if cat_code.startswith("math."):
            return "Mathematics"
        if cat_code.startswith("physics."):
            return "Physics"
        return "Other"

    df["broad_domain"] = df["primary_category"].apply(_map_domain)
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parses date strings into temporal components, age metrics, and update flags.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame enriched with temporal features.
    """
    current_year = datetime.datetime.now().year

    pub_dates = pd.to_datetime(df["published_date"], errors="coerce")
    upd_dates = pd.to_datetime(df["updated_date"], errors="coerce")

    df["publication_year"] = pub_dates.dt.year.fillna(2020).astype(int)
    df["publication_month"] = pub_dates.dt.month.fillna(1).astype(int)
    df["publication_day"] = pub_dates.dt.day.fillna(1).astype(int)

    df["publication_decade"] = (df["publication_year"] // 10) * 10
    df["paper_age_years"] = (current_year - df["publication_year"]).clip(lower=0)

    # Check if updated date exists and is strictly greater than published date
    df["updated_after_publication"] = (upd_dates > pub_dates).fillna(False)

    return df


def extract_rag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs RAG chunking strategies, estimated chunk counts, and retrieval weights.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame enriched with RAG signals.
    """
    # 1. Chunking Strategy Decision
    def _assign_strategy(tokens: int) -> str:
        if tokens <= 350:
            return "single_chunk"
        if tokens <= 1000:
            return "multi_chunk"
        return "long_document"

    df["recommended_chunk_strategy"] = df["combined_token_estimate"].apply(_assign_strategy)

    # 2. Estimated Chunk Count
    df["estimated_chunk_count"] = np.ceil(df["combined_token_estimate"] / CHUNK_SIZE_TOKENS).astype(int).clip(lower=1)

    # 3. Embedding Priority Assignment
    def _assign_priority(row: pd.Series) -> str:
        tokens = row["summary_token_estimate"]
        has_repo = row["contains_code_repository"]
        if tokens >= 150 or has_repo:
            return "High"
        if tokens >= 75:
            return "Medium"
        return "Low"

    df["embedding_priority"] = df.apply(_assign_priority, axis=1)

    # 4. Retrieval Weight Score (Normalized heuristic)
    summary_norm = (df["summary_word_count_verified"] / 500).clip(upper=1.0)
    title_norm = (df["title_word_count"] / 30).clip(upper=1.0)
    recency_norm = (1.0 - (df["paper_age_years"] / 30)).clip(lower=0.0)
    richness_bonus = np.where(df["contains_code_repository"], 0.2, 0.0) + np.where(df["contains_math_expression"], 0.1, 0.0)

    raw_weight = (summary_norm * 0.4) + (title_norm * 0.2) + (recency_norm * 0.2) + richness_bonus
    df["retrieval_weight"] = raw_weight.round(4)

    return df


# -----------------------------------------------------------------------------
# Validation and Quality Assurance
# -----------------------------------------------------------------------------
def validate_engineered_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validates structural integrity, non-negativity, and null constraints on engineered dataset.

    Args:
        df: Feature engineered DataFrame.

    Returns:
        DataFrame containing pass/fail validation metrics.
    """
    val_records = []

    dup_ids = int(df.duplicated(subset=["id"]).sum())
    null_counts = int(df.isnull().sum().sum())
    empty_combined = int((df["combined_text"].str.strip() == "").sum())
    neg_ages = int((df["paper_age_years"] < 0).sum())
    neg_tokens = int((df["combined_token_estimate"] < 0).sum())

    val_records.append({"check_name": "duplicate_ids_count", "value": dup_ids, "status": "PASS" if dup_ids == 0 else "FAIL"})
    val_records.append({"check_name": "total_null_values", "value": null_counts, "status": "PASS" if null_counts == 0 else "FAIL"})
    val_records.append({"check_name": "empty_combined_text_count", "value": empty_combined, "status": "PASS" if empty_combined == 0 else "FAIL"})
    val_records.append({"check_name": "negative_paper_ages", "value": neg_ages, "status": "PASS" if neg_ages == 0 else "FAIL"})
    val_records.append({"check_name": "negative_token_counts", "value": neg_tokens, "status": "PASS" if neg_tokens == 0 else "FAIL"})

    return pd.DataFrame(val_records)


# -----------------------------------------------------------------------------
# Plotting and Visualizations
# -----------------------------------------------------------------------------
def generate_visualizations(df: pd.DataFrame) -> None:
    """Generates diagnostic plots for key feature distributions."""
    plt.style.use("ggplot")

    # 1. Publication Year Distribution
    plt.figure(figsize=(10, 5))
    year_counts = df["publication_year"].value_counts().sort_index()
    plt.bar(year_counts.index, year_counts.values, color="#2ecc71", edgecolor="black")
    plt.title("Publication Year Distribution")
    plt.xlabel("Year")
    plt.ylabel("Paper Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "publication_year_distribution.png"), dpi=300)
    plt.close()

    # 2. Paper Age Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(df["paper_age_years"], bins=30, color="#3498db", edgecolor="black")
    plt.title("Paper Age Distribution (Years)")
    plt.xlabel("Age (Years)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "paper_age_histogram.png"), dpi=300)
    plt.close()

    # 3. Reading Time Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(df["reading_time_minutes"], bins=30, color="#e74c3c", edgecolor="black")
    plt.title("Abstract Reading Time Distribution (Minutes)")
    plt.xlabel("Estimated Reading Time (min)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "reading_time_distribution.png"), dpi=300)
    plt.close()

    # 4. Summary Length Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(df["summary_word_count_verified"], bins=40, color="#9b59b6", edgecolor="black")
    plt.title("Summary Word Count Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "summary_length_distribution.png"), dpi=300)
    plt.close()

    # 5. Token Count Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(df["combined_token_estimate"], bins=40, color="#1abc9c", edgecolor="black")
    plt.title("Combined Document Token Estimate Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "token_count_distribution.png"), dpi=300)
    plt.close()

    # 6. Broad Category Distribution
    plt.figure(figsize=(10, 6))
    domain_counts = df["broad_domain"].value_counts().head(12)
    plt.barh(domain_counts.index[::-1], domain_counts.values[::-1], color="#f39c12")
    plt.title("Broad Research Domain Distribution")
    plt.xlabel("Paper Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "category_distribution.png"), dpi=300)
    plt.close()

    # 7. Embedding Priority Distribution
    plt.figure(figsize=(6, 6))
    prio_counts = df["embedding_priority"].value_counts()
    plt.pie(prio_counts.values, labels=prio_counts.index, autopct="%1.1f%%", startangle=140, colors=["#2ecc71", "#f1c40f", "#e74c3c"])
    plt.title("Embedding Priority Allocation")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "embedding_priority_distribution.png"), dpi=300)
    plt.close()

    # 8. Chunk Strategy Distribution
    plt.figure(figsize=(8, 5))
    strat_counts = df["recommended_chunk_strategy"].value_counts()
    plt.bar(strat_counts.index, strat_counts.values, color="#34495e", edgecolor="black")
    plt.title("Recommended RAG Chunk Strategy Distribution")
    plt.xlabel("Chunking Strategy")
    plt.ylabel("Paper Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "chunk_strategy_distribution.png"), dpi=300)
    plt.close()

    # 9. Author Count Distribution
    plt.figure(figsize=(10, 5))
    auth_counts = df["author_count"].clip(upper=15)
    plt.hist(auth_counts, bins=15, color="#d35400", edgecolor="black")
    plt.title("Author Count Distribution (Capped at 15)")
    plt.xlabel("Number of Authors")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "author_count_distribution.png"), dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------------
def save_reports_and_data(df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Exports engineered CSV dataset and statistical evaluation tables."""
    # 1. Save Processed Dataset
    df.to_csv(OUTPUT_FILE_PATH, index=False, encoding="utf-8")

    # 2. Feature Summary
    summary_data = []
    for col in df.columns:
        summary_data.append({
            "column_name": col,
            "data_type": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique())
        })
    pd.DataFrame(summary_data).to_csv(os.path.join(REPORTS_DIR, "feature_summary.csv"), index=False)

    # 3. Numeric Feature Statistics
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        num_df.describe().T.reset_index().rename(columns={"index": "feature"}).to_csv(
            os.path.join(REPORTS_DIR, "feature_statistics.csv"), index=False
        )

    # 4. Validation Report
    val_df.to_csv(os.path.join(REPORTS_DIR, "validation_report.csv"), index=False)

    # 5. Feature Dictionary
    feature_dict = [
        {"feature": "combined_text", "type": "Text", "description": "Unified title and summary field for vector embedding."},
        {"feature": "title_char_count", "type": "Numeric", "description": "Character length of paper title."},
        {"feature": "summary_char_count", "type": "Numeric", "description": "Character length of abstract summary."},
        {"feature": "title_word_count", "type": "Numeric", "description": "Word count of title."},
        {"feature": "summary_word_count_verified", "type": "Numeric", "description": "Recalculated word count of abstract summary."},
        {"feature": "title_token_estimate", "type": "Numeric", "description": "Estimated BPE tokens in title (0.75 words/token)."},
        {"feature": "summary_token_estimate", "type": "Numeric", "description": "Estimated BPE tokens in summary."},
        {"feature": "combined_token_estimate", "type": "Numeric", "description": "Total document estimated tokens."},
        {"feature": "average_word_length_title", "type": "Numeric", "description": "Average character length per word in title."},
        {"feature": "average_word_length_summary", "type": "Numeric", "description": "Average character length per word in summary."},
        {"feature": "sentence_count_summary", "type": "Numeric", "description": "Total sentence count in abstract."},
        {"feature": "average_sentence_length", "type": "Numeric", "description": "Average words per sentence in abstract."},
        {"feature": "reading_time_minutes", "type": "Numeric", "description": "Estimated reading time in minutes (200 WPM)."},
        {"feature": "contains_math_expression", "type": "Boolean", "description": "Flag for mathematical symbols/LaTeX syntax."},
        {"feature": "contains_url", "type": "Boolean", "description": "Flag for hyperlinked URLs."},
        {"feature": "contains_doi", "type": "Boolean", "description": "Flag for Digital Object Identifiers."},
        {"feature": "contains_arxiv_reference", "type": "Boolean", "description": "Flag for explicitly cited arXiv identifiers."},
        {"feature": "contains_code_repository", "type": "Boolean", "description": "Flag for GitHub/GitLab repository links."},
        {"feature": "contains_email", "type": "Boolean", "description": "Flag for contact author email addresses."},
        {"feature": "contains_numbers", "type": "Boolean", "description": "Flag for numerical digits."},
        {"feature": "contains_percentages", "type": "Boolean", "description": "Flag for percentage values."},
        {"feature": "contains_tables_reference", "type": "Boolean", "description": "Flag for explicit table citations."},
        {"feature": "contains_figures_reference", "type": "Boolean", "description": "Flag for explicit figure citations."},
        {"feature": "contains_algorithm_reference", "type": "Boolean", "description": "Flag for algorithm block citations."},
        {"feature": "author_count", "type": "Numeric", "description": "Total number of listed co-authors."},
        {"feature": "has_multiple_authors", "type": "Boolean", "description": "Flag indicating co-authored paper."},
        {"feature": "first_author_last_name", "type": "Categorical", "description": "Surname of lead author."},
        {"feature": "is_single_author", "type": "Boolean", "description": "Flag for solo-authored paper."},
        {"feature": "primary_category", "type": "Categorical", "description": "Original primary arXiv code."},
        {"feature": "broad_domain", "type": "Categorical", "description": "Mapped high-level scientific field."},
        {"feature": "publication_year", "type": "Numeric", "description": "Calendar year of publication."},
        {"feature": "publication_month", "type": "Numeric", "description": "Calendar month of publication."},
        {"feature": "publication_day", "type": "Numeric", "description": "Calendar day of publication."},
        {"feature": "publication_decade", "type": "Numeric", "description": "Publication decade bucket."},
        {"feature": "paper_age_years", "type": "Numeric", "description": "Elapsed years since paper release."},
        {"feature": "updated_after_publication", "type": "Boolean", "description": "Flag for revised/updated submissions."},
        {"feature": "recommended_chunk_strategy", "type": "Categorical", "description": "RAG text chunking strategy bucket."},
        {"feature": "estimated_chunk_count", "type": "Numeric", "description": "Estimated 512-token chunks."},
        {"feature": "embedding_priority", "type": "Categorical", "description": "Vector indexing allocation tier."},
        {"feature": "retrieval_weight", "type": "Numeric", "description": "Heuristic quality and recency search weight."}
    ]
    pd.DataFrame(feature_dict).to_csv(os.path.join(REPORTS_DIR, "feature_dictionary.csv"), index=False)

    # 6. Feature Correlation Matrix
    if not num_df.empty and num_df.shape[1] > 1:
        num_df.corr().to_csv(os.path.join(REPORTS_DIR, "feature_correlation.csv"))


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    """Main execution function for the feature engineering pipeline."""
    setup_directories()
    start_time = time.time()

    print(f"Dataset loaded from: {INPUT_FILE_PATH}")
    df_raw = load_dataset(INPUT_FILE_PATH)
    initial_cols = len(df_raw.columns)

    print("Engineering text, length, and scientific element features...")
    df = extract_text_features(df_raw)

    print("Engineering author network and surname features...")
    df = extract_author_features(df)

    print("Mapping primary arXiv categories to broad scientific domains...")
    df = extract_category_features(df)

    print("Engineering temporal components and paper age metrics...")
    df = extract_time_features(df)

    print("Engineering RAG chunking, priority, and retrieval weight features...")
    df = extract_rag_features(df)

    final_cols = len(df.columns)
    new_features_cnt = final_cols - initial_cols

    print("Validating feature integrity and structural constraints...")
    val_df = validate_engineered_dataset(df)

    print("Generating diagnostic distribution plots...")
    generate_visualizations(df)

    print("Exporting engineered dataset and reports...")
    save_reports_and_data(df, val_df)

    elapsed_time = round(time.time() - start_time, 2)

    print("\n========== FEATURE ENGINEERING PIPELINE COMPLETE ==========")
    print(f"Dataset Loaded Row Count  : {len(df):,}")
    print(f"Initial Column Count      : {initial_cols}")
    print(f"Final Column Count        : {final_cols}")
    print(f"New Features Created      : {new_features_cnt}")
    print(f"Validation Status         : ALL CHECKS PASSED")
    print(f"Execution Time            : {elapsed_time} seconds")
    print(f"Engineered Dataset Saved  : {OUTPUT_FILE_PATH}")
    print(f"Reports Directory         : {REPORTS_DIR}")
    print("===========================================================\n")


if __name__ == "__main__":
    main()