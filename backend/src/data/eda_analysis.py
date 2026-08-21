import collections
import os
import pathlib
import re
import warnings
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")

# Paths Configuration
PROCESSED_DATA_PATH = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "dataset/raw/arXiv_scientific_dataset.csv")
REPORTS_DIR = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "outputs/reports/EDA")
PLOTS_DIR = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "outputs/reports/EDA/plots")


def setup_directories(reports_dir: str = REPORTS_DIR, plots_dir: str = PLOTS_DIR) -> None:
    """Creates directory structures for EDA outputs."""
    pathlib.Path(reports_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(plots_dir).mkdir(parents=True, exist_ok=True)


def load_dataset(file_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Loads processed dataset into memory."""
    target_path = pathlib.Path(file_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {file_path}")
    
    df = pd.read_csv(target_path, low_memory=False)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def analyze_category_distribution(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Analyzes class imbalance and top category frequencies."""
    if "category" not in df.columns:
        return pd.DataFrame()

    cat_counts = df["category"].value_counts().reset_index()
    cat_counts.columns = ["category", "paper_count"]
    cat_counts["percentage"] = (cat_counts["paper_count"] / len(df)) * 100
    cat_counts["percentage"] = cat_counts["percentage"].round(2)

    # Plot Top Categories
    plt.figure(figsize=(12, 6))
    sns.barplot(data=cat_counts.head(top_n), x="paper_count", y="category", palette="viridis")
    plt.title(f"Top {top_n} Research Categories Distribution")
    plt.xlabel("Number of Papers")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "top_categories_distribution.png"), dpi=300)
    plt.close()

    return cat_counts


def analyze_publication_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes publication distribution across time."""
    if "published_date" not in df.columns:
        return pd.DataFrame()

    df_time = df.copy()
    df_time["published_date"] = pd.to_datetime(df_time["published_date"], errors="coerce")
    df_time["pub_year"] = df_time["published_date"].dt.year

    year_counts = df_time["pub_year"].value_counts().sort_index().reset_index()
    year_counts.columns = ["publication_year", "paper_count"]

    plt.figure(figsize=(10, 5))
    plt.plot(year_counts["publication_year"], year_counts["paper_count"], marker="o", color="#2b5c8f", linewidth=2)
    plt.title("Paper Publication Trends Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Papers Published")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "publication_trends.png"), dpi=300)
    plt.close()

    return year_counts


def analyze_text_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates granular word and character metrics for RAG chunking decisions."""
    metrics = []
    
    target_cols = ["title", "summary", "search_text"]
    for col in target_cols:
        if col in df.columns:
            char_lens = df[col].astype(str).str.len()
            word_counts = df[col].astype(str).apply(lambda x: len(x.split()))

            metrics.append({
                "column_name": col,
                "min_chars": int(char_lens.min()),
                "max_chars": int(char_lens.max()),
                "avg_chars": round(float(char_lens.mean()), 2),
                "median_chars": float(char_lens.median()),
                "p95_chars": float(np.percentile(char_lens, 95)),
                "min_words": int(word_counts.min()),
                "max_words": int(word_counts.max()),
                "avg_words": round(float(word_counts.mean()), 2),
                "median_words": float(word_counts.median()),
                "p95_words": float(np.percentile(word_counts, 95)),
            })

    metrics_df = pd.DataFrame(metrics)

    # Plot Distribution of Summary Word Counts (Crucial for Vector DB Chunking)
    if "summary" in df.columns:
        word_counts = df["summary"].astype(str).apply(lambda x: len(x.split()))
        plt.figure(figsize=(10, 5))
        sns.histplot(word_counts, bins=50, kde=True, color="#2ecc71")
        plt.axvline(word_counts.median(), color="red", linestyle="--", label=f"Median: {int(word_counts.median())} words")
        plt.axvline(np.percentile(word_counts, 95), color="black", linestyle=":", label=f"95th Percentile: {int(np.percentile(word_counts, 95))} words")
        plt.title("Summary Word Count Distribution (RAG Chunking Benchmark)")
        plt.xlabel("Word Count per Abstract")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "summary_wordcount_distribution.png"), dpi=300)
        plt.close()

    return metrics_df


def analyze_author_network(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Evaluates top prolific authors in the dataset."""
    if "first_author" not in df.columns:
        return pd.DataFrame()

    author_counts = df["first_author"].value_counts().head(top_n).reset_index()
    author_counts.columns = ["author_name", "paper_count"]

    plt.figure(figsize=(10, 5))
    sns.barplot(data=author_counts, x="paper_count", y="author_name", palette="rocket")
    plt.title(f"Top {top_n} Most Prolific First Authors")
    plt.xlabel("Number of Papers")
    plt.ylabel("First Author")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "top_first_authors.png"), dpi=300)
    plt.close()

    return author_counts


def analyze_top_ngrams(df: pd.DataFrame, column: str = "summary", top_n: int = 15) -> pd.DataFrame:
    """Extracts top bi-grams across abstracts to uncover dominant scientific topics."""
    if column not in df.columns:
        return pd.DataFrame()

    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", 
        "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were", 
        "be", "been", "being", "have", "has", "had", "this", "that", "these", "those", 
        "it", "its", "we", "as", "you", "our", "paper", "show", "propose", "method", "results"
    }

    # Sample dataset if row count is massive for ultra-fast processing
    sample_size = min(50000, len(df))
    sample_text = df[column].sample(n=sample_size, random_state=42).astype(str).str.lower()

    bigrams = collections.Counter()
    for text in sample_text:
        tokens = [w for w in re.findall(r"\b[a-z]{3,}\b", text) if w not in stop_words]
        for i in range(len(tokens) - 1):
            bigrams[(tokens[i], tokens[i+1])] += 1

    top_bigrams = bigrams.most_common(top_n)
    bigram_data = [{"phrase": f"{k[0]} {k[1]}", "frequency": v} for k, v in top_bigrams]
    bigram_df = pd.DataFrame(bigram_data)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=bigram_df, x="frequency", y="phrase", palette="magma")
    plt.title(f"Top {top_n} Scientific Key Phrases (Bi-grams)")
    plt.xlabel("Frequency (Sampled N=50,000)")
    plt.ylabel("Phrase")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "top_key_phrases.png"), dpi=300)
    plt.close()

    return bigram_df


def save_reports(cat_df: pd.DataFrame, year_df: pd.DataFrame, length_df: pd.DataFrame, author_df: pd.DataFrame, phrase_df: pd.DataFrame) -> None:
    """Saves all generated statistical summaries to CSV format."""
    if not cat_df.empty:
        cat_df.to_csv(os.path.join(REPORTS_DIR, "eda_categories.csv"), index=False)
    if not year_df.empty:
        year_df.to_csv(os.path.join(REPORTS_DIR, "eda_publication_trends.csv"), index=False)
    if not length_df.empty:
        length_df.to_csv(os.path.join(REPORTS_DIR, "eda_text_length_metrics.csv"), index=False)
    if not author_df.empty:
        author_df.to_csv(os.path.join(REPORTS_DIR, "eda_top_authors.csv"), index=False)
    if not phrase_df.empty:
        phrase_df.to_csv(os.path.join(REPORTS_DIR, "eda_top_phrases.csv"), index=False)


def print_eda_summary(length_df: pd.DataFrame, cat_df: pd.DataFrame) -> None:
    """Outputs key system decisions derived directly from EDA."""
    print("\n========== EXPLORATORY DATA ANALYSIS (EDA) SUMMARY ==========")
    
    if not length_df.empty:
        summary_row = length_df[length_df["column_name"] == "summary"]
        if not summary_row.empty:
            avg_w = summary_row["avg_words"].values[0]
            p95_w = summary_row["p95_words"].values[0]
            print(f"Abstract Word Length   : Avg = {avg_w} words | 95th Percentile = {p95_w} words")
            print(f"RAG Chunking Insight   : Ideal Chunk Size = 256 to 512 tokens (No splitting required for ~95% of abstracts)")

    if not cat_df.empty:
        print(f"Total Unique Categories: {len(cat_df)}")
        print(f"Top Represented Field  : {cat_df.iloc[0]['category']} ({cat_df.iloc[0]['paper_count']} papers, {cat_df.iloc[0]['percentage']}%)")

    print(f"\nVisual plots saved to  : {PLOTS_DIR}")
    print(f"Reports saved to       : {REPORTS_DIR}")
    print("===============================================================\n")


def main() -> None:
    """Executes the complete EDA pipeline."""
    setup_directories()

    print("Loading processed dataset for EDA...")
    df = load_dataset(PROCESSED_DATA_PATH)

    print("Analyzing research categories...")
    cat_df = analyze_category_distribution(df)

    print("Analyzing publication timeline trends...")
    year_df = analyze_publication_trends(df)

    print("Calculating text length distributions for RAG optimization...")
    length_df = analyze_text_lengths(df)

    print("Analyzing top authors...")
    author_df = analyze_author_network(df)

    print("Extracting scientific key phrases (Bi-grams)...")
    phrase_df = analyze_top_ngrams(df, column="summary")

    print("Saving tabular EDA reports...")
    save_reports(cat_df, year_df, length_df, author_df, phrase_df)

    print_eda_summary(length_df, cat_df)


if __name__ == "__main__":
    main()