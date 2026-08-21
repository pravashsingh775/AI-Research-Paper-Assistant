import collections
import os
import pathlib
import re
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress visual deprecation warnings for cleaner log output
warnings.filterwarnings("ignore")

# Force matplotlib to use 'Agg' backend so plots save without rendering a UI window
plt.switch_backend("Agg")

# Dataset File Path Configuration
DATASET_PATH = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "dataset/raw/arXiv_scientific_dataset.csv")
OUTPUT_DIR = "outputs/reports"
PLOTS_DIR = "outputs/reports/plots"


def setup_directories(output_dir: str = OUTPUT_DIR, plots_dir: str = PLOTS_DIR) -> None:
    """Creates output directories for CSV reports and visual plots if they do not exist."""
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(plots_dir).mkdir(parents=True, exist_ok=True)


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads CSV dataset with fallback encodings and robust error handling."""
    target_path = pathlib.Path(file_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {file_path}")

    encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(target_path, encoding=encoding, low_memory=False)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    if df is None:
        raise ValueError("Failed to parse the CSV file with standard text encodings.")

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    return df


def get_basic_info(df: pd.DataFrame, file_path: str) -> Dict[str, object]:
    """Extracts high-level dataset metrics."""
    target_path = pathlib.Path(file_path)
    file_size_mb = target_path.stat().st_size / (1024 * 1024) if target_path.exists() else 0.0
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["category", "bool"]).columns.tolist()
    text_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in cat_cols]

    return {
        "dataset_name": target_path.name,
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "shape": f"{df.shape[0]}x{df.shape[1]}",
        "file_size_mb": round(file_size_mb, 2),
        "memory_usage_mb": round(memory_usage_mb, 2),
        "column_names": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "num_numerical_cols": len(num_cols),
        "num_categorical_cols": len(cat_cols),
        "num_text_cols": len(text_cols),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "text_columns": text_cols,
    }


def analyze_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Performs individual column evaluations, measuring values, missing counts, and length metrics."""
    records = []
    total_rows = len(df)

    for col in df.columns:
        series = df[col]
        str_series = series.astype(str)

        missing_cnt = series.isnull().sum()
        missing_pct = (missing_cnt / total_rows) * 100
        unique_cnt = series.nunique(dropna=False)
        duplicate_cnt = total_rows - unique_cnt

        sample_vals = series.dropna().unique()[:3].tolist()
        sample_str = ", ".join(map(str, sample_vals)) if len(sample_vals) > 0 else "None"

        lengths = str_series[series.notnull()].str.len()
        max_len = int(lengths.max()) if not lengths.empty else 0
        min_len = int(lengths.min()) if not lengths.empty else 0
        avg_len = float(lengths.mean()) if not lengths.empty else 0.0

        records.append({
            "column_name": col,
            "data_type": str(series.dtype),
            "missing_values": int(missing_cnt),
            "missing_percentage": round(missing_pct, 2),
            "unique_values": int(unique_cnt),
            "duplicate_values": int(duplicate_cnt),
            "sample_values": sample_str,
            "max_length": max_len,
            "min_length": min_len,
            "avg_length": round(avg_len, 2)
        })

    return pd.DataFrame(records)


def analyze_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes missing value metrics across all columns."""
    missing_cnt = df.isnull().sum()
    missing_pct = (missing_cnt / len(df)) * 100

    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_cnt.values,
        "missing_percentage": missing_pct.values.round(2)
    })
    return report.sort_values(by="missing_count", ascending=False).reset_index(drop=True)


def analyze_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Calculates row-level duplicate statistics and identifies duplicate record IDs if present."""
    total_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0.0

    id_col = None
    for col in df.columns:
        if re.search(r"\b(id|paper_id|arxiv_id)\b", col, re.I):
            id_col = col
            break

    duplicate_ids = []
    if id_col:
        dup_series = df[df.duplicated(subset=[id_col], keep=False)][id_col]
        duplicate_ids = dup_series.dropna().unique().tolist()[:50]

    summary = {
        "total_duplicates": duplicate_rows,
        "duplicate_percentage": round(duplicate_pct, 2),
        "id_column_found": id_col if id_col else "None",
        "sample_duplicate_ids": duplicate_ids
    }

    report = pd.DataFrame([summary])
    return report, summary


def analyze_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts numerical descriptive statistics."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()
    return num_df.describe().T.reset_index().rename(columns={"index": "column_name"})


def analyze_text_columns(df: pd.DataFrame, text_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates length distributions, whitespace patterns, vocabulary, and token frequencies for text fields."""
    text_summary = []
    special_char_summary = []

    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "over", "after",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "this", "that", "these", "those", "it", "its", "we", "as", "you", "our"
    }

    for col in text_cols:
        series = df[col].astype(str).fillna("")

        lengths = series.apply(len)
        min_len = int(lengths.min()) if not lengths.empty else 0
        max_len = int(lengths.max()) if not lengths.empty else 0
        avg_len = float(lengths.mean()) if not lengths.empty else 0.0
        median_len = float(lengths.median()) if not lengths.empty else 0.0

        empty_strings = int((series == "").sum())
        whitespace_only = int(series.apply(lambda s: len(s) > 0 and s.isspace()).sum())

        all_text = " ".join(series.tolist()).lower()
        tokens = re.findall(r"\b[a-z]{2,}\b", all_text)
        filtered_tokens = [t for t in tokens if t not in stop_words]

        vocab_size = len(set(tokens))
        word_counts = collections.Counter(filtered_tokens)
        most_common = word_counts.most_common(10)
        most_common_str = ", ".join([f"{k}:{v}" for k, v in most_common])

        text_summary.append({
            "column_name": col,
            "min_length": min_len,
            "max_length": max_len,
            "avg_length": round(avg_len, 2),
            "median_length": round(median_len, 2),
            "empty_strings": empty_strings,
            "whitespace_only_rows": whitespace_only,
            "vocabulary_size": vocab_size,
            "top_10_words": most_common_str
        })

        html_tags_cnt = int(series.apply(lambda s: bool(re.search(r"<[^>]+>", s))).sum())
        urls_cnt = int(series.apply(lambda s: bool(re.search(r"https?://\S+|www\.\S+", s))).sum())
        emails_cnt = int(series.apply(lambda s: bool(re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", s))).sum())
        unicode_cnt = int(series.apply(lambda s: bool(re.search(r"[^\x00-\x7F]", s))).sum())
        math_sym_cnt = int(series.apply(lambda s: bool(re.search(r"[\$\\=\+\-\*\/\^\_\{\}]", s))).sum())
        spec_sym_cnt = int(series.apply(lambda s: bool(re.search(r"[\[\]\(\)@#%&~`|]", s))).sum())

        special_char_summary.append({
            "column_name": col,
            "rows_with_html": html_tags_cnt,
            "rows_with_urls": urls_cnt,
            "rows_with_emails": emails_cnt,
            "rows_with_unicode": unicode_cnt,
            "rows_with_math_symbols": math_sym_cnt,
            "rows_with_special_symbols": spec_sym_cnt
        })

    return pd.DataFrame(text_summary), pd.DataFrame(special_char_summary)


def analyze_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluates individual columns for anomalies, mixed types, and corruption patterns."""
    quality_records = []

    for col in df.columns:
        series = df[col]
        null_cnt = series.isnull().sum()
        empty_str_cnt = (series.astype(str).str.strip() == "").sum() - null_cnt
        empty_str_cnt = max(0, int(empty_str_cnt))

        non_null_types = series.dropna().apply(type).unique()
        has_mixed_types = len(non_null_types) > 1

        unprintable_cnt = int(series.dropna().astype(str).apply(lambda x: bool(re.search(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", x))).sum())

        unexpected_cnt = 0
        if pd.api.types.is_numeric_dtype(series):
            unexpected_cnt = int((series < 0).sum())

        quality_records.append({
            "column_name": col,
            "null_values": int(null_cnt),
            "empty_values": empty_str_cnt,
            "corrupted_rows": unprintable_cnt,
            "has_mixed_datatypes": has_mixed_types,
            "detected_types": ", ".join([t.__name__ for t in non_null_types]),
            "unexpected_negative_values": unexpected_cnt
        })

    return pd.DataFrame(quality_records)


def generate_visualizations(df: pd.DataFrame, text_cols: List[str], plots_dir: str = PLOTS_DIR) -> None:
    """Generates and saves visual quality charts and distributions."""
    plt.style.use("ggplot")

    # 1. Missing Value Heatmap
    plt.figure(figsize=(10, 6))
    if df.isnull().sum().sum() > 0:
        plt.imshow(df.isnull(), cmap="viridis", aspect="auto", interpolation="none")
        plt.title("Missing Value Heatmap")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha="right")
    else:
        plt.text(0.5, 0.5, "No Missing Values Detected", horizontalalignment="center", verticalalignment="center", fontsize=14)
        plt.title("Missing Value Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "missing_value_heatmap.png"))
    plt.close()

    # 2. Missing Value Bar Chart
    plt.figure(figsize=(10, 5))
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        missing_counts.plot(kind="bar", color="#3498db")
        plt.title("Missing Values per Column")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
    else:
        plt.text(0.5, 0.5, "No Missing Values", horizontalalignment="center", verticalalignment="center", fontsize=14)
        plt.title("Missing Values per Column")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "missing_value_barchart.png"))
    plt.close()

    # 3. Data Type Distribution
    plt.figure(figsize=(6, 6))
    dtype_counts = df.dtypes.astype(str).value_counts()
    plt.pie(dtype_counts, labels=dtype_counts.index, autopct="%1.1f%%", startangle=140)
    plt.title("Data Type Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "datatype_distribution.png"))
    plt.close()

    # 4. Text Length Histogram
    if text_cols:
        primary_text_col = text_cols[0]
        lengths = df[primary_text_col].astype(str).fillna("").str.len()
        plt.figure(figsize=(10, 5))
        plt.hist(lengths, bins=30, color="#2ecc71", edgecolor="black")
        plt.title(f"Text Length Distribution ({primary_text_col})")
        plt.xlabel("Character Count")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "text_length_histogram.png"))
        plt.close()

    # 5. Top Frequent Words
    if text_cols:
        combined_text = " ".join(df[text_cols[0]].astype(str).fillna("")).lower()
        words = re.findall(r"\b[a-z]{3,}\b", combined_text)
        filtered = [w for w in words if w not in {"the", "and", "for", "with", "this", "that", "from"}]
        top_words = collections.Counter(filtered).most_common(15)

        if top_words:
            w_df = pd.DataFrame(top_words, columns=["word", "count"])
            plt.figure(figsize=(10, 5))
            plt.barh(w_df["word"], w_df["count"], color="#e74c3c")
            plt.gca().invert_yaxis()
            plt.title(f"Top 15 Words in {text_cols[0]}")
            plt.xlabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "top_frequent_words.png"))
            plt.close()

    # 6. Correlation Heatmap
    num_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(8, 6))
    if num_df.shape[1] > 1:
        corr = num_df.corr()
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Numerical Feature Correlation")
    else:
        plt.text(0.5, 0.5, "Insufficient Numerical Columns for Correlation", horizontalalignment="center", verticalalignment="center", fontsize=12)
        plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"))
    plt.close()


def generate_recommendations(df: pd.DataFrame, basic_info: Dict[str, object], column_analysis: pd.DataFrame) -> Dict[str, object]:
    """Generates automated downstream architectural and preprocessing recommendations based on analysis results."""
    cols_to_remove = []
    cols_to_clean = []
    nlp_cols = []
    rag_cols = []
    ignore_cols = []

    total_rows = len(df)

    for _, row in column_analysis.iterrows():
        col_name = row["column_name"]
        missing_pct = row["missing_percentage"]
        unique_cnt = row["unique_values"]
        avg_len = row["avg_length"]

        if missing_pct > 80.0:
            cols_to_remove.append(col_name)

        if missing_pct > 0.0 or row["duplicate_values"] > 0:
            cols_to_clean.append(col_name)

        if col_name in basic_info["text_columns"]:
            if avg_len >= 20:
                nlp_cols.append(col_name)
            if avg_len >= 100 or "abstract" in col_name.lower() or "text" in col_name.lower():
                rag_cols.append(col_name)

        if unique_cnt <= 1:
            ignore_cols.append(col_name)

    target_candidates = [c for c in df.columns if re.search(r"\b(category|label|target|subject|class)\b", c, re.I)]
    has_labels = len(target_candidates) > 0
    supervised_possible = has_labels and len(df) >= 50

    suitable_for_semantic_search = len(nlp_cols) > 0 or len(rag_cols) > 0
    suitable_for_rag = len(rag_cols) > 0
    suitable_for_recommender = total_rows >= 100 and (len(nlp_cols) > 0 or has_labels)
    suitable_for_kg = len(nlp_cols) > 0 and (has_labels or any("author" in c.lower() for c in df.columns))

    return {
        "columns_to_remove": list(set(cols_to_remove)),
        "columns_to_clean": list(set(cols_to_clean)),
        "nlp_columns": nlp_cols,
        "rag_columns": rag_cols,
        "ignored_columns": list(set(ignore_cols)),
        "has_labels": has_labels,
        "supervised_ml_possible": supervised_possible,
        "suitable_for_semantic_search": suitable_for_semantic_search,
        "suitable_for_rag": suitable_for_rag,
        "suitable_for_recommendation_systems": suitable_for_recommender,
        "suitable_for_knowledge_graphs": suitable_for_kg
    }


def save_reports(
    basic_info: Dict[str, object],
    col_analysis: pd.DataFrame,
    missing_df: pd.DataFrame,
    dup_df: pd.DataFrame,
    text_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR
) -> None:
    """Saves all tabular reports to CSV files inside the designated directory."""
    pd.DataFrame([basic_info]).to_csv(os.path.join(output_dir, "dataset_summary.csv"), index=False)
    col_analysis.to_csv(os.path.join(output_dir, "column_analysis.csv"), index=False)
    missing_df.to_csv(os.path.join(output_dir, "missing_report.csv"), index=False)
    dup_df.to_csv(os.path.join(output_dir, "duplicate_report.csv"), index=False)
    text_df.to_csv(os.path.join(output_dir, "text_analysis.csv"), index=False)
    quality_df.to_csv(os.path.join(output_dir, "quality_report.csv"), index=False)


def print_console_reports(
    basic_info: Dict[str, object],
    col_analysis: pd.DataFrame,
    quality_df: pd.DataFrame,
    text_df: pd.DataFrame,
    recommendations: Dict[str, object]
) -> None:
    """Displays structured summary reports to stdout."""
    print("\n========== DATASET SUMMARY ==========")
    print(f"Dataset Name        : {basic_info['dataset_name']}")
    print(f"Total Rows          : {basic_info['num_rows']}")
    print(f"Total Columns       : {basic_info['num_columns']}")
    print(f"File Size (MB)      : {basic_info['file_size_mb']}")
    print(f"Memory Usage (MB)   : {basic_info['memory_usage_mb']}")
    print(f"Numerical Columns   : {basic_info['num_numerical_cols']}")
    print(f"Categorical Columns : {basic_info['num_categorical_cols']}")
    print(f"Text Columns        : {basic_info['num_text_cols']}")

    print("\n========== COLUMN ANALYSIS ==========")
    print(col_analysis[["column_name", "data_type", "missing_percentage", "unique_values", "avg_length"]].to_string(index=False))

    print("\n========== DATA QUALITY REPORT ==========")
    print(quality_df.to_string(index=False))

    print("\n========== TEXT ANALYSIS ==========")
    if not text_df.empty:
        print(text_df[["column_name", "min_length", "max_length", "avg_length", "vocabulary_size"]].to_string(index=False))
    else:
        print("No text columns detected for analysis.")

    print("\n========== FINAL CONCLUSION ==========")
    print(f"Columns Recommended for Removal : {recommendations['columns_to_remove']}")
    print(f"Columns Requiring Cleaning       : {recommendations['columns_to_clean']}")
    print(f"Primary NLP Feature Columns      : {recommendations['nlp_columns']}")
    print(f"RAG Chunking Candidate Columns   : {recommendations['rag_columns']}")
    print(f"Supervised ML Training Possible  : {recommendations['supervised_ml_possible']}")
    print(f"Suitable for Semantic Search     : {recommendations['suitable_for_semantic_search']}")
    print(f"Suitable for Recommendation Systems: {recommendations['suitable_for_recommendation_systems']}")
    print(f"Suitable for Knowledge Graphs    : {recommendations['suitable_for_knowledge_graphs']}\n")


def main() -> None:
    """Main execution pipeline."""
    setup_directories(output_dir=OUTPUT_DIR, plots_dir=PLOTS_DIR)

    try:
        df = load_dataset(DATASET_PATH)
    except Exception as e:
        print(f"Critical Error during file loading: {e}")
        return

    basic_info = get_basic_info(df, DATASET_PATH)
    col_analysis = analyze_columns(df)
    missing_df = analyze_missing_data(df)
    dup_df, _ = analyze_duplicates(df)
    _ = analyze_statistics(df)
    text_df, _ = analyze_text_columns(df, basic_info["text_columns"])
    quality_df = analyze_data_quality(df)

    generate_visualizations(df, basic_info["text_columns"], plots_dir=PLOTS_DIR)
    recommendations = generate_recommendations(df, basic_info, col_analysis)

    save_reports(
        basic_info=basic_info,
        col_analysis=col_analysis,
        missing_df=missing_df,
        dup_df=dup_df,
        text_df=text_df,
        quality_df=quality_df,
        output_dir=OUTPUT_DIR
    )

    print_console_reports(basic_info, col_analysis, quality_df, text_df, recommendations)


if __name__ == "__main__":
    main()