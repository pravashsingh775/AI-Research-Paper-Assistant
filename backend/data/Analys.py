from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

# Matplotlib is intentionally imported with a non-interactive backend so the
# script works reliably from a Windows terminal / VS Code.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None


# ---------------------------------------------------------------------------
# Configuration & Aesthetics
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "Dataset_Report"

REPORT_DIR = OUTPUT_DIR / "reports"
TABLE_DIR = OUTPUT_DIR / "tables"
VIZ_DIR = OUTPUT_DIR / "visualizations"
LOG_DIR = OUTPUT_DIR / "logs"

for directory in (REPORT_DIR, TABLE_DIR, VIZ_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SUPPORTED_DATASET_TYPES = {".csv", ".parquet", ".json", ".jsonl", ".ndjson"}

LOGGER = logging.getLogger("dataset_analysis")

# Apply professional styling configuration
if sns is not None:
    sns.set_theme(style="whitegrid", palette="deep")
else:
    plt.style.use("ggplot")

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["figure.dpi"] = 300

PRIMARY_COLOR = "#2b5c8f"
SECONDARY_COLOR = "#4682b4"
ACCENT_COLOR = "#d95f02"

NULL_LIKE_VALUES = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "nil",
    "-",
    "--",
    "unknown",
    "not available",
    "not_applicable",
    "n.a.",
}

TEXT_NAME_HINTS = {
    "title": ["title", "paper_title", "name"],
    "abstract": ["abstract", "summary", "description", "paper_abstract"],
    "authors": ["author", "authors", "creator", "contributors"],
    "categories": ["category", "categories", "subject", "subjects", "topic", "topics"],
    "keywords": ["keyword", "keywords"],
    "journal": ["journal", "venue", "publication"],
    "comments": ["comment", "comments", "note", "notes"],
}

ID_HINTS = [
    "id",
    "paper_id",
    "arxiv_id",
    "doi",
    "identifier",
    "uuid",
    "url",
    "link",
]

DATE_HINTS = [
    "date",
    "published",
    "publication",
    "year",
    "created",
    "updated",
    "submitted",
    "timestamp",
]

CATEGORY_HINTS = [
    "category",
    "categories",
    "subject",
    "subjects",
    "topic",
    "topics",
    "field",
    "domain",
    "area",
]

HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
LATEX_RE = re.compile(r"(\\[a-zA-Z]+|\\\(\vert{}\\\)|\$\$|\\begin\{|\\end\{)")
MULTISPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
BAD_REPLACEMENT_RE = re.compile(r"\ufffd")
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


@dataclass
class DatasetInfo:
    path: str
    file_type: str
    file_size_bytes: int
    file_size_mb: float
    encoding: Optional[str]
    delimiter: Optional[str]
    row_count: int
    column_count: int
    memory_mb: float
    columns: list[str]


@dataclass
class ColumnProfile:
    column: str
    dtype: str
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    empty_string_count: int
    whitespace_only_count: int
    null_like_count: int
    example_value: str
    inferred_role: str
    notes: str


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        LOG_DIR / "analysis.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: Iterable[dict[str, Any]], columns: Optional[list[str]] = None) -> None:
    rows = list(rows)
    if not rows:
        pd.DataFrame(columns=columns or []).to_csv(path, index=False, encoding="utf-8-sig")
        return
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def normalized_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.lower().strip()
    text = HTML_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    return MULTISPACE_RE.sub(" ", text).strip()


def word_count(text: Any) -> int:
    if pd.isna(text):
        return 0
    return len(WORD_RE.findall(str(text)))


def text_stats(series: pd.Series) -> dict[str, Any]:
    lengths = series.fillna("").astype(str).map(len)
    words = series.fillna("").astype(str).map(word_count)

    def q(values: pd.Series, p: float) -> float:
        return float(values.quantile(p)) if len(values) else 0.0

    return {
        "count": int(len(series)),
        "empty_count": int((series.fillna("").astype(str).str.strip() == "").sum()),
        "min_chars": int(lengths.min()) if len(lengths) else 0,
        "max_chars": int(lengths.max()) if len(lengths) else 0,
        "mean_chars": float(lengths.mean()) if len(lengths) else 0.0,
        "median_chars": float(lengths.median()) if len(lengths) else 0.0,
        "std_chars": float(lengths.std(ddof=1)) if len(lengths) > 1 else 0.0,
        "q1_chars": q(lengths, 0.25),
        "q3_chars": q(lengths, 0.75),
        "p95_chars": q(lengths, 0.95),
        "p99_chars": q(lengths, 0.99),
        "min_words": int(words.min()) if len(words) else 0,
        "max_words": int(words.max()) if len(words) else 0,
        "mean_words": float(words.mean()) if len(words) else 0.0,
        "median_words": float(words.median()) if len(words) else 0.0,
        "p95_words": q(words, 0.95),
        "p99_words": q(words, 0.99),
    }


def save_current_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_").lower()


# ---------------------------------------------------------------------------
# Dataset discovery / loading
# ---------------------------------------------------------------------------

def discover_datasets(explicit: Optional[Path] = None) -> list[Path]:
    supported = {".csv", ".parquet", ".json", ".jsonl", ".ndjson"}

    if explicit is not None:
        path = explicit if explicit.is_absolute() else PROJECT_ROOT / explicit
        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        if not path.is_file():
            raise ValueError(f"Dataset path is not a file: {path}")

        if path.suffix.lower() not in supported:
            raise ValueError(
                f"Unsupported dataset type '{path.suffix}'. "
                f"Supported types: {sorted(supported)}"
            )

        return [path]

    raw_dir = RAW_DIR.resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw data path is not a directory: {raw_dir}")

    candidates = [
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in supported
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No supported dataset found under {raw_dir}. "
            f"Supported types: {sorted(supported)}"
        )

    return sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)


def detect_csv_format(path: Path) -> tuple[Optional[str], Optional[str]]:
    raw = path.read_bytes()[:2_000_000]

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    encoding = None
    sample = None

    for enc in encodings:
        try:
            sample = raw.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue

    if sample is None:
        raise UnicodeDecodeError("unknown", raw, 0, min(10, len(raw)), "Unable to decode sample")

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    return encoding, delimiter


def inspect_dataset(path: Path) -> DatasetInfo:
    suffix = path.suffix.lower()
    size = path.stat().st_size
    encoding = None
    delimiter = None

    if suffix == ".csv":
        encoding, delimiter = detect_csv_format(path)

    return DatasetInfo(
        path=str(path),
        file_type=suffix.lstrip("."),
        file_size_bytes=size,
        file_size_mb=round(size / (1024**2), 3),
        encoding=encoding,
        delimiter=delimiter,
        row_count=0,
        column_count=0,
        memory_mb=0.0,
        columns=[],
    )


def read_dataset(path: Path, sample_rows: int = 5000) -> tuple[pd.DataFrame, DatasetInfo]:
    info = inspect_dataset(path)
    suffix = path.suffix.lower()

    LOGGER.info("Reading dataset: %s", path)

    if suffix == ".csv":
        df = pd.read_csv(
            path,
            encoding=info.encoding or "utf-8",
            sep=info.delimiter or ",",
            low_memory=False,
            on_bad_lines="warn",
        )
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    info.row_count = int(len(df))
    info.column_count = int(len(df.columns))
    info.columns = [str(c) for c in df.columns]
    info.memory_mb = round(float(df.memory_usage(deep=True).sum()) / (1024**2), 3)

    LOGGER.info(
        "Loaded %s rows x %s columns | memory %.2f MB",
        info.row_count,
        info.column_count,
        info.memory_mb,
    )

    return df, info


# ---------------------------------------------------------------------------
# Column role inference
# ---------------------------------------------------------------------------

def find_column(df: pd.DataFrame, hints: list[str]) -> Optional[str]:
    normalized = {c: normalize_column_name(c) for c in df.columns}

    for col, norm in normalized.items():
        if norm in hints:
            return col

    for col, norm in normalized.items():
        if any(h in norm for h in hints):
            return col

    return None


def identify_special_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    return {
        "title": find_column(df, TEXT_NAME_HINTS["title"]),
        "abstract": find_column(df, TEXT_NAME_HINTS["abstract"]),
        "authors": find_column(df, TEXT_NAME_HINTS["authors"]),
        "categories": find_column(df, CATEGORY_HINTS),
        "keywords": find_column(df, TEXT_NAME_HINTS["keywords"]),
        "journal": find_column(df, TEXT_NAME_HINTS["journal"]),
        "comments": find_column(df, TEXT_NAME_HINTS["comments"]),
        "id": find_column(df, ID_HINTS),
        "date": find_column(df, DATE_HINTS),
    }


def infer_role(column: str, series: pd.Series, special: dict[str, Optional[str]]) -> tuple[str, str]:
    if column == special.get("title"):
        return "title", "Likely paper title based on column name and text profile."
    if column == special.get("abstract"):
        return "abstract", "Likely paper abstract based on column name and text profile."
    if column == special.get("authors"):
        return "authors", "Likely author information."
    if column == special.get("categories"):
        return "category", "Likely research category/topic field."
    if column == special.get("keywords"):
        return "keywords", "Likely keyword field."
    if column == special.get("journal"):
        return "venue", "Likely journal/venue field."
    if column == special.get("comments"):
        return "comments", "Likely comments/notes metadata."
    if column == special.get("id"):
        return "identifier", "Likely identifier/URL field."
    if column == special.get("date"):
        return "date", "Likely temporal metadata."

    norm = normalize_column_name(column)
    nunique = int(series.nunique(dropna=True))
    n = max(len(series), 1)
    ratio = nunique / n

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime", "Datetime-like column."

    if pd.api.types.is_numeric_dtype(series):
        if ratio > 0.98:
            return "possible_identifier", "Numeric field with near-unique values."
        return "numeric", "Numeric metadata."

    if pd.api.types.is_string_dtype(series):
        avg_words = float(series.fillna("").astype(str).map(word_count).mean())
        if ratio > 0.98 and any(h in norm for h in ["id", "doi", "url", "link"]):
            return "identifier", "High-cardinality identifier-like text field."
        if avg_words >= 8:
            return "text", "High-information free-text field."
        if ratio < 0.05:
            return "categorical", "Low-cardinality categorical field."
        return "text_or_metadata", "String field requiring contextual interpretation."

    return "other", "No strong semantic role inferred."


# ---------------------------------------------------------------------------
# Schema / missingness / uniqueness
# ---------------------------------------------------------------------------

def analyze_schema(df: pd.DataFrame, special: dict[str, Optional[str]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    schema_json: dict[str, Any] = {}

    for col in df.columns:
        s = df[col]
        null_mask = s.isna()

        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            as_text = s.fillna("").astype(str)
            empty = int((as_text.str.strip() == "").sum())
            null_like = int(
                as_text.str.strip().str.lower().isin(NULL_LIKE_VALUES).sum()
            )
            whitespace_only = int(
                ((as_text != "") & (as_text.str.strip() == "")).sum()
            )
        else:
            empty = 0
            whitespace_only = 0
            null_like = 0

        non_null = int(s.notna().sum())
        unique = int(s.nunique(dropna=True))
        role, notes = infer_role(str(col), s, special)

        examples = s.dropna().head(3).tolist()
        example = json.dumps(json_safe(examples[0]), ensure_ascii=False) if examples else ""

        profile = ColumnProfile(
            column=str(col),
            dtype=str(s.dtype),
            non_null_count=non_null,
            null_count=int(null_mask.sum()),
            null_percentage=round(float(null_mask.mean() * 100), 4),
            unique_count=unique,
            unique_percentage=round(float(unique / max(len(s), 1) * 100), 4),
            empty_string_count=empty,
            whitespace_only_count=whitespace_only,
            null_like_count=null_like,
            example_value=example,
            inferred_role=role,
            notes=notes,
        )
        profiles.append(asdict(profile))

        schema_json[str(col)] = {
            "name": str(col),
            "dtype": str(s.dtype),
            "nullable": bool(null_mask.any()),
            "unique_count": unique,
            "missing_count": int(null_mask.sum()),
            "missing_percentage": round(float(null_mask.mean() * 100), 4),
            "sample_values": [json_safe(x) for x in examples],
            "inferred_role": role,
            "notes": notes,
        }

    result = pd.DataFrame(profiles)
    result.to_csv(TABLE_DIR / "column_summary.csv", index=False, encoding="utf-8-sig")

    missing = result[
        [
            "column",
            "null_count",
            "null_percentage",
            "empty_string_count",
            "whitespace_only_count",
            "null_like_count",
        ]
    ].sort_values("null_percentage", ascending=False)
    missing.to_csv(TABLE_DIR / "missing_values.csv", index=False, encoding="utf-8-sig")

    unique_df = result[
        ["column", "unique_count", "unique_percentage", "dtype", "inferred_role"]
    ].sort_values("unique_count", ascending=False)
    unique_df.to_csv(TABLE_DIR / "unique_values.csv", index=False, encoding="utf-8-sig")

    return result, schema_json


def duplicate_analysis(df: pd.DataFrame, special: dict[str, Optional[str]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "row_count": int(len(df)),
        "exact_duplicate_rows": int(df.duplicated(keep=False).sum()),
        "exact_duplicate_row_groups": int(df.duplicated(keep="first").sum()),
    }

    checks: list[dict[str, Any]] = []

    def add_duplicate_check(name: str, column_names: list[str]) -> None:
        if not all(c in df.columns for c in column_names):
            checks.append({
                "check": name,
                "status": "not_applicable",
                "columns": column_names,
                "duplicate_rows": None,
                "duplicate_percentage": None,
                "reason": "Required column(s) not available.",
            })
            return

        temp = df[column_names].copy()
        for c in column_names:
            temp[c] = temp[c].map(normalized_text)

        combined = temp.astype(str).agg(" || ".join, axis=1)
        duplicated = combined.ne("") & combined.duplicated(keep=False)
        checks.append({
            "check": name,
            "status": "analyzed",
            "columns": column_names,
            "duplicate_rows": int(duplicated.sum()),
            "duplicate_percentage": round(float(duplicated.mean() * 100), 4),
            "reason": "",
        })

    title = special.get("title")
    abstract = special.get("abstract")
    identifier = special.get("id")

    if identifier:
        add_duplicate_check("duplicate_identifier", [identifier])
    if title:
        add_duplicate_check("duplicate_title", [title])
    if abstract:
        add_duplicate_check("duplicate_abstract", [abstract])
    if title and abstract:
        add_duplicate_check("duplicate_title_abstract", [title, abstract])

    result["checks"] = checks

    if title:
        normalized_titles = df[title].map(normalized_text)
        normalized_titles = normalized_titles[normalized_titles != ""]
        result["normalized_nonempty_title_count"] = int(len(normalized_titles))
        result["normalized_unique_title_count"] = int(normalized_titles.nunique())
    else:
        result["normalized_nonempty_title_count"] = None
        result["normalized_unique_title_count"] = None

    pd.DataFrame(checks).to_csv(
        TABLE_DIR / "duplicate_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return result


# ---------------------------------------------------------------------------
# Text analysis & Enhanced Visualizations
# ---------------------------------------------------------------------------

def detect_text_quality(series: pd.Series) -> dict[str, Any]:
    s = series.fillna("").astype(str)
    nonempty = s.str.strip().ne("")

    html_count = int(s.str.contains(HTML_RE, regex=True, na=False).sum())
    url_count = int(s.str.contains(URL_RE, regex=True, na=False).sum())
    latex_count = int(s.str.contains(LATEX_RE, regex=True, na=False).sum())
    replacement_count = int(s.str.contains(BAD_REPLACEMENT_RE, regex=True, na=False).sum())
    excessive_punct = int(
        s.map(
            lambda x: (
                len(PUNCT_RE.findall(x)) / max(len(x), 1)
            ) > 0.35
        ).sum()
    )

    words = s.map(word_count)

    return {
        "rows": int(len(s)),
        "empty_count": int((~nonempty).sum()),
        "very_short_lt_5_words": int((words < 5).sum()),
        "short_lt_20_words": int((words < 20).sum()),
        "long_gt_2000_words": int((words > 2000).sum()),
        "html_xml_count": html_count,
        "url_count": url_count,
        "latex_count": latex_count,
        "replacement_character_count": replacement_count,
        "excessive_punctuation_count": excessive_punct,
        "normalized_duplicate_count": int(
            s[s.str.strip().ne("")].map(normalized_text).duplicated(keep=False).sum()
        ),
    }


def analyze_text_columns(df: pd.DataFrame, special: dict[str, Optional[str]]) -> dict[str, Any]:
    candidates: list[str] = []

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            candidates.append(str(col))

    stats_rows = []
    details: dict[str, Any] = {}

    for col in candidates:
        stats = text_stats(df[col])
        quality = detect_text_quality(df[col])
        row = {
            "column": col,
            **stats,
            **quality,
        }
        stats_rows.append(row)
        details[col] = row

    pd.DataFrame(stats_rows).to_csv(
        TABLE_DIR / "text_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    title = special.get("title")
    abstract = special.get("abstract")

    if title:
        _plot_text_distribution(df[title], "Paper Title Character Length", VIZ_DIR / "title_length_distribution.png")
    if abstract:
        _plot_text_distribution(df[abstract], "Paper Abstract Character Length", VIZ_DIR / "abstract_length_distribution.png")

    if title or abstract:
        combined = pd.Series([""] * len(df), index=df.index, dtype="string")
        if title:
            combined = combined + df[title].fillna("").astype(str)
        if title and abstract:
            combined = combined + " "
        if abstract:
            combined = combined + df[abstract].fillna("").astype(str)

        _plot_text_distribution(
            combined,
            "Combined Title and Abstract Character Length",
            VIZ_DIR / "text_length_distribution.png",
        )

        details["_combined_title_abstract"] = text_stats(combined)

    return details


def _plot_text_distribution(series: pd.Series, title: str, path: Path) -> None:
    lengths = series.fillna("").astype(str).map(len)
    if lengths.empty:
        return

    upper = float(lengths.quantile(0.99))
    plot_values = lengths.clip(upper=upper)

    plt.figure(figsize=(11, 6))
    if sns is not None:
        sns.histplot(plot_values, kde=True, color=PRIMARY_COLOR, bins=50)
    else:
        plt.hist(plot_values, bins=50, color=PRIMARY_COLOR, edgecolor="white")

    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Characters", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Papers", fontsize=11, fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    save_current_plot(path)


def split_multilabel_values(value: Any) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if re.search(r"[\s,;|]+", text):
        parts = re.split(r"[\s,;|]+", text)
    else:
        parts = [text]

    return [p.strip() for p in parts if p.strip()]


def analyze_categories(df: pd.DataFrame, special: dict[str, Optional[str]], top_n: int) -> dict[str, Any]:
    category_columns = [
        c for c in df.columns
        if any(h in normalize_column_name(c) for h in CATEGORY_HINTS)
    ]

    if not category_columns:
        return {"status": "not_available", "columns": []}

    all_results = []
    selected = special.get("categories")

    for col in category_columns:
        counter = Counter()
        delimiter_signal = 0

        for value in df[col].dropna():
            parts = split_multilabel_values(value)
            if len(parts) > 1:
                delimiter_signal += 1
            for part in parts:
                counter[part] += 1

        total_assignments = sum(counter.values()) or 1
        rows = [
            {
                "column": col,
                "category": cat,
                "count": count,
                "percentage_of_assignments": round(count / total_assignments * 100, 4),
            }
            for cat, count in counter.most_common()
        ]

        all_results.extend(rows)

        if col == selected or (selected is None and rows):
            top = rows[:top_n]
            if top:
                plt.figure(figsize=(11, max(6, min(14, len(top) * 0.35))))
                plot_df = pd.DataFrame(top).sort_values("count")
                
                if sns is not None:
                    sns.barplot(x=plot_df["count"], y=plot_df["category"], color=PRIMARY_COLOR)
                else:
                    plt.barh(plot_df["category"], plot_df["count"], color=PRIMARY_COLOR)

                plt.title(f"Top {len(top)} Research Categories", fontsize=14, fontweight="bold", pad=15)
                plt.xlabel("Count", fontsize=11, fontweight="bold")
                plt.ylabel("Category", fontsize=11, fontweight="bold")
                plt.grid(axis="x", alpha=0.3, linestyle="--")
                save_current_plot(VIZ_DIR / "top_categories.png")

    category_df = pd.DataFrame(all_results)
    category_df.to_csv(
        TABLE_DIR / "class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "status": "analyzed",
        "columns": category_columns,
        "selected_column": selected,
        "total_rows": int(len(df)),
        "top_categories": category_df.head(top_n).to_dict(orient="records"),
        "unique_labels_by_column": {
            col: int(category_df.loc[category_df["column"] == col, "category"].nunique())
            for col in category_columns
        },
    }


def detect_date_column(df: pd.DataFrame, special: dict[str, Optional[str]]) -> Optional[str]:
    candidate = special.get("date")
    if candidate:
        return candidate

    for col in df.columns:
        norm = normalize_column_name(col)
        if any(h in norm for h in DATE_HINTS):
            return str(col)

    return None


def analyze_temporal(df: pd.DataFrame, special: dict[str, Optional[str]]) -> dict[str, Any]:
    col = detect_date_column(df, special)
    if not col:
        return {"status": "not_available", "column": None}

    raw = df[col]

    if pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce")
        valid_year = numeric.between(1800, datetime.now().year + 1)
        dates = pd.to_datetime(
            numeric.where(valid_year).astype("Int64").astype(str),
            format="%Y",
            errors="coerce",
        )
    else:
        dates = pd.to_datetime(raw, errors="coerce", utc=True)

    valid = dates.notna()
    if not valid.any():
        return {
            "status": "unparseable",
            "column": col,
            "valid_count": 0,
            "invalid_count": int((~valid).sum()),
        }

    year = dates[valid].dt.year
    current_year = datetime.now(timezone.utc).year
    suspicious_future = int((year > current_year + 1).sum())
    suspicious_old = int((year < 1800).sum())

    yearly = year.value_counts().sort_index()
    monthly = dates[valid].dt.tz_localize(None).dt.to_period("M").astype(str).value_counts().sort_index()

    pd.DataFrame({
        "year": yearly.index.astype(int),
        "paper_count": yearly.values.astype(int),
    }).to_csv(TABLE_DIR / "temporal_statistics.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(12, 6))
    plt.plot(yearly.index, yearly.values, marker="o", linewidth=2, color=PRIMARY_COLOR)
    plt.title(f"Papers Published by Year — {col}", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Year", fontsize=11, fontweight="bold")
    plt.ylabel("Paper Count", fontsize=11, fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--")
    save_current_plot(VIZ_DIR / "year_distribution.png")

    if len(monthly) >= 6:
        plt.figure(figsize=(13, 6))
        plt.plot(range(len(monthly)), monthly.values, linewidth=1.5, color=SECONDARY_COLOR)
        tick_positions = np.linspace(0, len(monthly) - 1, min(12, len(monthly)), dtype=int)
        plt.xticks(tick_positions, [monthly.index[i] for i in tick_positions], rotation=45, ha="right")
        plt.title(f"Papers Published by Month — {col}", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Month", fontsize=11, fontweight="bold")
        plt.ylabel("Paper Count", fontsize=11, fontweight="bold")
        plt.grid(alpha=0.3, linestyle="--")
        save_current_plot(VIZ_DIR / "month_distribution.png")

    growth = {}
    if len(yearly) >= 2:
        prev = None
        for y, count in yearly.items():
            if prev and prev > 0:
                growth[str(int(y))] = round((count - prev) / prev * 100, 2)
            prev = count

    return {
        "status": "analyzed",
        "column": col,
        "valid_count": int(valid.sum()),
        "invalid_count": int((~valid).sum()),
        "missing_count": int(raw.isna().sum()),
        "earliest_year": int(year.min()),
        "latest_year": int(year.max()),
        "suspicious_future_dates": suspicious_future,
        "suspicious_old_dates": suspicious_old,
        "papers_per_year": {str(int(k)): int(v) for k, v in yearly.items()},
        "annual_growth_percentage": growth,
    }


def analyze_authors(df: pd.DataFrame, special: dict[str, Optional[str]], top_n: int) -> dict[str, Any]:
    col = special.get("authors")
    if not col:
        return {"status": "not_available", "column": None}

    s = df[col].fillna("").astype(str).str.strip()

    def parse_authors(value: str) -> list[str]:
        if not value:
            return []
        if ";" in value:
            parts = value.split(";")
        elif "|" in value:
            parts = value.split("|")
        elif "\n" in value:
            parts = value.splitlines()
        elif "," in value:
            parts = value.split(",")
        else:
            return [value]
        return [p.strip() for p in parts if p.strip()]

    parsed = s.map(parse_authors)
    author_counts = parsed.map(len)

    counter = Counter()
    for authors in parsed:
        counter.update(authors)

    top = counter.most_common(top_n)

    if top:
        labels = [x[0] for x in top][::-1]
        values = [x[1] for x in top][::-1]
        plt.figure(figsize=(11, max(6, min(14, len(labels) * 0.35))))
        plt.barh(labels, values, color=PRIMARY_COLOR)
        plt.title(f"Top {len(top)} Authors", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Paper Occurrences", fontsize=11, fontweight="bold")
        plt.ylabel("Author", fontsize=11, fontweight="bold")
        plt.grid(axis="x", alpha=0.3, linestyle="--")
        save_current_plot(VIZ_DIR / "top_authors.png")

    return {
        "status": "analyzed",
        "column": col,
        "rows": int(len(df)),
        "rows_with_authors": int((author_counts > 0).sum()),
        "mean_authors_per_row": float(author_counts.mean()),
        "median_authors_per_row": float(author_counts.median()),
        "max_authors_per_row": int(author_counts.max()) if len(author_counts) else 0,
        "unique_parsed_authors": int(len(counter)),
        "top_authors": [
            {"author": name, "count": count}
            for name, count in top
        ],
        "parsing_note": (
            "Author counts are estimates based on detected separators. "
            "Exact counts depend on the source formatting."
        ),
    }


def analyze_numeric(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_outliers = int(((s < lower) | (s > upper)).sum())

        rows.append({
            "column": str(col),
            "count": int(s.count()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "p01": float(s.quantile(0.01)),
            "p99": float(s.quantile(0.99)),
            "iqr_outlier_count": iqr_outliers,
            "negative_count": int((s < 0).sum()),
            "zero_count": int((s == 0).sum()),
        })

        outliers.append({
            "column": str(col),
            "method": "IQR 1.5x",
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": iqr_outliers,
            "outlier_percentage": round(iqr_outliers / max(len(s), 1) * 100, 4),
        })

    pd.DataFrame(rows).to_csv(
        TABLE_DIR / "numeric_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(outliers).to_csv(
        TABLE_DIR / "numeric_outlier_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for col in numeric_cols[:8]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        upper = s.quantile(0.99)
        plt.figure(figsize=(10, 5))
        if sns is not None:
            sns.histplot(s.clip(upper=upper), kde=True, color=PRIMARY_COLOR, bins=40)
        else:
            plt.hist(s.clip(upper=upper), bins=40, color=PRIMARY_COLOR, edgecolor="white")
        plt.title(f"Distribution — {col}", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel(str(col), fontsize=11, fontweight="bold")
        plt.ylabel("Count", fontsize=11, fontweight="bold")
        plt.grid(alpha=0.3, linestyle="--")
        save_current_plot(VIZ_DIR / f"numeric_{slug(str(col))}.png")

    return {
        "numeric_columns": [str(c) for c in numeric_cols],
        "statistics": rows,
        "outliers": outliers,
    }


def analyze_categorical(df: pd.DataFrame) -> dict[str, Any]:
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    rows: list[dict[str, Any]] = []

    for col in categorical_cols:
        s = df[col].dropna().astype(str)
        if s.empty:
            continue

        unique = int(s.nunique())
        counts = s.value_counts()
        rows.append({
            "column": str(col),
            "unique_categories": unique,
            "top_category": str(counts.index[0]),
            "top_category_count": int(counts.iloc[0]),
            "top_category_percentage": round(float(counts.iloc[0] / len(s) * 100), 4),
            "rare_categories_count": int((counts <= max(5, len(s) * 0.001)).sum()),
            "high_cardinality": bool(unique > min(1000, len(s) * 0.5)),
            "near_unique": bool(unique / max(len(s), 1) > 0.98),
        })

    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "categorical_statistics.csv", index=False, encoding="utf-8-sig")

    return {
        "categorical_columns": [str(c) for c in categorical_cols],
        "statistics": rows,
    }


def analyze_correlations(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number])

    if numeric.shape[1] < 2:
        return {"status": "not_available", "reason": "Fewer than two numeric columns."}

    keep = []
    for col in numeric.columns:
        nunique_ratio = numeric[col].nunique(dropna=True) / max(len(numeric), 1)
        if nunique_ratio < 0.98:
            keep.append(col)

    if len(keep) < 2:
        return {
            "status": "not_available",
            "reason": "No meaningful pair of numeric fields after excluding near-unique columns.",
        }

    selected = numeric[keep]
    pearson = selected.corr(method="pearson")
    spearman = selected.corr(method="spearman")

    plt.figure(figsize=(max(8, len(keep) * 0.8), max(6, len(keep) * 0.7)))
    if sns is not None:
        sns.heatmap(
            pearson,
            annot=len(keep) <= 12,
            fmt=".2f",
            center=0,
            cmap="vlag",
            square=True,
            cbar_kws={"shrink": 0.8},
        )
    else:
        plt.imshow(pearson, aspect="auto", cmap="coolwarm")
        plt.colorbar()
        plt.xticks(range(len(keep)), keep, rotation=45, ha="right")
        plt.yticks(range(len(keep)), keep)

    plt.title("Pearson Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    save_current_plot(VIZ_DIR / "correlation_matrix.png")

    return {
        "status": "analyzed",
        "columns": [str(c) for c in keep],
        "pearson": pearson.round(4).to_dict(),
        "spearman": spearman.round(4).to_dict(),
    }


# ---------------------------------------------------------------------------
# Additional Analysis: Missing Value Heatmap Pattern
# ---------------------------------------------------------------------------

def create_missing_pattern_visualization(df: pd.DataFrame) -> None:
    if df.empty or len(df.columns) == 0:
        return
    plt.figure(figsize=(12, 6))
    if sns is not None:
        sns.heatmap(df.isna(), cbar=False, cmap="viridis", yticklabels=False)
    else:
        plt.imshow(df.isna().values, aspect="auto", cmap="binary")
    plt.title("Missing Value Distribution Pattern", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Columns", fontsize=11, fontweight="bold")
    plt.ylabel("Rows", fontsize=11, fontweight="bold")
    save_current_plot(VIZ_DIR / "missing_pattern_heatmap.png")


# ---------------------------------------------------------------------------
# Semantic Readiness & Scoring
# ---------------------------------------------------------------------------

def build_canonical_text_preview(
    df: pd.DataFrame,
    special: dict[str, Optional[str]],
) -> tuple[Optional[pd.Series], dict[str, Any]]:
    title = special.get("title")
    abstract = special.get("abstract")

    options: dict[str, Any] = {}

    if title:
        t = df[title].fillna("").astype(str).str.strip()
        options["title"] = {
            "available": True,
            "nonempty_count": int(t.ne("").sum()),
            "coverage_percentage": round(float(t.ne("").mean() * 100), 4),
            "mean_chars": float(t.map(len).mean()),
        }

    if abstract:
        a = df[abstract].fillna("").astype(str).str.strip()
        options["abstract"] = {
            "available": True,
            "nonempty_count": int(a.ne("").sum()),
            "coverage_percentage": round(float(a.ne("").mean() * 100), 4),
            "mean_chars": float(a.map(len).mean()),
        }

    if title and abstract:
        combined = (
            df[title].fillna("").astype(str).str.strip()
            + " "
            + df[abstract].fillna("").astype(str).str.strip()
        ).str.strip()

        options["title_plus_abstract"] = {
            "available": True,
            "nonempty_count": int(combined.ne("").sum()),
            "coverage_percentage": round(float(combined.ne("").mean() * 100), 4),
            "mean_chars": float(combined.map(len).mean()),
        }
    else:
        combined = None

    return combined, options


def score_text_quality(
    title_available: bool,
    abstract_available: bool,
    title_coverage: float,
    abstract_coverage: float,
    duplicate_rate: float,
) -> tuple[float, dict[str, float]]:
    components = {
        "title_coverage": min(title_coverage, 100.0),
        "abstract_coverage": min(abstract_coverage, 100.0),
        "uniqueness": max(0.0, 100.0 - min(duplicate_rate, 100.0)),
        "metadata_availability": (
            100.0 if title_available and abstract_available
            else 75.0 if title_available or abstract_available
            else 0.0
        ),
    }

    score = (
        components["title_coverage"] * 0.25
        + components["abstract_coverage"] * 0.40
        + components["uniqueness"] * 0.20
        + components["metadata_availability"] * 0.15
    )

    return round(score, 2), components


def assess_semantic_readiness(
    df: pd.DataFrame,
    special: dict[str, Optional[str]],
    duplicate_info: dict[str, Any],
    text_info: dict[str, Any],
) -> dict[str, Any]:
    title = special.get("title")
    abstract = special.get("abstract")
    categories = special.get("categories")
    identifier = special.get("id")
    date = special.get("date")

    title_cov = 0.0
    abstract_cov = 0.0

    if title:
        title_cov = float(
            df[title].fillna("").astype(str).str.strip().ne("").mean() * 100
        )
    if abstract:
        abstract_cov = float(
            df[abstract].fillna("").astype(str).str.strip().ne("").mean() * 100
        )

    duplicate_rate = 0.0
    for check in duplicate_info.get("checks", []):
        if check.get("check") == "duplicate_title_abstract" and check.get("duplicate_percentage") is not None:
            duplicate_rate = float(check["duplicate_percentage"])
            break

    score, components = score_text_quality(
        bool(title),
        bool(abstract),
        title_cov,
        abstract_cov,
        duplicate_rate,
    )

    if title and abstract and title_cov >= 95 and abstract_cov >= 90 and duplicate_rate < 5:
        readiness = "READY_WITH_STANDARD_PREPROCESSING"
    elif (title or abstract) and max(title_cov, abstract_cov) >= 70:
        readiness = "CONDITIONALLY_READY"
    else:
        readiness = "NOT_READY_WITHOUT_ADDITIONAL_DATA_CLEANING"

    if title and abstract:
        embedding_recommendation = "title + abstract"
    elif abstract:
        embedding_recommendation = "abstract"
    elif title:
        embedding_recommendation = "title"
    else:
        embedding_recommendation = None

    metadata_fields = [
        c for c in [identifier, categories, date, special.get("authors"), special.get("journal")]
        if c
    ]

    limitations = []
    if not title:
        limitations.append("No confidently identified title field.")
    if not abstract:
        limitations.append("No confidently identified abstract field; semantic retrieval quality may be limited.")
    if duplicate_rate >= 5:
        limitations.append(f"Title+abstract duplicate rate is {duplicate_rate:.2f}%.")
    if abstract and abstract_cov < 90:
        limitations.append(f"Abstract coverage is only {abstract_cov:.2f}%.")

    return {
        "readiness_status": readiness,
        "readiness_score": score,
        "score_components": components,
        "score_methodology": (
            "Engineering heuristic only: title coverage 25%, abstract coverage 40%, "
            "uniqueness 20%, metadata availability 15%. It is not a model accuracy metric."
        ),
        "identified_fields": {
            "title": title,
            "abstract": abstract,
            "identifier": identifier,
            "categories": categories,
            "date": date,
        },
        "title_coverage_percentage": round(title_cov, 4),
        "abstract_coverage_percentage": round(abstract_cov, 4),
        "duplicate_title_abstract_percentage": duplicate_rate,
        "recommended_embedding_text": embedding_recommendation,
        "recommended_metadata_fields": metadata_fields,
        "recommended_exclusions": [
            c for c in df.columns
            if c in {identifier, date} and c is not None
        ],
        "limitations": limitations,
        "next_stage": [
            "Create a non-destructive processed corpus.",
            "Normalize text while preserving scientific notation.",
            "Remove only demonstrably invalid/empty retrieval records.",
            "Generate scientific embeddings.",
            "Build and evaluate FAISS retrieval.",
        ],
    }


def mode1_feasibility(
    df: pd.DataFrame,
    special: dict[str, Optional[str]],
) -> list[dict[str, Any]]:
    title = special.get("title")
    abstract = special.get("abstract")
    fulltext_signals = [
        c for c in df.columns
        if any(
            token in normalize_column_name(c)
            for token in ["full_text", "fulltext", "body", "content", "paper_text", "document"]
        )
    ]

    has_fulltext = bool(fulltext_signals)

    rows = [
        {
            "feature": "Short Summary",
            "available_directly": bool(abstract or fulltext_signals),
            "required_fields": "Abstract or full paper text",
            "can_be_inferred": bool(abstract or fulltext_signals),
            "risk_level": "Low" if has_fulltext or abstract else "High",
            "recommendation": "LLM summarization grounded in retrieved paper text.",
        },
        {
            "feature": "Methodology",
            "available_directly": has_fulltext,
            "required_fields": "Full paper text, preferably methodology sections",
            "can_be_inferred": bool(abstract),
            "risk_level": "High" if not has_fulltext else "Medium",
            "recommendation": (
                "Use full text for reliable extraction; abstract-only inference must be labeled as limited."
            ),
        },
        {
            "feature": "Dataset",
            "available_directly": has_fulltext,
            "required_fields": "Full paper text / experiments section",
            "can_be_inferred": False,
            "risk_level": "High" if not has_fulltext else "Medium",
            "recommendation": "Extract named datasets from grounded full text.",
        },
        {
            "feature": "Model",
            "available_directly": has_fulltext,
            "required_fields": "Full paper text / methodology",
            "can_be_inferred": bool(abstract),
            "risk_level": "High" if not has_fulltext else "Medium",
            "recommendation": "Extract model names and variants from grounded paper text.",
        },
        {
            "feature": "Key Findings",
            "available_directly": bool(abstract or fulltext_signals),
            "required_fields": "Abstract or full paper text",
            "can_be_inferred": bool(abstract or fulltext_signals),
            "risk_level": "Low" if has_fulltext or abstract else "High",
            "recommendation": "Ground findings in abstract/full text and cite evidence.",
        },
        {
            "feature": "Strong Points",
            "available_directly": False,
            "required_fields": "Full paper text + comparative/evidence analysis",
            "can_be_inferred": True,
            "risk_level": "High",
            "recommendation": "Generate evidence-based strengths; clearly distinguish analysis from author claims.",
        },
        {
            "feature": "Weak Points",
            "available_directly": False,
            "required_fields": "Full paper text + limitations/results analysis",
            "can_be_inferred": True,
            "risk_level": "High",
            "recommendation": "Generate evidence-based limitations; avoid unsupported criticism.",
        },
    ]

    return rows


def build_quality_scorecard(
    df: pd.DataFrame,
    schema: pd.DataFrame,
    duplicate_info: dict[str, Any],
    text_info: dict[str, Any],
    temporal_info: dict[str, Any],
    category_info: dict[str, Any],
    semantic_info: dict[str, Any],
) -> list[dict[str, Any]]:
    n = max(len(df), 1)

    missing_rate = float(df.isna().mean().mean() * 100)
    completeness = max(0.0, 100.0 - min(missing_rate, 100.0))

    exact_dup_rate = float(
        duplicate_info.get("exact_duplicate_rows", 0) / n * 100
    )
    uniqueness = max(0.0, 100.0 - min(exact_dup_rate, 100.0))

    invalid_dates = int(temporal_info.get("invalid_count", 0)) if temporal_info.get("status") == "analyzed" else 0
    valid_temporal = max(0.0, 100.0 - invalid_dates / n * 100) if temporal_info.get("status") == "analyzed" else 50.0

    text_cols = [c for c in schema["column"].tolist() if c in text_info and not c.startswith("_")]
    if text_cols:
        text_empty_rates = []
        for c in text_cols:
            rows = text_info[c]["rows"]
            empty = text_info[c]["empty_count"]
            text_empty_rates.append(empty / max(rows, 1) * 100)
        text_quality = max(0.0, 100.0 - float(np.mean(text_empty_rates)))
    else:
        text_quality = 0.0

    metadata_quality = 100.0 if semantic_info["identified_fields"].get("title") else 50.0
    if semantic_info["identified_fields"].get("abstract"):
        metadata_quality = min(100.0, metadata_quality + 25.0)

    category_quality = 100.0 if category_info.get("status") == "analyzed" else 50.0

    retrieval_readiness = float(semantic_info["readiness_score"])

    rows = [
        {
            "dimension": "Completeness",
            "score": round(completeness, 2),
            "severity": severity_from_score(completeness),
            "finding": f"Average cell-level missingness is {missing_rate:.2f}%.",
            "recommendation": "Handle missing values field-by-field; do not blindly impute scientific text.",
        },
        {
            "dimension": "Uniqueness",
            "score": round(uniqueness, 2),
            "severity": severity_from_score(uniqueness),
            "finding": f"Exact duplicate-row rate is {exact_dup_rate:.2f}%.",
            "recommendation": "Investigate duplicates before indexing; retain raw data unchanged.",
        },
        {
            "dimension": "Consistency",
            "score": round(max(0.0, 100.0 - min(exact_dup_rate * 2, 100.0)), 2),
            "severity": severity_from_score(max(0.0, 100.0 - min(exact_dup_rate * 2, 100.0))),
            "finding": "Consistency is estimated from duplicate signals and structural stability.",
            "recommendation": "Normalize metadata representations in the processed corpus.",
        },
        {
            "dimension": "Validity",
            "score": round(min(completeness, uniqueness), 2),
            "severity": severity_from_score(min(completeness, uniqueness)),
            "finding": "Validity combines observed completeness and uniqueness signals.",
            "recommendation": "Review suspicious values and outliers before indexing.",
        },
        {
            "dimension": "Text Quality",
            "score": round(text_quality, 2),
            "severity": severity_from_score(text_quality),
            "finding": "Text quality is based on empty-text rates across detected text fields.",
            "recommendation": "Run targeted scientific-text normalization and quality filtering.",
        },
        {
            "dimension": "Metadata Quality",
            "score": round(metadata_quality, 2),
            "severity": severity_from_score(metadata_quality),
            "finding": "Metadata score reflects detected title/abstract availability.",
            "recommendation": "Retain stable identifiers, dates, categories and author metadata.",
        },
        {
            "dimension": "Temporal Quality",
            "score": round(valid_temporal, 2),
            "severity": severity_from_score(valid_temporal),
            "finding": (
                f"{invalid_dates} temporal values were not parseable."
                if temporal_info.get("status") == "analyzed"
                else "No reliable temporal field was detected."
            ),
            "recommendation": "Validate dates before using temporal filters or trends.",
        },
        {
            "dimension": "Category Quality",
            "score": round(category_quality, 2),
            "severity": severity_from_score(category_quality),
            "finding": (
                "Research categories were detected and analyzed."
                if category_info.get("status") == "analyzed"
                else "No reliable category field was detected."
            ),
            "recommendation": "Preserve category labels as metadata; analyze multi-label structure.",
        },
        {
            "dimension": "Retrieval Readiness",
            "score": retrieval_readiness,
            "severity": severity_from_score(retrieval_readiness),
            "finding": semantic_info["readiness_status"],
            "recommendation": "Proceed to preprocessing/embedding only after reviewing this report.",
        },
    ]

    pd.DataFrame(rows).to_csv(
        TABLE_DIR / "data_quality_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(11, 6))
    if sns is not None:
        sns.barplot(x=plot_df["score"], y=plot_df["dimension"], color=PRIMARY_COLOR)
    else:
        plt.barh(plot_df["dimension"], plot_df["score"], color=PRIMARY_COLOR)

    plt.xlim(0, 100)
    plt.xlabel("Score (Engineering Heuristic / 100)", fontsize=11, fontweight="bold")
    plt.ylabel("Quality Dimension", fontsize=11, fontweight="bold")
    plt.title("Dataset Quality Scorecard", fontsize=14, fontweight="bold", pad=15)
    plt.grid(axis="x", alpha=0.3, linestyle="--")
    save_current_plot(VIZ_DIR / "data_quality_summary.png")

    return rows


def severity_from_score(score: float) -> str:
    if score >= 90:
        return "LOW"
    if score >= 75:
        return "MODERATE"
    if score >= 50:
        return "HIGH"
    return "CRITICAL"


# ---------------------------------------------------------------------------
# Overview Visualizations
# ---------------------------------------------------------------------------

def create_overview_visualization(
    info: DatasetInfo,
    schema: pd.DataFrame,
    duplicate_info: dict[str, Any],
) -> None:
    labels = ["Rows", "Columns", "Duplicate Rows"]
    values = [info.row_count, info.column_count, duplicate_info.get("exact_duplicate_rows", 0)]

    plt.figure(figsize=(10, 6))
    if sns is not None:
        sns.barplot(x=labels, y=values, palette="Blues_d")
    else:
        plt.bar(labels, values, color=PRIMARY_COLOR)

    plt.title("Dataset Overview", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Count", fontsize=11, fontweight="bold")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    save_current_plot(VIZ_DIR / "dataset_overview.png")


def create_missing_visualization(schema: pd.DataFrame) -> None:
    top = schema.sort_values("null_percentage", ascending=False).head(25)

    plt.figure(figsize=(11, max(6, min(14, len(top) * 0.35))))
    plot_df = top.sort_values("null_percentage")
    
    if sns is not None:
        sns.barplot(x=plot_df["null_percentage"], y=plot_df["column"], color=ACCENT_COLOR)
    else:
        plt.barh(plot_df["column"], plot_df["null_percentage"], color=ACCENT_COLOR)

    plt.xlabel("Missing Values (%)", fontsize=11, fontweight="bold")
    plt.ylabel("Column", fontsize=11, fontweight="bold")
    plt.title("Missing Values by Column", fontsize=14, fontweight="bold", pad=15)
    plt.xlim(0, 100)
    plt.grid(axis="x", alpha=0.3, linestyle="--")
    save_current_plot(VIZ_DIR / "missing_values.png")


def create_duplicate_visualization(duplicate_info: dict[str, Any]) -> None:
    checks = duplicate_info.get("checks", [])
    rows = [
        x for x in checks
        if x.get("duplicate_percentage") is not None
    ]
    if not rows:
        return

    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(11, 6))
    if sns is not None:
        sns.barplot(x=plot_df["check"], y=plot_df["duplicate_percentage"], palette="muted")
    else:
        plt.bar(plot_df["check"], plot_df["duplicate_percentage"], color=PRIMARY_COLOR)

    plt.ylabel("Duplicated Rows (%)", fontsize=11, fontweight="bold")
    plt.xlabel("Duplicate Check", fontsize=11, fontweight="bold")
    plt.title("Duplicate Analysis", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    save_current_plot(VIZ_DIR / "duplicate_analysis.png")


# ---------------------------------------------------------------------------
# Markdown Report Generation
# ---------------------------------------------------------------------------

def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No applicable data._"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.head(max_rows).to_string(index=False) + "\n```"


def generate_report(
    info: DatasetInfo,
    schema: pd.DataFrame,
    special: dict[str, Optional[str]],
    duplicate_info: dict[str, Any],
    text_info: dict[str, Any],
    category_info: dict[str, Any],
    temporal_info: dict[str, Any],
    author_info: dict[str, Any],
    numeric_info: dict[str, Any],
    categorical_info: dict[str, Any],
    correlation_info: dict[str, Any],
    semantic_info: dict[str, Any],
    feasibility: list[dict[str, Any]],
    quality_scorecard: list[dict[str, Any]],
) -> None:
    report: list[str] = []

    report.append("# Dataset Analysis Report")
    report.append("")
    report.append("> Generated automatically by script execution. "
                  "Measured statistics are separated from engineering recommendations.")
    report.append("")

    report.append("## 1. Executive Summary")
    report.append("")
    report.append(f"- Dataset: `{Path(info.path).name}`")
    report.append(f"- Rows: **{info.row_count:,}**")
    report.append(f"- Columns: **{info.column_count:,}**")
    report.append(f"- File Size: **{info.file_size_mb:.2f} MB**")
    report.append(f"- In-Memory Size: **{info.memory_mb:.2f} MB**")
    report.append(f"- Exact Duplicate Rows: **{duplicate_info.get('exact_duplicate_rows', 0):,}**")
    report.append(f"- Semantic-Search Readiness: **{semantic_info['readiness_status']}**")
    report.append(f"- Readiness Score: **{semantic_info['readiness_score']:.2f}/100**")
    report.append("")

    report.append("## 2. Dataset Identity")
    report.append("")
    report.append(f"- Path: `{info.path}`")
    report.append(f"- File Type: `{info.file_type}`")
    report.append(f"- Encoding: `{info.encoding or 'not applicable/detected'}`")
    report.append(f"- Delimiter: `{info.delimiter or 'not applicable'}`")
    report.append("")

    report.append("## 3. Schema")
    report.append("")
    report.append(md_table(schema[
        ["column", "dtype", "non_null_count", "null_count",
         "null_percentage", "unique_count", "inferred_role"]
    ], 100))
    report.append("")

    report.append("## 4. Missing Values")
    report.append("")
    report.append(md_table(
        schema[
            ["column", "null_count", "null_percentage",
             "empty_string_count", "whitespace_only_count", "null_like_count"]
        ].sort_values("null_percentage", ascending=False),
        100,
    ))
    report.append("")

    report.append("## 5. Duplicate Analysis")
    report.append("")
    report.append(md_table(pd.DataFrame(duplicate_info.get("checks", []))))
    report.append("")

    report.append("## 6. Data Quality Scorecard")
    report.append("")
    report.append(md_table(pd.DataFrame(quality_scorecard)))
    report.append("")

    report_path = REPORT_DIR / "dataset_analysis_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    LOGGER.info("Report written: %s", report_path)


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete audit of the research dataset with enhanced visualizations."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional explicit dataset path.",
    )
    parser.add_argument(
        "--top-categories",
        type=int,
        default=20,
        help="Number of top categories or authors to include.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5000,
        help="Reserved sample size.",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    started = datetime.now(timezone.utc)

    LOGGER.info("=" * 72)
    LOGGER.info("DATASET ANALYSIS STARTED")
    LOGGER.info("=" * 72)

    try:
        candidates = discover_datasets(args.dataset)
        dataset_path = candidates[0].resolve()
        LOGGER.info("Selected dataset: %s", dataset_path)

        df, info = read_dataset(dataset_path, sample_rows=args.sample_rows)

        special = identify_special_columns(df)
        schema, schema_json = analyze_schema(df, special)

        duplicate_info = duplicate_analysis(df, special)
        text_info = analyze_text_columns(df, special)
        category_info = analyze_categories(df, special, args.top_categories)
        temporal_info = analyze_temporal(df, special)
        author_info = analyze_authors(df, special, args.top_categories)
        numeric_info = analyze_numeric(df)
        categorical_info = analyze_categorical(df)
        correlation_info = analyze_correlations(df)

        create_overview_visualization(info, schema, duplicate_info)
        create_missing_visualization(schema)
        create_missing_pattern_visualization(df)
        create_duplicate_visualization(duplicate_info)

        _, canonical_options = build_canonical_text_preview(df, special)
        semantic_info = assess_semantic_readiness(df, special, duplicate_info, text_info)
        semantic_info["canonical_text_options"] = canonical_options

        feasibility = mode1_feasibility(df, special)
        quality_scorecard = build_quality_scorecard(
            df, schema, duplicate_info, text_info, temporal_info, category_info, semantic_info
        )

        generate_report(
            info, schema, special, duplicate_info, text_info, category_info,
            temporal_info, author_info, numeric_info, categorical_info,
            correlation_info, semantic_info, feasibility, quality_scorecard
        )

        finished = datetime.now(timezone.utc)
        elapsed = (finished - started).total_seconds()
        LOGGER.info("Dataset analysis completed in %.2f seconds", elapsed)
        return 0

    except KeyboardInterrupt:
        LOGGER.warning("Analysis interrupted by user.")
        return 130
    except Exception:
        LOGGER.exception("Dataset analysis failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())