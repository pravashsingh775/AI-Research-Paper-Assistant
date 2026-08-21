"""Production-Grade Semantic Search Engine for AI Research Paper Assistant.

This module implements a production-ready, modular, and explainable semantic search 
infrastructure using FAISS, BAAI/bge-small-en-v1.5 sentence embeddings, and pandas metadata.
It provides natural-language querying, metadata filtering, similarity score normalization, 
retrieval confidence scoring, explainable retrieval reasons, automated benchmarking, 
validation suite, and diagnostic plot generation.

Location: src/rag/semantic_search_engine.py
"""

import ast
import datetime
import json
import logging
import os
import pathlib
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sentence_transformers import SentenceTransformer

# Suppress non-critical matplotlib and pandas warnings
warnings.filterwarnings("ignore")

# Force matplotlib to use 'Agg' backend for headless plot generation
plt.switch_backend("Agg")

# =============================================================================
# Centralized Configuration
# =============================================================================
MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
EXPECTED_DIMENSION: int = 384
DEFAULT_TOP_K: int = 10
MAX_TOP_K: int = 100
DEFAULT_FILTER_FETCH_MULTIPLIER: int = 12
MAX_FILTER_FETCH: int = 10000
MIN_CONTENT_WORDS: int = 8
SIMILARITY_METRIC: str = "INNER_PRODUCT"  # Cosine similarity for L2-normalized vectors

# BGE retrieval instruction. This improves query/passage alignment for short
# natural-language search queries while remaining backward compatible with the
# already-built FAISS document embeddings.
BGE_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

# The current project metadata does not contain a dedicated `abstract` column.
# `combined_text` is the authoritative scientific text field.
CONTENT_FIELD_PRIORITY: Tuple[str, ...] = (
    "combined_text",
    "abstract",
    "summary",
    "text",
    "content",
    "title",
)

INPUT_VECTOR_DB_DIR: str = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "dataset/vector_database")
INPUT_EMBEDDINGS_DIR: str = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "dataset/embeddings")

FAISS_INDEX_PATH: str = os.path.join(INPUT_VECTOR_DB_DIR, "faiss_index.bin")
METADATA_PATH: str = os.path.join(INPUT_EMBEDDINGS_DIR, "paper_metadata.parquet")
PAPER_IDS_PATH: str = os.path.join(INPUT_EMBEDDINGS_DIR, "paper_ids.npy")

REPORTS_DIR: str = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "outputs/reports/semantic_search")
PLOTS_DIR: str = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "outputs/reports/semantic_search/plots")
LOGS_DIR: str = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "outputs/reports/semantic_search/logs")


# =============================================================================
# Logging Setup
# =============================================================================
def setup_logging() -> logging.Logger:
    """Configures structured console and file logging for search analytics."""
    logger = logging.getLogger("SemanticSearchEngine")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        pathlib.Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
        pathlib.Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(LOGS_DIR, "search_engine.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


logger = setup_logging()


# =============================================================================
# Helper Utilities & System Setup
# =============================================================================
def setup_directories() -> None:
    """Creates directory trees for storing output reports, visual plots, and session logs."""
    pathlib.Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Core Semantic Search Engine Architecture
# =============================================================================
class SemanticSearchEngine:
    """Production-grade Semantic Search Engine powering RAG and Paper Retrieval."""

    def __init__(
        self,
        index_path: str = FAISS_INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        paper_ids_path: str = PAPER_IDS_PATH,
        model_name: str = MODEL_NAME,
    ) -> None:
        """Initializes search engine by loading vector index, metadata table, and encoder model.

        Args:
            index_path: Path to serialized binary FAISS index file.
            metadata_path: Path to paper metadata parquet file.
            paper_ids_path: Path to paper IDs numpy binary file.
            model_name: Identifier for SentenceTransformer encoder model.
        """
        setup_directories()
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.paper_ids_path = paper_ids_path
        self.model_name = model_name

        self.index: Optional[faiss.Index] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self.paper_ids: Optional[np.ndarray] = None
        self.encoder: Optional[SentenceTransformer] = None

        self._load_resources()

    def _load_resources(self) -> None:
        """Loads and verifies FAISS binary index, metadata dataframe, and sentence encoder."""
        logger.info("Initializing Semantic Search Engine resource loader...")

        # 1. Load FAISS Binary Index
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index binary not found at: {self.index_path}")
        logger.info(f"Loading FAISS binary vector index from: '{self.index_path}'...")
        t0 = time.time()
        self.index = faiss.read_index(self.index_path)
        logger.info(
            f"FAISS index successfully loaded in {time.time() - t0:.2f}s "
            f"(Total Vectors: {self.index.ntotal:,}, Dim: {self.index.d})."
        )

        if self.index.d != EXPECTED_DIMENSION:
            raise ValueError(
                f"FAISS index dimension mismatch: expected {EXPECTED_DIMENSION}, got {self.index.d}."
            )

        # 2. Load Paper IDs Array
        if not os.path.exists(self.paper_ids_path):
            raise FileNotFoundError(f"Paper IDs file not found at: {self.paper_ids_path}")
        logger.info(f"Loading paper IDs array from: '{self.paper_ids_path}'...")
        self.paper_ids = np.load(self.paper_ids_path, allow_pickle=True)

        # 3. Load Paper Metadata Parquet
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata Parquet file not found at: {self.metadata_path}")
        logger.info(f"Loading paper metadata table from: '{self.metadata_path}'...")
        t0 = time.time()
        self.metadata_df = pd.read_parquet(self.metadata_path)
        logger.info(
            f"Metadata parquet successfully loaded in {time.time() - t0:.2f}s "
            f"({len(self.metadata_df):,} records)."
        )
        logger.info(f"Metadata columns: {list(self.metadata_df.columns)}")
        self._validate_metadata_schema()

        # Parity Validation
        if not (self.index.ntotal == len(self.paper_ids) == len(self.metadata_df)):
            raise ValueError(
                f"Row parity mismatch across index ({self.index.ntotal}), "
                f"paper_ids ({len(self.paper_ids)}), and metadata ({len(self.metadata_df)})."
            )

        # 4. Load Sentence Transformer Encoder
        logger.info(f"Initializing query embedding encoder model: '{self.model_name}'...")
        t0 = time.time()
        self.encoder = SentenceTransformer(self.model_name)
        logger.info(f"Encoder model successfully initialized in {time.time() - t0:.2f}s.")

    @staticmethod
    def _clean_text(value: Any) -> str:
        """Normalize metadata text without destroying scientific content."""
        if value is None:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return ""
        text = str(value).replace("\\r", " ").replace("\\n", " ")
        return re.sub(r"\\s+", " ", text).strip()

    def _extract_content(self, row: pd.Series) -> Tuple[str, str]:
        """Return the best available scientific text and its source field.

        Priority is intentionally `combined_text` first because that is the
        field present in the project's 287,419-row metadata table.
        """
        for field in CONTENT_FIELD_PRIORITY:
            if field not in row.index:
                continue
            content = self._clean_text(row.get(field))
            if not content:
                continue

            # Avoid returning a placeholder as evidence.
            if content.lower() in {"n/a", "na", "none", "nan", "null"}:
                continue

            return content, field

        return "N/A", "none"

    @staticmethod
    def _parse_authors(value: Any) -> List[str]:
        """Safely parse the project's stringified author-list representation."""
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        raw = str(value).strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except (ValueError, SyntaxError):
            pass
        return [p.strip() for p in raw.split(",") if p.strip()]

    def _validate_metadata_schema(self) -> None:
        """Validate the minimum schema required by search and RAG."""
        required = {"id", "title"}
        missing = sorted(required - set(self.metadata_df.columns))
        if missing:
            raise ValueError(
                f"Metadata schema missing required columns: {missing}. "
                f"Available columns: {list(self.metadata_df.columns)}"
            )

        if not any(field in self.metadata_df.columns for field in CONTENT_FIELD_PRIORITY):
            raise ValueError(
                "No usable scientific text field found. Expected one of: "
                f"{list(CONTENT_FIELD_PRIORITY)}"
            )

        if self.metadata_df["id"].duplicated().any():
            dup_count = int(self.metadata_df["id"].duplicated().sum())
            logger.warning("Metadata contains %d duplicated paper IDs.", dup_count)

    def embed_query(self, query_text: str) -> Tuple[np.ndarray, float]:
        """Encodes query string into a unit L2-normalized dense embedding vector.

        Args:
            query_text: Natural language user prompt or research query string.

        Returns:
            Tuple of (L2-normalized float32 numpy vector, encoding_time_seconds).
        """
        if not query_text or not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("Search query string must be a non-empty, non-whitespace string.")

        t0 = time.time()
        query_text_clean = query_text.strip()
        encoded_query = BGE_QUERY_PREFIX + query_text_clean

        query_vec = self.encoder.encode(
            [encoded_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        encoding_time = time.time() - t0

        return query_vec, encoding_time

    def apply_filters(
        self,
        candidate_indices: np.ndarray,
        distances: np.ndarray,
        year_range: Optional[Tuple[int, int]] = None,
        category: Optional[str] = None,
        broad_domain: Optional[str] = None,
        author: Optional[str] = None,
        is_single_author: Optional[bool] = None,
        has_multiple_authors: Optional[bool] = None,
    ) -> Tuple[List[int], List[float]]:
        """Filters ANN search candidates against structured metadata fields.

        Args:
            candidate_indices: Array of integer row indices from FAISS.
            distances: Array of raw similarity scores from FAISS.
            year_range: Tuple of (min_year, max_year) for publication date filtering.
            category: Specific arXiv category string filter.
            broad_domain: High-level mapped scientific domain string filter.
            author: Specific author name string filter.
            is_single_author: Boolean flag for solo-authored paper filter.
            has_multiple_authors: Boolean flag for co-authored paper filter.

        Returns:
            Tuple of (filtered_indices_list, filtered_distances_list).
        """
        filtered_indices = []
        filtered_distances = []

        for idx, dist in zip(candidate_indices, distances):
            if idx < 0 or idx >= len(self.metadata_df):
                continue

            row = self.metadata_df.iloc[idx]

            # 1. Publication Year Range Filter
            if year_range is not None:
                pub_year = row.get("publication_year", None)
                if pd.notnull(pub_year):
                    min_y, max_y = year_range
                    if not (min_y <= int(pub_year) <= max_y):
                        continue

            # 2. Specific Category Filter
            if category is not None:
                row_cat = str(row.get("category", "")).lower()
                if category.lower() not in row_cat:
                    continue

            # 3. Broad Domain Filter
            if broad_domain is not None:
                row_domain = str(row.get("broad_domain", "")).lower()
                if broad_domain.lower() != row_domain:
                    continue

            # 4. Author Name Filter
            if author is not None:
                row_authors = str(row.get("authors", "")).lower()
                if author.lower() not in row_authors:
                    continue

            # 5. Author-count filters. The current metadata does not expose
            # dedicated boolean columns, so derive them from the authors field.
            authors_list = self._parse_authors(row.get("authors", ""))
            author_count = len(authors_list)

            if is_single_author is not None:
                if (author_count == 1) != is_single_author:
                    continue

            if has_multiple_authors is not None:
                if (author_count > 1) != has_multiple_authors:
                    continue

            filtered_indices.append(int(idx))
            filtered_distances.append(float(dist))

        return filtered_indices, filtered_distances

    @staticmethod
    def _compute_confidence_score(similarity_score: float) -> float:
        """Calculates a normalized 0-100% retrieval confidence score from similarity.

        Args:
            similarity_score: Raw Cosine Similarity score [-1.0, 1.0].

        Returns:
            Normalized confidence percentage value [0.0, 100.0].
        """
        # Linear rescaling from Cosine Similarity range [-1, 1] to [0, 100]
        confidence = ((similarity_score + 1.0) / 2.0) * 100.0
        return round(float(np.clip(confidence, 0.0, 100.0)), 2)

    @staticmethod
    def _generate_explainable_reason(
        query: str, row: pd.Series, similarity_score: float, rank: int
    ) -> str:
        """Generates a structured, human-readable justification for paper retrieval.

        Args:
            query: User search query.
            row: Pandas Series containing paper metadata.
            similarity_score: Cosine similarity score.
            rank: Rank position integer.

        Returns:
            Formatted explanation string describing the retrieval match.
        """
        domain = row.get("broad_domain", "Scientific Field")
        year = row.get("publication_year", "N/A")
        title = row.get("title", "Research Paper")

        if similarity_score >= 0.75:
            match_strength = "high high-dimensional semantic overlap"
        elif similarity_score >= 0.55:
            match_strength = "moderate topical and contextual alignment"
        else:
            match_strength = "broad conceptual relevance"

        reason = (
            f"Rank #{rank} match retrieved via {match_strength} (Cosine Sim: {similarity_score:.4f}). "
            f"Matches query concepts for '{query[:40]}...' within domain '{domain}' ({year})."
        )
        return reason

    def rank_results(
        self,
        indices: List[int],
        distances: List[float],
        query: str,
        sort_by: str = "similarity_score",
        applied_filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Ranks retrieved candidates and formats rich, explainable metadata DataFrame.

        Args:
            indices: Filtered candidate row indices.
            distances: Corresponding similarity scores.
            query: Original user search query.
            sort_by: Attribute to sort results by ('similarity_score', 'newest', 'oldest', 'title').
            applied_filters: Dictionary logging applied filter criteria.

        Returns:
            Structured pandas DataFrame matching target search output schema.
        """
        records = []
        filter_summary_str = json.dumps(applied_filters) if applied_filters else "None"

        for idx, dist in zip(indices, distances):
            row = self.metadata_df.iloc[idx]
            paper_id = str(self.paper_ids[idx])
            sim_score = float(dist)
            conf_score = self._compute_confidence_score(sim_score)

            scientific_text, content_source = self._extract_content(row)
            content_words = len(scientific_text.split()) if scientific_text != "N/A" else 0

            records.append({
                "Paper ID": paper_id,
                "Similarity Score": round(sim_score, 4),
                "Confidence Score (%)": conf_score,
                "Title": str(row.get("title", "N/A")),
                "Authors": str(row.get("authors", "N/A")),
                "Category": str(row.get("category", "N/A")),
                "Broad Domain": str(row.get("broad_domain", "N/A")),
                "Publication Year": int(row.get("publication_year", 0)),
                # Backward-compatible field consumed by RetrieverPipeline.
                # It now contains the real scientific content from `combined_text`.
                "Abstract": scientific_text,
                "Content": scientific_text,
                "Content Source": content_source,
                "Content Word Count": content_words,
                "Matched Query": query,
                "Applied Filters": filter_summary_str,
                "retrieved_index": idx,
            })

        results_df = pd.DataFrame(records)
        if results_df.empty:
            return pd.DataFrame(columns=[
                "Rank", "Similarity Score", "Confidence Score (%)", "Paper ID", 
                "Title", "Authors", "Category", "Broad Domain", 
                "Publication Year", "Abstract", "Content", "Content Source",
                "Content Word Count", "Matched Query", "Applied Filters", "Explanation"
            ])

        # Sorting Layer
        if sort_by == "newest":
            results_df = results_df.sort_values(by=["Publication Year", "Similarity Score"], ascending=[False, False])
        elif sort_by == "oldest":
            results_df = results_df.sort_values(by=["Publication Year", "Similarity Score"], ascending=[True, False])
        elif sort_by == "title":
            results_df = results_df.sort_values(by="Title", ascending=True)
        else:  # Default: similarity_score
            results_df = results_df.sort_values(by="Similarity Score", ascending=False)

        results_df = results_df.reset_index(drop=True)
        results_df["Rank"] = range(1, len(results_df) + 1)

        # Generate Explainable Retrieval Column
        explanations = []
        for idx_pos, row_item in results_df.iterrows():
            exp = self._generate_explainable_reason(
                query=query,
                row=row_item,
                similarity_score=row_item["Similarity Score"],
                rank=row_item["Rank"],
            )
            explanations.append(exp)

        results_df["Explanation"] = explanations

        # Re-order output columns to strictly match requested specification
        output_cols = [
            "Rank",
            "Similarity Score",
            "Confidence Score (%)",
            "Paper ID",
            "Title",
            "Authors",
            "Category",
            "Broad Domain",
            "Publication Year",
            "Abstract",
            "Content",
            "Content Source",
            "Content Word Count",
            "Matched Query",
            "Applied Filters",
            "Explanation",
        ]
        return results_df[output_cols]

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        year_range: Optional[Tuple[int, int]] = None,
        category: Optional[str] = None,
        broad_domain: Optional[str] = None,
        author: Optional[str] = None,
        is_single_author: Optional[bool] = None,
        has_multiple_authors: Optional[bool] = None,
        sort_by: str = "similarity_score",
    ) -> pd.DataFrame:
        """Executes full end-to-end semantic search pipeline.

        Args:
            query: Natural language research query.
            top_k: Number of top nearest paper results to retrieve.
            year_range: Optional tuple of (start_year, end_year).
            category: Optional specific category code filter.
            broad_domain: Optional high-level scientific field filter.
            author: Optional author name filter.
            is_single_author: Optional solo-author flag.
            has_multiple_authors: Optional co-author flag.
            sort_by: Ranking attribute ('similarity_score', 'newest', 'oldest', 'title').

        Returns:
            DataFrame containing ranked research paper search results.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be a non-empty string.")
        if not isinstance(top_k, int) or top_k < 1 or top_k > MAX_TOP_K:
            raise ValueError(f"top_k must be an integer between 1 and {MAX_TOP_K}.")

        start_total = time.time()
        applied_filters = {}

        if year_range:
            applied_filters["year_range"] = year_range
        if category:
            applied_filters["category"] = category
        if broad_domain:
            applied_filters["broad_domain"] = broad_domain
        if author:
            applied_filters["author"] = author
        if is_single_author is not None:
            applied_filters["is_single_author"] = is_single_author
        if has_multiple_authors is not None:
            applied_filters["has_multiple_authors"] = has_multiple_authors

        logger.info(f"Executing search query: '{query}' (top_k={top_k}, filters={applied_filters})...")

        # Step 1: Generate Query Embedding
        query_vec, embed_time = self.embed_query(query)

        # Step 2-3: ANN retrieval + adaptive metadata filtering.
        # A fixed 10x over-fetch can silently return too few papers for selective
        # filters. We progressively expand the candidate pool until top_k is
        # satisfied or the configured safety limit is reached.
        has_filters = len(applied_filters) > 0
        raw_k = min(
            max(top_k * DEFAULT_FILTER_FETCH_MULTIPLIER, top_k),
            self.index.ntotal,
        ) if has_filters else min(top_k, self.index.ntotal)

        all_filtered_indices: List[int] = []
        all_filtered_distances: List[float] = []
        total_faiss_time = 0.0
        total_filter_time = 0.0

        while True:
            t_search_0 = time.time()
            distances, indices = self.index.search(query_vec, k=raw_k)
            total_faiss_time += time.time() - t_search_0

            t_filter_0 = time.time()
            filtered_indices, filtered_distances = self.apply_filters(
                candidate_indices=indices[0],
                distances=distances[0],
                year_range=year_range,
                category=category,
                broad_domain=broad_domain,
                author=author,
                is_single_author=is_single_author,
                has_multiple_authors=has_multiple_authors,
            )
            total_filter_time += time.time() - t_filter_0

            all_filtered_indices = filtered_indices
            all_filtered_distances = filtered_distances

            enough = len(all_filtered_indices) >= top_k
            exhausted = raw_k >= self.index.ntotal
            safety_limit = min(self.index.ntotal, MAX_FILTER_FETCH)
            reached_safety = raw_k >= safety_limit

            if not has_filters or enough or exhausted or reached_safety:
                break

            raw_k = min(raw_k * 2, self.index.ntotal, safety_limit)

        faiss_time = total_faiss_time
        filter_time = total_filter_time

        # Truncate to desired top_k post-filtering
        final_indices = all_filtered_indices[:top_k]
        final_distances = all_filtered_distances[:top_k]

        # Step 4: Rank and Format Results
        t_rank_0 = time.time()
        results_df = self.rank_results(
            indices=final_indices,
            distances=final_distances,
            query=query,
            sort_by=sort_by,
            applied_filters=applied_filters,
        )
        rank_time = time.time() - t_rank_0

        total_latency_ms = (time.time() - start_total) * 1000.0
        logger.info(
            f"Search completed in {total_latency_ms:.2f} ms "
            f"(Embed: {embed_time*1000:.1f}ms, FAISS: {faiss_time*1000:.1f}ms, "
            f"Filter: {filter_time*1000:.1f}ms, Rank: {rank_time*1000:.1f}ms)."
        )

        # Evidence integrity diagnostics for downstream RAG.
        if not results_df.empty:
            content_available = int(
                (results_df["Content Word Count"] >= MIN_CONTENT_WORDS).sum()
            )
            logger.info(
                f"Evidence content available: {content_available}/{len(results_df)} "
                f"results meet minimum content threshold ({MIN_CONTENT_WORDS} words)."
            )
        else:
            logger.warning("Search returned zero results after filtering.")

        # Log session metadata
        self._log_session(query, len(results_df), total_latency_ms, applied_filters)

        return results_df

    def _log_session(
        self, query: str, num_results: int, latency_ms: float, filters: Dict[str, Any]
    ) -> None:
        """Persists structured search query execution logs to disk."""
        log_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "num_results": num_results,
            "latency_ms": round(latency_ms, 2),
            "filters": filters,
        }
        log_path = os.path.join(LOGS_DIR, "search_session_history.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")


# =============================================================================
# Benchmarking & Analytics Engine
# =============================================================================
def benchmark_search_engine(
    engine: SemanticSearchEngine,
    benchmark_queries: List[str],
    top_k_list: List[int] = [5, 10, 20, 50],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs latency, memory, and throughput benchmarks across benchmark queries.

    Args:
        engine: Initialized SemanticSearchEngine instance.
        benchmark_queries: List of representative search prompts.
        top_k_list: List of top_k values to evaluate.

    Returns:
        Tuple of (benchmark_summary_df, latency_metrics_dict).
    """
    logger.info("Executing Search Engine Performance Benchmark Suite...")
    benchmark_records = []
    latencies_all = []

    for k in top_k_list:
        k_latencies_ms = []
        for q in benchmark_queries:
            t0 = time.time()
            _ = engine.search(query=q, top_k=k)
            t_elapsed = (time.time() - t0) * 1000.0
            k_latencies_ms.append(t_elapsed)
            latencies_all.append(t_elapsed)

        qps = len(benchmark_queries) / (sum(k_latencies_ms) / 1000.0) if sum(k_latencies_ms) > 0 else 0.0

        benchmark_records.append({
            "top_k": k,
            "queries_evaluated": len(benchmark_queries),
            "avg_latency_ms": round(float(np.mean(k_latencies_ms)), 2),
            "median_latency_ms": round(float(np.median(k_latencies_ms)), 2),
            "p95_latency_ms": round(float(np.percentile(k_latencies_ms, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(k_latencies_ms, 99)), 2),
            "queries_per_second": round(qps, 2),
        })

    bench_df = pd.DataFrame(benchmark_records)
    bench_df.to_csv(os.path.join(REPORTS_DIR, "benchmark_report.csv"), index=False)

    process_mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    metrics_summary = {
        "avg_latency_ms": round(float(np.mean(latencies_all)), 2),
        "overall_qps": round(len(latencies_all) / (sum(latencies_all) / 1000.0), 2),
        "ram_usage_mb": round(process_mem_mb, 2),
    }

    return bench_df, metrics_summary


# =============================================================================
# Automated Quality Validation Engine
# =============================================================================
def validate_search_engine(engine: SemanticSearchEngine) -> pd.DataFrame:
    """Runs automated validation tests: self-similarity, exact title match, and duplicate check.

    Args:
        engine: Initialized SemanticSearchEngine instance.

    Returns:
        DataFrame containing test validation metrics and pass/fail statuses.
    """
    logger.info("Executing Automated Search Engine Validation Suite...")
    val_records = []

    # Test 1: Self-Similarity & Title Query Match
    sample_paper = engine.metadata_df.iloc[10]
    sample_title = sample_paper["title"]
    expected_id = str(engine.paper_ids[10])

    res_df = engine.search(query=sample_title, top_k=5)
    top_1_id = res_df.iloc[0]["Paper ID"] if not res_df.empty else "NONE"
    is_self_similar = (expected_id == top_1_id)

    val_records.append({
        "validation_test": "self_similarity_title_retrieval",
        "expected_value": expected_id,
        "actual_value": top_1_id,
        "status": "PASS" if is_self_similar else "CHECK",
    })

    # Test 2: Duplicate Detection in Results
    res_multi = engine.search(query="deep neural networks for computer vision", top_k=20)
    has_duplicates = res_multi["Paper ID"].duplicated().any() if not res_multi.empty else False

    val_records.append({
        "validation_test": "duplicate_result_detection",
        "expected_value": "No Duplicates",
        "actual_value": "Duplicates Found" if has_duplicates else "No Duplicates",
        "status": "PASS" if not has_duplicates else "FAIL",
    })

    # Test 3: Scientific Content Availability
    res_content = engine.search(query="vision transformers autonomous perception", top_k=5)
    content_ok = (
        not res_content.empty
        and "Content" in res_content.columns
        and int((res_content["Content Word Count"] >= MIN_CONTENT_WORDS).sum()) > 0
    )
    val_records.append({
        "validation_test": "scientific_content_availability",
        "expected_value": f">=1 result with >= {MIN_CONTENT_WORDS} words",
        "actual_value": (
            f"{int((res_content['Content Word Count'] >= MIN_CONTENT_WORDS).sum())} qualifying results"
            if not res_content.empty else "0 results"
        ),
        "status": "PASS" if content_ok else "FAIL",
    })

    # Test 4: Metadata Alignment Verification
    res_align = engine.search(query="large language models RAG", top_k=5)
    aligned_all = True
    for idx_pos, r in res_align.iterrows():
        p_id = r["Paper ID"]
        matched_meta = engine.metadata_df[engine.metadata_df["id"] == p_id]
        if matched_meta.empty:
            aligned_all = False
            break

    val_records.append({
        "validation_test": "metadata_id_alignment",
        "expected_value": "100% Aligned",
        "actual_value": "100% Aligned" if aligned_all else "Alignment Mismatch",
        "status": "PASS" if aligned_all else "FAIL",
    })

    val_df = pd.DataFrame(val_records)
    val_df.to_csv(os.path.join(REPORTS_DIR, "validation_report.csv"), index=False)
    return val_df


# =============================================================================
# Diagnostic Visualization Generator
# =============================================================================
def generate_search_visualizations(
    bench_df: pd.DataFrame, sample_results_df: pd.DataFrame
) -> None:
    """Generates visual charts for search latency, throughput, similarity distribution, and categories.

    Args:
        bench_df: Benchmark metrics DataFrame.
        sample_results_df: Sample retrieval output DataFrame.
    """
    logger.info("Generating diagnostic search analytics and performance plots...")
    plt.style.use("ggplot")

    # Plot 1: Latency Distribution vs Top-K
    plt.figure(figsize=(8, 5))
    plt.plot(bench_df["top_k"], bench_df["avg_latency_ms"], marker="o", color="#3498db", label="Avg Latency (ms)")
    plt.plot(bench_df["top_k"], bench_df["p95_latency_ms"], marker="s", linestyle="--", color="#e74c3c", label="P95 Latency (ms)")
    plt.title("Search Latency Distribution vs. Top-K")
    plt.xlabel("Top-K Value")
    plt.ylabel("Latency (milliseconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency_distribution.png"), dpi=300)
    plt.close()

    # Plot 2: Query Throughput (QPS)
    plt.figure(figsize=(8, 5))
    plt.bar(bench_df["top_k"].astype(str), bench_df["queries_per_second"], color="#2ecc71", edgecolor="black")
    plt.title("Query Throughput (Queries Per Second)")
    plt.xlabel("Top-K Parameter")
    plt.ylabel("QPS")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "query_throughput.png"), dpi=300)
    plt.close()

    if not sample_results_df.empty:
        # Plot 3: Similarity Score Distribution
        plt.figure(figsize=(8, 5))
        plt.hist(sample_results_df["Similarity Score"], bins=15, color="#9b59b6", edgecolor="black")
        plt.title("Similarity Score Distribution (Sample Query)")
        plt.xlabel("Cosine Similarity Score")
        plt.ylabel("Retrieved Count")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "similarity_score_distribution.png"), dpi=300)
        plt.close()

        # Plot 4: Retrieved Categories
        plt.figure(figsize=(8, 5))
        domain_counts = sample_results_df["Broad Domain"].value_counts()
        plt.barh(domain_counts.index, domain_counts.values, color="#f39c12", edgecolor="black")
        plt.title("Retrieved Scientific Domains Distribution")
        plt.xlabel("Paper Count")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "retrieved_domains.png"), dpi=300)
        plt.close()


# =============================================================================
# Export Utilities
# =============================================================================
def export_search_results(df: pd.DataFrame, file_prefix: str = "sample_search") -> None:
    """Exports search result DataFrames to CSV and formatted JSON formats.

    Args:
        df: Search results DataFrame.
        file_prefix: Output filename prefix.
    """
    csv_path = os.path.join(REPORTS_DIR, f"{file_prefix}_results.csv")
    json_path = os.path.join(REPORTS_DIR, f"{file_prefix}_results.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    logger.info(f"Exported search results to '{csv_path}' and '{json_path}'.")


# =============================================================================
# Main Execution Entry Point
# =============================================================================
def main() -> None:
    """Main function executing the Semantic Search Engine pipeline."""
    setup_directories()
    logger.info("Initializing Production Semantic Search Engine...")

    # Initialize Engine
    engine = SemanticSearchEngine()

    # Representative Test Query
    test_query = "Attention mechanisms for transformer architecture in computer vision"
    logger.info(f"\nExecuting Sample Search Query: '{test_query}'")

    results_df = engine.search(
        query=test_query,
        top_k=10,
        broad_domain="Computer Vision",
        sort_by="similarity_score",
    )

    export_search_results(results_df, file_prefix="computer_vision_query")

    # Run Automated Engine Validation
    val_df = validate_search_engine(engine)

    # Benchmark Performance Across Multiple Queries
    benchmark_queries = [
        "Deep reinforcement learning for autonomous driving",
        "Large language models for medical diagnosis and clinical text",
        "Graph neural networks for drug discovery and molecular property prediction",
        "Diffusion models for image generation and text-to-image synthesis",
        "Retrieval-augmented generation for scientific paper question answering",
    ]

    bench_df, metrics_summary = benchmark_search_engine(engine, benchmark_queries)
    generate_search_visualizations(bench_df, results_df)

    logger.info("\n========== SEMANTIC SEARCH ENGINE READY ==========")
    logger.info(f"Sample Query Yielded  : {len(results_df)} Top Papers")
    logger.info(f"Average Search Latency: {metrics_summary['avg_latency_ms']} ms")
    logger.info(f"Query Throughput      : {metrics_summary['overall_qps']} QPS")
    validation_passed = bool((val_df["status"] == "PASS").all())
    logger.info(
        f"Validation Suite      : {'ALL CHECKS PASSED' if validation_passed else 'CHECKS REQUIRE ATTENTION'}"
    )
    logger.info(f"Reports Saved To      : {REPORTS_DIR}")
    logger.info("==================================================\n")


if __name__ == "__main__":
    main()