"""
Production FAISS Vector Database Module for AI Research Paper Assistant.

This module loads dense paper embeddings and associated metadata, performs quality and 
alignment validation, normalizes vectors for Cosine Similarity, constructs a configurable FAISS 
index, validates retrieval self-consistency, benchmarks query latency, generates performance reports,
and persists all database artifacts.

Index Justification:
--------------------
For a dataset size of 287,419 vectors of dimension 384 (total uncompressed memory footprint ~441 MB),
an in-memory IndexFlatIP (Flat Inner Product on L2-normalized vectors) or IndexHNSWFlat provides 
an optimal trade-off:
1. IndexFlatIP offers 100% exact recall (zero approximation error) with query latencies < 1-2 ms on modern CPUs.
2. It avoids the training overhead and recall drop of IVF or PQ quantization methods.
3. Memory consumption remains modest (~450 MB RAM), making exact search the most robust choice for RAG pipelines.

Location: src/models/faiss_vector_index.py
"""

import logging
import os
import pathlib
import pickle
import time
import warnings
from typing import Any, Dict, List, Tuple, Optional

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")

# -----------------------------------------------------------------------------
# Configuration Section
# -----------------------------------------------------------------------------
# Index configuration: 'FLAT' (IndexFlatIP), 'IVF' (IndexIVFFlat), 'HNSW' (IndexHNSWFlat)
INDEX_TYPE = "FLAT"
EXPECTED_DIMENSION = 384
DEFAULT_TOP_K = 10

INPUT_EMBEDDINGS_DIR = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "dataset/embeddings")
EMBEDDINGS_FILE = os.path.join(INPUT_EMBEDDINGS_DIR, "paper_embeddings.npy")
METADATA_FILE = os.path.join(INPUT_EMBEDDINGS_DIR, "paper_metadata.parquet")
IDS_FILE = os.path.join(INPUT_EMBEDDINGS_DIR, "paper_ids.npy")

VECTOR_DB_DIR = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "dataset/vector_database")
REPORTS_DIR = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "outputs/reports/faiss")
PLOTS_DIR = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "outputs/reports/faiss/plots")

INDEX_FILE_PATH = os.path.join(VECTOR_DB_DIR, "faiss_index.bin")
CONFIG_METADATA_PATH = os.path.join(VECTOR_DB_DIR, "index_metadata.pkl")


# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Configures console and file logging handlers."""
    logger = logging.getLogger("FAISSVectorDB")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        pathlib.Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(os.path.join(REPORTS_DIR, "faiss_pipeline.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger


logger = setup_logging()


# -----------------------------------------------------------------------------
# System Setup & Data Loaders
# -----------------------------------------------------------------------------
def setup_directories() -> None:
    """Creates directory paths for database storage, reports, and visualization charts."""
    pathlib.Path(VECTOR_DB_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)


def load_input_artifacts() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Loads embeddings binary, paper IDs array, and metadata parquet table.

    Returns:
        Tuple containing (embeddings_array, ids_array, metadata_df).
    
    Raises:
        FileNotFoundError: If any expected input artifact is missing.
    """
    for file_path in [EMBEDDINGS_FILE, IDS_FILE, METADATA_FILE]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required artifact not found: {file_path}")

    logger.info(f"Loading dense embeddings from '{EMBEDDINGS_FILE}'...")
    embeddings = np.load(EMBEDDINGS_FILE)

    logger.info(f"Loading paper IDs from '{IDS_FILE}'...")
    paper_ids = np.load(IDS_FILE, allow_pickle=True)

    logger.info(f"Loading paper metadata from '{METADATA_FILE}'...")
    metadata_df = pd.read_parquet(METADATA_FILE)

    return embeddings, paper_ids, metadata_df


# -----------------------------------------------------------------------------
# Preprocessing and Validation
# -----------------------------------------------------------------------------
def validate_inputs(embeddings: np.ndarray, paper_ids: np.ndarray, df: pd.DataFrame) -> Tuple[np.ndarray, bool]:
    """
    Validates dimensional conformity, NaN/Inf presence, row-count parity, and ID alignment.

    Args:
        embeddings: Dense embedding matrix.
        paper_ids: Array of unique paper IDs.
        df: Metadata Parquet DataFrame.

    Returns:
        Tuple of (L2-normalized embeddings matrix, validation_pass_boolean).
    """
    logger.info("Executing pre-indexing validation and integrity checks...")
    num_embeddings, dim = embeddings.shape
    num_ids = len(paper_ids)
    num_meta = len(df)

    if dim != EXPECTED_DIMENSION:
        raise ValueError(f"Embedding dimension mismatch: expected {EXPECTED_DIMENSION}, got {dim}.")

    if not (num_embeddings == num_ids == num_meta):
        raise ValueError(
            f"Row count mismatch across artifacts: Embeddings ({num_embeddings}), "
            f"IDs ({num_ids}), Metadata ({num_meta})."
        )

    nan_count = int(np.isnan(embeddings).sum())
    inf_count = int(np.isinf(embeddings).sum())

    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"Corrupted embeddings detected: NaNs={nan_count}, Infs={inf_count}.")

    dup_ids = int(pd.Series(paper_ids).duplicated().sum())
    if dup_ids > 0:
        logger.warning(f"Detected {dup_ids} duplicate paper IDs in input dataset.")

    # Ensure L2 normalization for Cosine Similarity via Inner Product Index
    logger.info("Normalizing embeddings to unit L2 norm for Cosine Similarity...")
    normalized_embeddings = embeddings.copy().astype("float32")
    faiss.normalize_L2(normalized_embeddings)

    norms = np.linalg.norm(normalized_embeddings, axis=1)
    invalid_norms = int(np.sum((norms < 0.99) | (norms > 1.01)))

    if invalid_norms > 0:
        logger.error(f"L2 normalization failed for {invalid_norms} vectors.")
        return normalized_embeddings, False

    logger.info(f"Validation completed successfully for {num_embeddings:,} vectors.")
    return normalized_embeddings, True


# -----------------------------------------------------------------------------
# Vector Index Factory & Construction
# -----------------------------------------------------------------------------
def build_faiss_index(embeddings: np.ndarray, index_type: str = "FLAT") -> Tuple[faiss.Index, Dict[str, Any]]:
    """
    Constructs and populates a FAISS vector index based on configured strategy.

    Args:
        embeddings: L2-normalized float32 numpy matrix.
        index_type: Strategy key ('FLAT', 'IVF', 'HNSW').

    Returns:
        Tuple of (populated FAISS index instance, construction_stats_dict).
    """
    num_vectors, dim = embeddings.shape
    logger.info(f"Building FAISS vector index strategy: '{index_type}' (Vectors: {num_vectors:,}, Dim: {dim})...")

    start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    start_time = time.time()

    if index_type.upper() == "FLAT":
        index = faiss.IndexFlatIP(dim)
    elif index_type.upper() == "IVF":
        nlist = int(4 * np.sqrt(num_vectors))  # Rule of thumb for cluster centroids
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        logger.info(f"Training IVF index with {nlist} clusters...")
        index.train(embeddings)
        index.nprobe = min(16, nlist)
    elif index_type.upper() == "HNSW":
        M = 32  # Number of connections per node
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = 64
        index.hnsw.efConstruction = 64
    else:
        raise ValueError(f"Unsupported FAISS index strategy: {index_type}")

    # Add vectors to index
    logger.info("Adding normalized vectors to index...")
    index.add(embeddings)

    build_time = time.time() - start_time
    end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    mem_used_mb = max(0.0, end_mem - start_mem)

    stats = {
        "index_type": index_type.upper(),
        "total_vectors": index.ntotal,
        "dimension": dim,
        "is_trained": index.is_trained,
        "build_time_seconds": round(build_time, 4),
        "build_memory_mb": round(mem_used_mb, 2)
    }

    logger.info(f"FAISS index built successfully in {build_time:.2f} seconds.")
    return index, stats


# -----------------------------------------------------------------------------
# Automated Search Validation & Retrieval Consistency
# -----------------------------------------------------------------------------
def validate_retrieval_consistency(
    index: faiss.Index,
    embeddings: np.ndarray,
    paper_ids: np.ndarray,
    df: pd.DataFrame,
    sample_size: int = 100
) -> pd.DataFrame:
    """
    Executes top-1 self-consistency search and metadata alignment checks on random samples.

    Args:
        index: Populated FAISS index.
        embeddings: Query embedding matrix.
        paper_ids: Array of paper IDs.
        df: Metadata DataFrame.
        sample_size: Number of random query vectors to validate.

    Returns:
        DataFrame containing sample search validation records.
    """
    logger.info(f"Executing automated retrieval validation on {sample_size} random queries...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(embeddings), size=min(sample_size, len(embeddings)), replace=False)

    results = []
    top_1_matches = 0

    for idx in sample_indices:
        query_vec = embeddings[idx:idx + 1]
        expected_id = str(paper_ids[idx])

        distances, indices = index.search(query_vec, k=DEFAULT_TOP_K)
        retrieved_idx = indices[0][0]
        retrieved_id = str(paper_ids[retrieved_idx]) if retrieved_idx >= 0 else "NONE"
        top_dist = float(distances[0][0])

        is_match = (expected_id == retrieved_id)
        if is_match:
            top_1_matches += 1

        meta_match = False
        if retrieved_idx >= 0:
            meta_id = str(df.iloc[retrieved_idx]["id"])
            meta_match = (retrieved_id == meta_id)

        results.append({
            "query_index": int(idx),
            "expected_id": expected_id,
            "retrieved_top1_id": retrieved_id,
            "top1_similarity": round(top_dist, 4),
            "self_similarity_pass": is_match,
            "metadata_alignment_pass": meta_match
        })

    accuracy_pct = (top_1_matches / len(sample_indices)) * 100
    logger.info(f"Retrieval Self-Consistency Validation Accuracy: {accuracy_pct:.2f}% ({top_1_matches}/{len(sample_indices)}).")

    val_df = pd.DataFrame(results)
    val_df.to_csv(os.path.join(REPORTS_DIR, "search_validation.csv"), index=False)
    return val_df


# -----------------------------------------------------------------------------
# Latency & Throughput Benchmarking
# -----------------------------------------------------------------------------
def benchmark_search_performance(
    index: faiss.Index,
    embeddings: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20],
    num_queries: int = 1000
) -> pd.DataFrame:
    """
    Measures search latency distributions and query throughput across varying top-k parameters.

    Args:
        index: Populated FAISS index.
        embeddings: Vector source matrix for query sampling.
        k_values: List of K nearest neighbors to evaluate.
        num_queries: Total random search queries to benchmark.

    Returns:
        DataFrame containing benchmark latency metrics.
    """
    logger.info(f"Running latency and throughput benchmarks across {num_queries:,} queries...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(embeddings), size=min(num_queries, len(embeddings)), replace=False)
    query_vectors = embeddings[sample_indices]

    benchmarks = []

    for k in k_values:
        latencies_ms = []
        
        # Warmup query
        index.search(query_vectors[:5], k=k)

        start_time = time.time()
        for i in range(len(query_vectors)):
            q = query_vectors[i:i + 1]
            t0 = time.time()
            index.search(q, k=k)
            t1 = time.time()
            latencies_ms.append((t1 - t0) * 1000.0)

        total_wall_time = time.time() - start_time
        qps = round(len(query_vectors) / total_wall_time, 2) if total_wall_time > 0 else 0.0

        benchmarks.append({
            "top_k": k,
            "queries_evaluated": len(query_vectors),
            "avg_latency_ms": round(float(np.mean(latencies_ms)), 4),
            "median_latency_ms": round(float(np.median(latencies_ms)), 4),
            "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 4),
            "p99_latency_ms": round(float(np.percentile(latencies_ms, 99)), 4),
            "queries_per_second": qps
        })

    bench_df = pd.DataFrame(benchmarks)
    bench_df.to_csv(os.path.join(REPORTS_DIR, "performance_report.csv"), index=False)
    return bench_df


# -----------------------------------------------------------------------------
# Visualization & Report Export
# -----------------------------------------------------------------------------
def generate_reports_and_visualizations(
    index_stats: Dict[str, Any],
    bench_df: pd.DataFrame,
    index_file_path: str
) -> None:
    """Generates visual charts and exports summary reports."""
    logger.info("Generating system performance and diagnostic plots...")
    
    file_size_mb = os.path.getsize(index_file_path) / (1024 * 1024) if os.path.exists(index_file_path) else 0.0

    # 1. Index Statistics CSV
    stats_combined = {**index_stats, "index_file_size_mb": round(file_size_mb, 2)}
    pd.DataFrame([stats_combined]).to_csv(os.path.join(REPORTS_DIR, "index_statistics.csv"), index=False)

    # 2. Plot: Latency vs. Top-K
    plt.figure(figsize=(8, 5))
    plt.plot(bench_df["top_k"], bench_df["avg_latency_ms"], marker="o", color="#e74c3c", label="Avg Latency")
    plt.plot(bench_df["top_k"], bench_df["p95_latency_ms"], marker="s", linestyle="--", color="#3498db", label="P95 Latency")
    plt.title("Search Latency vs. Top-K Parameter")
    plt.xlabel("Top-K")
    plt.ylabel("Latency (milliseconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "search_latency_distribution.png"), dpi=300)
    plt.close()

    # 3. Plot: Query Throughput (QPS)
    plt.figure(figsize=(8, 5))
    plt.bar(bench_df["top_k"].astype(str), bench_df["queries_per_second"], color="#2ecc71", edgecolor="black")
    plt.title("Query Throughput (Queries / Second)")
    plt.xlabel("Top-K")
    plt.ylabel("QPS")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "query_throughput.png"), dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Persistence & Reload Validation
# -----------------------------------------------------------------------------
def save_vector_database(
    index: faiss.Index,
    paper_ids: np.ndarray,
    metadata_df: pd.DataFrame,
    index_stats: Dict[str, Any]
) -> None:
    """Serializes binary FAISS index and metadata configurations to disk."""
    logger.info(f"Saving binary FAISS index to: '{INDEX_FILE_PATH}'...")
    faiss.write_index(index, INDEX_FILE_PATH)

    config_data = {
        "index_type": INDEX_TYPE,
        "dimension": EXPECTED_DIMENSION,
        "total_vectors": index.ntotal,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paper_ids": paper_ids,
        "metadata_schema": list(metadata_df.columns)
    }

    logger.info(f"Saving index metadata mapping to: '{CONFIG_METADATA_PATH}'...")
    with open(CONFIG_METADATA_PATH, "wb") as f:
        pickle.dump(config_data, f)


def reload_and_verify_database() -> bool:
    """Verifies index serialization integrity by reloading index from disk."""
    logger.info("Executing reload verification on saved FAISS binary index...")
    if not os.path.exists(INDEX_FILE_PATH):
        logger.error("Index binary file missing post-save.")
        return False

    reloaded_index = faiss.read_index(INDEX_FILE_PATH)
    if reloaded_index.ntotal == 0 or reloaded_index.d != EXPECTED_DIMENSION:
        logger.error("Reloaded index state is corrupted.")
        return False

    logger.info(f"Reload verification passed. Total vectors in reloaded index: {reloaded_index.ntotal:,}.")
    return True


# -----------------------------------------------------------------------------
# Main Execution Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    """Executes the complete FAISS vector database pipeline."""
    setup_directories()
    logger.info("Starting FAISS Vector Database Construction Pipeline...")

    try:
        embeddings, paper_ids, metadata_df = load_input_artifacts()
    except Exception as e:
        logger.critical(f"Failed to load required input artifacts: {e}")
        return

    normalized_embeddings, valid = validate_inputs(embeddings, paper_ids, metadata_df)
    if not valid:
        logger.critical("Embedding validation failed. Aborting index construction.")
        return

    index, index_stats = build_faiss_index(normalized_embeddings, index_type=INDEX_TYPE)

    _ = validate_retrieval_consistency(index, normalized_embeddings, paper_ids, metadata_df, sample_size=100)
    bench_df = benchmark_search_performance(index, normalized_embeddings, k_values=[1, 5, 10, 20], num_queries=500)

    save_vector_database(index, paper_ids, metadata_df, index_stats)
    generate_reports_and_visualizations(index_stats, bench_df, INDEX_FILE_PATH)

    reload_success = reload_and_verify_database()

    logger.info("\n========== FAISS VECTOR DATABASE READY ==========")
    logger.info(f"Total Vectors Indexed : {index.ntotal:,}")
    logger.info(f"Vector Dimensions     : {index.d}")
    logger.info(f"Index Strategy        : {INDEX_TYPE}")
    logger.info(f"Reload Verification   : {'PASSED' if reload_success else 'FAILED'}")
    logger.info(f"FAISS Index Path      : {INDEX_FILE_PATH}")
    logger.info(f"Metadata Config Path  : {CONFIG_METADATA_PATH}")
    logger.info("=================================================\n")


if __name__ == "__main__":
    main()