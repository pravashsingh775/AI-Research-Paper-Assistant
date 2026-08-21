"""
Production Dense Embedding Generation Pipeline for AI Research Paper Assistant.

This module loads preprocessed scientific paper records, formats text documents, 
initializes sentence transformer models, generates L2-normalized dense embeddings in batches,
and persists embeddings, metadata, and validation reports.

Location: src/models/embedding_generation.py
"""

import logging
import os
import pathlib
import time
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

# Force matplotlib to use 'Agg' backend for headless plot generation
plt.switch_backend("Agg")

# -----------------------------------------------------------------------------
# Configuration Section
# -----------------------------------------------------------------------------
MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Swappable model identifier
DEFAULT_BATCH_SIZE = 256
MAX_SEQ_LENGTH = 512

INPUT_FILE_PATH = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "dataset/processed/arXiv_feature_engineered_dataset.csv")
EMBEDDINGS_DIR = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "dataset/embeddings")
REPORTS_DIR = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "outputs/reports/embedding_generation")
PLOTS_DIR = rstr(pathlib.Path(__file__).resolve().parent.parent.parent / "outputs/reports/embedding_generation/plots")

EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "paper_embeddings.npy")
METADATA_FILE = os.path.join(EMBEDDINGS_DIR, "paper_metadata.parquet")
IDS_FILE = os.path.join(EMBEDDINGS_DIR, "paper_ids.npy")
CHECKPOINT_FILE = os.path.join(EMBEDDINGS_DIR, "checkpoint_state.npz")

METADATA_COLUMNS = [
    "id",
    "title",
    "authors",
    "category",
    "broad_domain",
    "publication_year",
    "retrieval_weight",
    "embedding_priority",
    "combined_text"
]


# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Configures structured console and file logging."""
    logger = logging.getLogger("EmbeddingGenerator")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        pathlib.Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(os.path.join(REPORTS_DIR, "embedding_pipeline.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger


logger = setup_logging()


# -----------------------------------------------------------------------------
# System Setup & Data Loaders
# -----------------------------------------------------------------------------
def setup_directories() -> None:
    """Creates directory trees for embeddings, reports, and visualization plots."""
    pathlib.Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Detects available hardware acceleration device (CUDA/MPS/CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Hardware Acceleration: CUDA GPU detected ({torch.cuda.get_device_name(0)}).")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Hardware Acceleration: Apple Metal (MPS) detected.")
    else:
        device = torch.device("cpu")
        logger.info("Hardware Acceleration: CPU execution mode.")
    return device


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads preprocessed dataset with encoding fallback and column validation."""
    path = pathlib.Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input feature engineered dataset not found: {file_path}")

    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            if df.empty:
                raise ValueError("Dataset file is empty.")
            logger.info(f"Dataset successfully loaded from '{file_path}' ({len(df):,} rows).")
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    raise ValueError(f"Failed to load dataset at '{file_path}' with supported encodings.")


def format_embedding_text(df: pd.DataFrame) -> List[str]:
    """Formats title and abstract text into structured document strings for embedding."""
    titles = df["title"].fillna("").astype(str)
    summaries = df["summary"].fillna("").astype(str)
    
    formatted_docs = [f"{t}\n\n{s}" for t, s in zip(titles, summaries)]
    return formatted_docs


# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
def initialize_model(model_name: str, device: torch.device) -> SentenceTransformer:
    """Initializes and configures sentence transformer embedding model."""
    logger.info(f"Initializing embedding model: '{model_name}'...")
    start_t = time.time()
    
    model = SentenceTransformer(model_name, device=str(device))
    model.max_seq_length = MAX_SEQ_LENGTH
    
    init_time = time.time() - start_t
    logger.info(f"Model initialized in {init_time:.2f} seconds.")
    return model


# -----------------------------------------------------------------------------
# Embedding Generation Engine
# -----------------------------------------------------------------------------
def generate_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generates L2-normalized dense embeddings in batches with resume capability."""
    total_docs = len(texts)
    start_idx = 0
    existing_embeddings = []

    # Check for existing partial execution checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        try:
            chk = np.load(CHECKPOINT_FILE)
            checkpoint_data = chk["embeddings"]
            saved_count = len(checkpoint_data)
            if saved_count < total_docs:
                start_idx = saved_count
                existing_embeddings.append(checkpoint_data)
                logger.info(f"Resuming embedding generation from checkpoint at index {start_idx:,}/{total_docs:,}.")
            elif saved_count == total_docs:
                logger.info("Found complete checkpoint matching dataset size.")
                stats = {
                    "generation_time": 0.0,
                    "papers_per_second": 0.0,
                    "avg_batch_time": 0.0,
                    "total_batches": 0
                }
                return checkpoint_data, stats
        except Exception as e:
            logger.warning(f"Failed to load checkpoint file: {e}. Starting fresh generation.")

    logger.info(f"Generating dense embeddings for {total_docs - start_idx:,} remaining documents...")
    start_time = time.time()
    batch_times = []
    new_embeddings = []

    for i in tqdm(range(start_idx, total_docs, batch_size), desc="Embedding Batches"):
        batch_texts = texts[i:i + batch_size]
        b_start = time.time()

        # Generate embeddings with explicit L2 normalization for Cosine Similarity
        b_embeddings = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        new_embeddings.append(b_embeddings)
        b_time = time.time() - b_start
        batch_times.append(b_time)

        # Save checkpoint every 25,000 papers
        if (i + len(batch_texts)) % 25000 == 0:
            temp_concat = np.vstack(existing_embeddings + new_embeddings)
            np.savez_compressed(CHECKPOINT_FILE, embeddings=temp_concat)
            logger.info(f"Checkpoint saved at {i + len(batch_texts):,} records.")

    all_embeddings = np.vstack(existing_embeddings + new_embeddings) if existing_embeddings else np.vstack(new_embeddings)
    elapsed_time = time.time() - start_time

    # Clean up temporary checkpoint upon full completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    stats = {
        "generation_time": round(elapsed_time, 2),
        "papers_per_second": round(total_docs / elapsed_time, 2) if elapsed_time > 0 else 0.0,
        "avg_batch_time": round(float(np.mean(batch_times)), 4) if batch_times else 0.0,
        "total_batches": len(batch_times)
    }

    return all_embeddings, stats


# -----------------------------------------------------------------------------
# Validation & Quality Assurance
# -----------------------------------------------------------------------------
def validate_embeddings(embeddings: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """Performs validation checks on generated embedding matrix."""
    logger.info("Executing validation suite on generated embeddings...")
    num_papers = len(df)
    num_embeddings = embeddings.shape[0]

    nan_count = int(np.isnan(embeddings).sum())
    inf_count = int(np.isinf(embeddings).sum())
    
    norms = np.linalg.norm(embeddings, axis=1)
    invalid_norms = int(np.sum((norms < 0.99) | (norms > 1.01)))
    dup_ids = int(df.duplicated(subset=["id"]).sum())

    val_records = [
        {"check_name": "row_count_match", "metric": f"{num_embeddings} == {num_papers}", "status": "PASS" if num_embeddings == num_papers else "FAIL"},
        {"check_name": "nan_values_count", "metric": nan_count, "status": "PASS" if nan_count == 0 else "FAIL"},
        {"check_name": "inf_values_count", "metric": inf_count, "status": "PASS" if inf_count == 0 else "FAIL"},
        {"check_name": "normalized_l2_norms", "metric": f"Invalid Norms: {invalid_norms}", "status": "PASS" if invalid_norms == 0 else "FAIL"},
        {"check_name": "duplicate_ids", "metric": dup_ids, "status": "PASS" if dup_ids == 0 else "FAIL"}
    ]

    val_df = pd.DataFrame(val_records)
    val_df.to_csv(os.path.join(REPORTS_DIR, "embedding_validation.csv"), index=False)
    return val_df


# -----------------------------------------------------------------------------
# Report & Visualization Generation
# -----------------------------------------------------------------------------
def generate_reports_and_plots(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    exec_stats: Dict[str, Any],
    device: torch.device
) -> None:
    """Generates numerical summary reports and diagnostic visual charts."""
    logger.info("Generating statistical reports and distribution plots...")
    norms = np.linalg.norm(embeddings, axis=1)

    # 1. Summary Statistics Report
    summary_stats = {
        "model_name": MODEL_NAME,
        "number_of_papers": len(embeddings),
        "embedding_dimension": embeddings.shape[1],
        "average_norm": round(float(np.mean(norms)), 4),
        "min_norm": round(float(np.min(norms)), 4),
        "max_norm": round(float(np.max(norms)), 4),
        "generation_time_seconds": exec_stats["generation_time"],
        "average_papers_per_second": exec_stats["papers_per_second"],
        "average_batch_time_seconds": exec_stats["avg_batch_time"],
        "device_used": str(device),
    }

    pd.DataFrame([summary_stats]).to_csv(os.path.join(REPORTS_DIR, "embedding_statistics.csv"), index=False)

    # 2. Plot: Embedding Norm Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(norms, bins=30, color="#2ecc71", edgecolor="black")
    plt.title("L2 Embedding Norm Distribution (Target: 1.0)")
    plt.xlabel("L2 Norm")
    plt.ylabel("Paper Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "embedding_norm_distribution.png"), dpi=300)
    plt.close()

    # 3. Plot: Embedding Dimension Magnitude Summary
    plt.figure(figsize=(10, 5))
    dim_means = np.mean(embeddings[:5000], axis=0)  # Sample first 5000 for fast summary
    plt.plot(dim_means, color="#3498db", linewidth=1)
    plt.title(f"Average Vector Magnitude across {embeddings.shape[1]} Dimensions")
    plt.xlabel("Vector Dimension Index")
    plt.ylabel("Mean Activation")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "embedding_dimension_summary.png"), dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Data Persistence
# -----------------------------------------------------------------------------
def save_artifacts(embeddings: np.ndarray, df: pd.DataFrame) -> None:
    """Persists binary embeddings, paper ID arrays, and parquet metadata."""
    logger.info(f"Saving binary embedding matrix to: {EMBEDDINGS_FILE}")
    np.save(EMBEDDINGS_FILE, embeddings)

    logger.info(f"Saving paper ID array to: {IDS_FILE}")
    np.save(IDS_FILE, df["id"].astype(str).values)

    logger.info(f"Saving metadata Parquet table to: {METADATA_FILE}")
    available_cols = [c for c in METADATA_COLUMNS if c in df.columns]
    df[available_cols].to_parquet(METADATA_FILE, index=False)


# -----------------------------------------------------------------------------
# Main Execution Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    """Executes the complete embedding generation pipeline."""
    setup_directories()
    logger.info("Starting Dense Embedding Generation Pipeline...")

    device = get_device()
    df = load_dataset(INPUT_FILE_PATH)

    # Prepare document texts for embedding
    df["combined_text"] = format_embedding_text(df)
    formatted_texts = df["combined_text"].tolist()

    model = initialize_model(MODEL_NAME, device)

    embeddings, exec_stats = generate_embeddings(
        texts=formatted_texts,
        model=model,
        batch_size=DEFAULT_BATCH_SIZE
    )

    val_df = validate_embeddings(embeddings, df)
    generate_reports_and_plots(embeddings, df, exec_stats, device)
    save_artifacts(embeddings, df)

    logger.info("\n========== EMBEDDING GENERATION COMPLETE ==========")
    logger.info(f"Total Papers Embedded  : {len(embeddings):,}")
    logger.info(f"Embedding Dimension    : {embeddings.shape[1]}")
    logger.info(f"Generation Throughput  : {exec_stats['papers_per_second']:,} papers/sec")
    logger.info(f"Embeddings Saved To    : {EMBEDDINGS_FILE}")
    logger.info(f"Metadata Saved To      : {METADATA_FILE}")
    logger.info("===================================================\n")


if __name__ == "__main__":
    main()