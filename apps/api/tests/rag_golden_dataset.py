"""Golden evaluation dataset for Lumen Research RAG pipeline benchmarking.

Contains 3 multi-page synthetic scientific papers with realistic academic sections:
- Paper A: Genomic Transformers (Sparse attention, DNA-500K benchmark)
- Paper B: Multimodal Knowledge Graphs (CLIP+KG fusion, ImageNet-KG)
- Paper C: Federated Quantization (Edge IoT, Non-IID FedAvg)

Covers 22 comprehensive test queries across 10 evaluation categories:
1. factual
2. methodology
3. dataset
4. results
5. limitations
6. comparison
7. citation
8. multihop
9. negative (unanswerable - must abstain)
10. entity_confusion (distractor entity present)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GoldenQuery:
    query_id: str
    question: str
    category: str
    target_paper_id: str
    expected_chunk_indices: list[int]
    expected_facts: list[str]
    answerable: bool
    distractor_paper_id: str | None = None


PAPERS_CORPUS: dict[str, dict[str, Any]] = {
    "paper-genomics-2025": {
        "title": "Sparse Transformer Architectures for Long-Context Genomic Sequence Modeling",
        "abstract": "Deep learning models in genomics require processing long sequence windows exceeding 100,000 base pairs. We present GenoSparse, a sparse attention transformer with linear computational complexity O(N). Evaluated on the human GRCh38 benchmark and the synthetic GeneSplice dataset, GenoSparse achieves 94.2% AUROC in splice site identification while reducing GPU memory consumption by 68% compared to standard quadratic self-attention.",
        "text": """Abstract
Deep learning models in genomics require processing long sequence windows exceeding 100,000 base pairs. We present GenoSparse, a sparse attention transformer with linear computational complexity O(N). Evaluated on the human GRCh38 benchmark and the synthetic GeneSplice dataset, GenoSparse achieves 94.2% AUROC in splice site identification while reducing GPU memory consumption by 68% compared to standard quadratic self-attention.

Introduction
Genomic modeling presents computational bottlenecks for deep attention mechanisms due to sequence lengths spanning hundreds of thousands of nucleotides. Prior work such as Enformer and Nucleotide Transformer utilized fixed convolution pyramids or truncated context windows, sacrificing long-range enhancer-promoter regulatory interactions. In this work, we demonstrate that block-sparse banded attention with learnable genomic landmark tokens captures regulatory biology across 128k nucleotide context windows.

Materials and Methods
The GenoSparse model implements a block-sparse attention mask where each nucleotide token attends to a local window of 256 surrounding base pairs and 64 global landmark tokens placed at transcription start sites. The architecture employs 24 transformer layers with hidden dimension 768 and 12 attention heads. We optimize model parameters using AdamW with cosine learning rate scheduling from 1e-4 down to 1e-6 across 150 epochs with weight decay 0.01.

Datasets
Experiments were conducted on the ENCODE GRCh38 human reference genome release 42 containing 3.2 billion base pairs partitioned into 22 autosomes. The training partition comprises chromosomes 1 through 18, chromosome 19 is reserved for hyperparameter validation, and chromosomes 20 through 22 constitute the held-out test evaluation set. We also evaluate zero-shot transfer on the GeneSplice benchmark consisting of 450,000 annotated donor and acceptor splice junctions.

Results and Findings
On the held-out test set (chromosomes 20-22), GenoSparse achieved an AUROC of 94.2% and an average precision (AUPRC) of 89.7% on splice site classification, outperforming the full quadratic baseline (91.8% AUROC) while training 3.4 times faster. Memory utilization remained bounded under 14.2 GB VRAM on a single NVIDIA A100 GPU for sequence lengths of 131,072 base pairs.

Strengths
GenoSparse scales linearly with context length while retaining single-nucleotide resolution. The integration of transcription landmark tokens preserves distal enhancer-promoter contact dynamics that window-only sparse transformers routinely fail to detect.

Limitations and Weaknesses
Our approach assumes pre-identified transcription start site landmarks, making it less applicable to unannotated non-model organisms. Furthermore, inference latency increases on unoptimized hardware lacking dedicated block-sparse tensor kernels.

Discussion and Future Work
Future work will focus on learnable dynamic landmark discovery to support metagenomic assembly and exploring cross-species evolutionary adaptation without fixed reference genomes.
""",
    },
    "paper-multimodal-2025": {
        "title": "Contrastive Multimodal Representation Learning with Knowledge Graph Guidance",
        "abstract": "Visual-language foundation models often struggle with complex factual reasoning and entity relationships. We introduce KG-CLIP, an architecture that injects structured relational knowledge from Wikidata directly into contrastive multi-modal embeddings. Across ImageNet-KG and MS-COCO zero-shot benchmarks, KG-CLIP demonstrates a 6.3% top-1 accuracy improvement over standard CLIP while substantially reducing semantic hallucination.",
        "text": """Abstract
Visual-language foundation models often struggle with complex factual reasoning and entity relationships. We introduce KG-CLIP, an architecture that injects structured relational knowledge from Wikidata directly into contrastive multi-modal embeddings. Across ImageNet-KG and MS-COCO zero-shot benchmarks, KG-CLIP demonstrates a 6.3% top-1 accuracy improvement over standard CLIP while substantially reducing semantic hallucination.

Introduction
Contrastive visual-language pre-training pairs images with descriptive captions but ignores dense relational knowledge between entities. When asked to distinguish between morphologically similar species or fine-grained historical artifacts, conventional models produce ungrounded associations. We propose injecting tripartite knowledge graph embeddings into both visual and textual projections.

Methodology
KG-CLIP incorporates a dual-encoder backbone featuring a ViT-L/14 visual encoder and a 12-layer text Transformer. Relational knowledge graphs are encoded using a 3-layer Relational Graph Convolutional Network (R-GCN) operating over Wikidata triplets (head, relation, tail). The cross-modal contrastive loss is augmented with a structured knowledge alignment penalty with margin gamma = 0.5.

Datasets
We evaluate on the curated ImageNet-KG dataset comprising 1,000 fine-grained visual categories linked to Wikidata entity nodes with an average graph depth of 4.2 hops. Pre-training utilized CC3M (Conceptual Captions) aligned with 1.4 million entity facts. Zero-shot image-text retrieval is tested on the 5,000-image MS-COCO test split.

Results and Findings
KG-CLIP achieves 81.4% zero-shot top-1 accuracy on ImageNet-KG, exceeding standard CLIP (75.1%) by 6.3 percentage points. On MS-COCO zero-shot text-to-image retrieval, KG-CLIP reaches Recall@1 of 44.8% compared to 38.2% for the baseline, demonstrating superior entity disambiguation.

Strengths
The model demonstrates high factual precision and significantly reduced hallucinations on rare entity queries. Knowledge graph guidance enables zero-shot classification on concepts unrepresented in textual captions during training.

Limitations
The primary limitation is dependency on high-quality knowledge graphs; incomplete or noisy relational edges degrade cross-modal alignment. Additionally, computing graph convolutions introduces an 18% computational overhead during initial feature indexing.

Future Work
We plan to extend KG-CLIP to continuous knowledge graph refinement where the model self-corrects graph inconsistencies during multimodal pre-training.
""",
    },
    "paper-federated-2025": {
        "title": "Federated Optimization on Heterogeneous Edge Devices with Adaptive Quantization",
        "abstract": "Federated learning across heterogeneous mobile edge devices suffers from severe communication bottlenecks and client drift under non-IID data distributions. We propose AdaQuant-FL, an adaptive 2-to-8 bit gradient quantization scheme paired with momentum compensation. On the CIFAR-10 and Shakespeare edge federated benchmarks, AdaQuant-FL cuts communication overhead by 78% while matching full-precision convergence speed within 1.2% final accuracy.",
        "text": """Abstract
Federated learning across heterogeneous mobile edge devices suffers from severe communication bottlenecks and client drift under non-IID data distributions. We propose AdaQuant-FL, an adaptive 2-to-8 bit gradient quantization scheme paired with momentum compensation. On the CIFAR-10 and Shakespeare edge federated benchmarks, AdaQuant-FL cuts communication overhead by 78% while matching full-precision convergence speed within 1.2% final accuracy.

Introduction
Decentralized edge computing promises private machine learning without centralized data aggregation. However, client devices such as smartphones, smart home hubs, and wearable sensors have highly constrained wireless uplink bandwidth and heterogeneous computational capabilities. Uniform quantization leads to quantization error accumulation and divergence under non-IID client partitions.

Methodology
AdaQuant-FL employs dynamic bit-width allocation determined by client gradient variance and instantaneous network latency. Each participating device measures gradient variance across local batches and compresses updates into ternary (2-bit), nibble (4-bit), or byte (8-bit) representations. A centralized server applies momentum-corrected stochastic gradient reconstruction to offset quantization variance.

Datasets
Evaluation was performed across two federated benchmark benchmarks: Dirichlet non-IID partitioned CIFAR-10 across 100 simulated edge clients (alpha = 0.1), and the LEAF Shakespeare language modeling benchmark containing 1,129 speaking roles simulating extreme natural client data heterogeneity.

Results and Findings
AdaQuant-FL reduced total uploaded megabytes from 4.8 GB (full precision 32-bit) down to 1.05 GB per client across 200 communication rounds, achieving a 78% communication reduction. Final top-1 classification accuracy on non-IID CIFAR-10 reached 86.4%, within 0.8% of uncompressed FedAvg (87.2%).

Strengths
AdaQuant-FL guarantees convergence under arbitrary client data heterogeneity and automatically adapts to fluctuating edge wireless link conditions without manual hyperparameter tuning.

Limitations
Quantization compression requires extra local compute cycles for variance calculation, leading to a 4% increase in on-device battery consumption during local client rounds.

Future Work
Future investigations will explore asynchronous client updates and hardware-accelerated integer arithmetic on specialized microcontrollers.
""",
    },
}

GOLDEN_QUERIES: list[GoldenQuery] = [
    # 1. Factual
    GoldenQuery(
        query_id="q01_factual_splice_auroc",
        question="What AUROC does GenoSparse achieve in splice site identification?",
        category="factual",
        target_paper_id="paper-genomics-2025",
        expected_chunk_indices=[0, 4],
        expected_facts=["94.2%", "AUROC", "splice site"],
        answerable=True,
    ),
    # 2. Methodology
    GoldenQuery(
        query_id="q02_method_attention_window",
        question="What local window size and landmark tokens does GenoSparse use in its block-sparse attention?",
        category="methodology",
        target_paper_id="paper-genomics-2025",
        expected_chunk_indices=[2],
        expected_facts=["256", "base pairs", "64", "global landmark tokens"],
        answerable=True,
    ),
    # 3. Dataset
    GoldenQuery(
        query_id="q03_dataset_encode_chromosomes",
        question="Which chromosomes were used for training, validation, and testing in the human genome evaluation?",
        category="dataset",
        target_paper_id="paper-genomics-2025",
        expected_chunk_indices=[3],
        expected_facts=["chromosomes 1 through 18", "chromosome 19", "chromosomes 20 through 22"],
        answerable=True,
    ),
    # 4. Results
    GoldenQuery(
        query_id="q04_results_imagenet_accuracy",
        question="What was the zero-shot top-1 accuracy achieved by KG-CLIP on ImageNet-KG compared to standard CLIP?",
        category="results",
        target_paper_id="paper-multimodal-2025",
        expected_chunk_indices=[0, 4],
        expected_facts=["81.4%", "75.1%", "6.3%"],
        answerable=True,
    ),
    # 5. Limitations
    GoldenQuery(
        query_id="q05_limitations_genomics_landmarks",
        question="What is the primary limitation of GenoSparse regarding non-model organisms?",
        category="limitations",
        target_paper_id="paper-genomics-2025",
        expected_chunk_indices=[6],
        expected_facts=["pre-identified transcription start site landmarks", "unannotated non-model organisms"],
        answerable=True,
    ),
    # 6. Limitations
    GoldenQuery(
        query_id="q06_limitations_kg_clip_noise",
        question="What limitations exist in KG-CLIP when relational edges are incomplete or noisy?",
        category="limitations",
        target_paper_id="paper-multimodal-2025",
        expected_chunk_indices=[6],
        expected_facts=["incomplete or noisy relational edges", "18% computational overhead"],
        answerable=True,
    ),
    # 7. Comparison
    GoldenQuery(
        query_id="q07_comparison_full_precision_cifar",
        question="How does AdaQuant-FL compare to uncompressed FedAvg in final accuracy on CIFAR-10?",
        category="comparison",
        target_paper_id="paper-federated-2025",
        expected_chunk_indices=[0, 4],
        expected_facts=["86.4%", "87.2%", "0.8%"],
        answerable=True,
    ),
    # 8. Methodology
    GoldenQuery(
        query_id="q08_method_graph_convolution",
        question="What graph neural network architecture is used in KG-CLIP to encode knowledge graph triplets?",
        category="methodology",
        target_paper_id="paper-multimodal-2025",
        expected_chunk_indices=[2],
        expected_facts=["Relational Graph Convolutional Network", "R-GCN", "3-layer"],
        answerable=True,
    ),
    # 9. Dataset
    GoldenQuery(
        query_id="q09_dataset_federated_leaf",
        question="Which benchmarks were used to evaluate AdaQuant-FL under non-IID edge conditions?",
        category="dataset",
        target_paper_id="paper-federated-2025",
        expected_chunk_indices=[3],
        expected_facts=["CIFAR-10", "Shakespeare", "100 simulated edge clients", "1,129 speaking roles"],
        answerable=True,
    ),
    # 10. Multi-hop
    GoldenQuery(
        query_id="q10_multihop_genomics_context_vram",
        question="What maximum sequence context length does GenoSparse process and how much GPU VRAM does it consume?",
        category="multihop",
        target_paper_id="paper-genomics-2025",
        expected_chunk_indices=[1, 4],
        expected_facts=["131,072", "14.2 GB VRAM"],
        answerable=True,
    ),
    # 11. Multi-hop
    GoldenQuery(
        query_id="q11_multihop_adaquant_bandwidth",
        question="What was the total uploaded data reduction and final megabyte volume per client in AdaQuant-FL?",
        category="multihop",
        target_paper_id="paper-federated-2025",
        expected_chunk_indices=[0, 4],
        expected_facts=["78%", "1.05 GB", "4.8 GB"],
        answerable=True,
    ),
    # 12. Citation / Provenance
    GoldenQuery(
        query_id="q12_citation_coco_retrieval",
        question="What zero-shot text-to-image Recall@1 score on MS-COCO is reported for KG-CLIP?",
        category="citation",
        target_paper_id="paper-multimodal-2025",
        expected_chunk_indices=[4],
        expected_facts=["44.8%", "Recall@1", "38.2%"],
        answerable=True,
    ),
    # 13. Strengths
    GoldenQuery(
        query_id="q13_strengths_genomics_distal",
        question="What biological interactions do transcription landmark tokens preserve in GenoSparse?",
        category="strengths",
        target_paper_id="paper-genomics-2025",
        expected_chunk_indices=[1, 5],
        expected_facts=["enhancer-promoter", "distal enhancer-promoter contact dynamics"],
        answerable=True,
    ),
    # 14. Strengths
    GoldenQuery(
        query_id="q14_strengths_edge_heterogeneity",
        question="What convergence guarantee does AdaQuant-FL provide under edge conditions?",
        category="strengths",
        target_paper_id="paper-federated-2025",
        expected_chunk_indices=[5],
        expected_facts=["arbitrary client data heterogeneity", "fluctuating edge wireless"],
        answerable=True,
    ),
    # 15. Negative (Unanswerable - Must Abstain)
    GoldenQuery(
        query_id="q15_negative_quantum_computing",
        question="What quantum annealing schedule was employed to optimize the GenoSparse Hamiltonian?",
        category="negative",
        target_paper_id="paper-genomics-2025",
        expected_chunk_indices=[],
        expected_facts=[],
        answerable=False,
    ),
    # 16. Negative (Unanswerable - Must Abstain)
    GoldenQuery(
        query_id="q16_negative_satellite_imagery",
        question="How many multispectral satellite imagery bands were used for KG-CLIP earth observation?",
        category="negative",
        target_paper_id="paper-multimodal-2025",
        expected_chunk_indices=[],
        expected_facts=[],
        answerable=False,
    ),
    # 17. Negative (Unanswerable - Must Abstain)
    GoldenQuery(
        query_id="q17_negative_blockchain_consensus",
        question="What proof-of-stake consensus protocol validates client gradient blocks in AdaQuant-FL?",
        category="negative",
        target_paper_id="paper-federated-2025",
        expected_chunk_indices=[],
        expected_facts=[],
        answerable=False,
    ),
    # 18. Entity Confusion (Distractor Test)
    GoldenQuery(
        query_id="q18_distractor_vit_in_genomics",
        question="Does GenoSparse use a Vision Transformer ViT-L/14 backbone for DNA encoding?",
        category="entity_confusion",
        target_paper_id="paper-genomics-2025",
        expected_chunk_indices=[],
        expected_facts=[],
        answerable=False,
        distractor_paper_id="paper-multimodal-2025",
    ),
    # 19. Entity Confusion (Distractor Test)
    GoldenQuery(
        query_id="q19_distractor_genomic_benchmarks_in_federated",
        question="Was AdaQuant-FL evaluated on the ENCODE GRCh38 human genome dataset?",
        category="entity_confusion",
        target_paper_id="paper-federated-2025",
        expected_chunk_indices=[],
        expected_facts=[],
        answerable=False,
        distractor_paper_id="paper-genomics-2025",
    ),
    # 20. Future Work
    GoldenQuery(
        query_id="q20_future_work_metagenomics",
        question="What directions for learnable dynamic landmark discovery are proposed for future work in GenoSparse?",
        category="future_work",
        target_paper_id="paper-genomics-2025",
        expected_chunk_indices=[7],
        expected_facts=["metagenomic assembly", "cross-species evolutionary adaptation"],
        answerable=True,
    ),
    # 21. Future Work
    GoldenQuery(
        query_id="q21_future_work_self_correcting_kg",
        question="What self-correction mechanism is planned for future work in KG-CLIP?",
        category="future_work",
        target_paper_id="paper-multimodal-2025",
        expected_chunk_indices=[7],
        expected_facts=["continuous knowledge graph refinement", "self-corrects graph inconsistencies"],
        answerable=True,
    ),
    # 22. Limitations
    GoldenQuery(
        query_id="q22_limitations_edge_battery",
        question="How much does AdaQuant-FL increase on-device battery consumption during local client rounds?",
        category="limitations",
        target_paper_id="paper-federated-2025",
        expected_chunk_indices=[6],
        expected_facts=["4%", "battery consumption", "variance calculation"],
        answerable=True,
    ),
]
