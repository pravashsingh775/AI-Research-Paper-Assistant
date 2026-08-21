# SCIENTIFIC EVIDENCE RETRIEVAL REPORT

**Query:** Attention mechanisms and vision transformers for autonomous perception

**Task:** `QUESTION_ANSWERING`

**Retrieval Profile:** `PRECISION_ORIENTED`

**Reranker:** `CROSS_ENCODER`

**Evaluation Mode:** `PROXY_ESTIMATE`

## Retrieval Metrics

- Precision@K: 0.0
- Recall@K: 0.0
- MRR: 1.0
- MAP: 0.0
- nDCG@K: 1.0
- Diversity: 0.875
- Coverage: 91.39

## Coverage

- Overall: 48.5%
- Missing topics: Applications

## Evidence Context

```text
# SCIENTIFIC EVIDENCE CONTEXT PACKAGE

Use only the evidence below for grounded claims.

Do not infer experimental results that are not present in the excerpts.



--- EVIDENCE BLOCK [1] ---
Evidence ID: abs-2507.17616v1_C01
Paper ID: abs-2507.17616v1
Title: Vision Transformer attention alignment with human visual perception in aesthetic object evaluation
Authors: ['Miguel Carrasco', 'César González-Martín', 'José Aranda', 'Luis Oliveros']
Category: Computer Vision and Pattern Recognition
Domain: Computer Vision
Publication Year: 2025
Section: Experiment
Similarity Score: 0.8210
Rerank Score: 0.9850
Evidence Quality Score: 0.9485
Excerpt:
"Visual attention mechanisms play a crucial role in human perception and aesthetic evaluation. Recent advances in Vision Transformers (ViTs) have demonstrated remarkable capabilities in computer vision tasks, yet their alignment with human visual attention patterns remains underexplored, particularly in aesthetic contexts. This study investigates the correlation between human visual attention and ViT attention mechanisms when evaluating handcrafted objects. We conducted an eye-tracking experiment with 30 participants (9 female, 21 male, mean age 24.6 years) who viewed 20 artisanal objects comprising basketry bags and ginger jars. Using a Pupil Labs eye-tracker, we recorded gaze patterns and generated heat maps representing human visual attention. Simultaneously, we analyzed the same objects using a pre-trained ViT model with DINO (Self-DIstillation with NO Labels), extracting attention maps from each of the 12 attention heads. We compared human and ViT attention distributions using Kullback-Leibler divergence across varying Gaussian parameters (sigma=0.1 to 3.0). Statistical analysis revealed optimal correlation at sigma=2.4 +-0.03, with attention head #12 showing the strongest alignment with human visual patterns. Significant differences were found between attention heads, with heads #7 and #9 demonstrating the"

--- EVIDENCE BLOCK [2] ---
Evidence ID: abs-2507.17616v1_C02
Paper ID: abs-2507.17616v1
Title: Vision Transformer attention alignment with human visual perception in aesthetic object evaluation
Authors: ['Miguel Carrasco', 'César González-Martín', 'José Aranda', 'Luis Oliveros']
Category: Computer Vision and Pattern Recognition
Domain: Computer Vision
Publication Year: 2025
Section: Results
Similarity Score: 0.8210
Rerank Score: 0.9850
Evidence Quality Score: 0.9485
Excerpt:
"3.0). Statistical analysis revealed optimal correlation at sigma=2.4 +-0.03, with attention head #12 showing the strongest alignment with human visual patterns. Significant differences were found between attention heads, with heads #7 and #9 demonstrating the greatest divergence from human attention (p< 0.05, Tukey HSD test). Results indicate that while ViTs exhibit more global attention patterns compared to human focal attention, certain attention heads can approximate human visual behavior, particularly for specific object features like buckles in basketry items. These findings suggest potential applications of ViT attention mechanisms in product design and aesthetic evaluation, while highlighting fundamental differences in attention strategies between human perception and current AI models."

--- EVIDENCE BLOCK [3] ---
Evidence ID: abs-2512.21542v1_C01
Paper ID: abs-2512.21542v1
Title: Vision Transformers are Circulant Attention Learners
Authors: ['Dongchen Han', 'Tianyu Li', 'Ziyi Wang', 'Gao Huang']
Category: Computer Vision and Pattern Recognition
Domain: Computer Vision
Publication Year: 2025
Section: Experiments
Similarity Score: 0.8079
Rerank Score: 0.9385
Evidence Quality Score: 0.9202
Excerpt:
"The self-attention mechanism has been a key factor in the advancement of vision Transformers. However, its quadratic complexity imposes a heavy computational burden in high-resolution scenarios, restricting the practical application. Previous methods attempt to mitigate this issue by introducing handcrafted patterns such as locality or sparsity, which inevitably compromise model capacity. In this paper, we present a novel attention paradigm termed \textbf{Circulant Attention} by exploiting the inherent efficient pattern of self-attention. Specifically, we first identify that the self-attention matrix in vision Transformers often approximates the Block Circulant matrix with Circulant Blocks (BCCB), a kind of structured matrix whose multiplication with other matrices can be performed in $\mathcal{O}(N\log N)$ time. Leveraging this interesting pattern, we explicitly model the attention map as its nearest BCCB matrix and propose an efficient computation algorithm for fast calculation. The resulting approach closely mirrors vanilla self-attention, differing only in its use of BCCB matrices. Since our design is inspired by the inherent efficient paradigm, it not only delivers $\mathcal{O}(N\log N)$ computation complexity, but also largely maintains the capacity of standard self-attention. Extensive experiments on diverse visual"

--- EVIDENCE BLOCK [4] ---
Evidence ID: abs-2207.08569v3_C01
Paper ID: abs-2207.08569v3
Title: Multi-manifold Attention for Vision Transformers
Authors: ['Dimitrios Konstantinidis', 'Ilias Papastratis', 'Kosmas Dimitropoulos', 'Petros Daras']
Category: Computer Vision and Pattern Recognition
Domain: Computer Vision
Publication Year: 2022
Section: Results
Similarity Score: 0.8151
Rerank Score: 0.9431
Evidence Quality Score: 0.9134
Excerpt:
"Vision Transformers are very popular nowadays due to their state-of-the-art performance in several computer vision tasks, such as image classification and action recognition. Although their performance has been greatly enhanced through highly descriptive patch embeddings and hierarchical structures, there is still limited research on utilizing additional data representations so as to refine the selfattention map of a Transformer. To address this problem, a novel attention mechanism, called multi-manifold multihead attention, is proposed in this work to substitute the vanilla self-attention of a Transformer. The proposed mechanism models the input space in three distinct manifolds, namely Euclidean, Symmetric Positive Definite and Grassmann, thus leveraging different statistical and geometrical properties of the input for the computation of a highly descriptive attention map. In this way, the proposed attention mechanism can guide a Vision Transformer to become more attentive towards important appearance, color and texture features of an image, leading to improved classification and segmentation results, as shown by the experimental results on well-known datasets."

--- EVIDENCE BLOCK [5] ---
Evidence ID: abs-2108.01778v1_C01
Paper ID: abs-2108.01778v1
Title: Armour: Generalizable Compact Self-Attention for Vision Transformers
Authors: ['Lingchuan Meng']
Category: Computer Vision and Pattern Recognition
Domain: Computer Vision
Publication Year: 2021
Section: Method
Similarity Score: 0.8372
Rerank Score: 0.8906
Evidence Quality Score: 0.9004
Excerpt:
"Attention-based transformer networks have demonstrated promising potential as their applications extend from natural language processing to vision. However, despite the recent improvements, such as sub-quadratic attention approximation and various training enhancements, the compact vision transformers to date using the regular attention still fall short in comparison with its convnet counterparts, in terms of \textit{accuracy,} \textit{model size}, \textit{and} \textit{throughput}. This paper introduces a compact self-attention mechanism that is fundamental and highly generalizable. The proposed method reduces redundancy and improves efficiency on top of the existing attention optimizations. We show its drop-in applicability for both the regular attention mechanism and some most recent variants in vision transformers. As a result, we produced smaller and faster models with the same or better accuracies."

--- EVIDENCE BLOCK [6] ---
Evidence ID: abs-2506.12982v1_C01
Paper ID: abs-2506.12982v1
Title: DuoFormer: Leveraging Hierarchical Representations by Local and Global Attention Vision Transformer
Authors: ['Xiaoya Tang', 'Bodong Zhang', 'Man Minh Ho', 'Beatrice S. Knudsen', 'Tolga Tasdizen']
Category: Computer Vision and Pattern Recognition
Domain: Computer Vision
Publication Year: 2025
Section: Abstract
Similarity Score: 0.8133
Rerank Score: 0.8263
Evidence Quality Score: 0.8962
Excerpt:
"Despite the widespread adoption of transformers in medical applications, the exploration of multi-scale learning through transformers remains limited, while hierarchical representations are considered advantageous for computer-aided medical diagnosis. We propose a novel hierarchical transformer model that adeptly integrates the feature extraction capabilities of Convolutional Neural Networks (CNNs) with the advanced representational potential of Vision Transformers (ViTs). Addressing the lack of inductive biases and dependence on extensive training datasets in ViTs, our model employs a CNN backbone to generate hierarchical visual representations. These representations are adapted for transformer input through an innovative patch tokenization process, preserving the inherited multi-scale inductive biases. We also introduce a scale-wise attention mechanism that directly captures intra-scale and inter-scale associations. This mechanism complements patch-wise attention by enhancing spatial understanding and preserving global perception, which we refer to as local and global attention, respectively. Our model significantly outperforms baseline models in terms of classification accuracy, demonstrating its efficiency in bridging the gap between Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). The components are designed as plug-and-play for different CNN architectures and can be adapted for multiple"

--- EVIDENCE BLOCK [7] ---
Evidence ID: abs-2201.10801v1_C01
Paper ID: abs-2201.10801v1
Title: When Shift Operation Meets Vision Transformer: An Extremely Simple Alternative to Attention Mechanism
Authors: ['Guangting Wang', 'Yucheng Zhao', 'Chuanxin Tang', 'Chong Luo', 'Wenjun Zeng']
Category: Computer Vision and Pattern Recognition
Domain: Computer Vision
Publication Year: 2022
Section: Results
Similarity Score: 0.8500
Rerank Score: 0.8663
Evidence Quality Score: 0.8930
Excerpt:
"Attention mechanism has been widely believed as the key to success of vision transformers (ViTs), since it provides a flexible and powerful way to model spatial relationships. However, is the attention mechanism truly an indispensable part of ViT? Can it be replaced by some other alternatives? To demystify the role of attention mechanism, we simplify it into an extremely simple case: ZERO FLOP and ZERO parameter. Concretely, we revisit the shift operation. It does not contain any parameter or arithmetic calculation. The only operation is to exchange a small portion of the channels between neighboring features. Based on this simple operation, we construct a new backbone network, namely ShiftViT, where the attention layers in ViT are substituted by shift operations. Surprisingly, ShiftViT works quite well in several mainstream tasks, e.g., classification, detection, and segmentation. The performance is on par with or even better than the strong baseline Swin Transformer. These results suggest that the attention mechanism might not be the vital factor that makes ViT successful. It can be even replaced by a zero-parameter operation. We should pay more attentions"

--- EVIDENCE BLOCK [8] ---
Evidence ID: abs-2504.19414v1_C01
Paper ID: abs-2504.19414v1
Title: GMAR: Gradient-Driven Multi-Head Attention Rollout for Vision Transformer Interpretability
Authors: ['Sehyeong Jo', 'Gangjae Jang', 'Haesol Park']
Category: Computer Vision and Pattern Recognition
Domain: Computer Vision
Publication Year: 2025
Section: Method
Similarity Score: 0.8227
Rerank Score: 0.8230
Evidence Quality Score: 0.8910
Excerpt:
"The Vision Transformer (ViT) has made significant advancements in computer vision, utilizing self-attention mechanisms to achieve state-of-the-art performance across various tasks, including image classification, object detection, and segmentation. Its architectural flexibility and capabilities have made it a preferred choice among researchers and practitioners. However, the intricate multi-head attention mechanism of ViT presents significant challenges to interpretability, as the underlying prediction process remains opaque. A critical limitation arises from an observation commonly noted in transformer architectures: "Not all attention heads are equally meaningful." Overlooking the relative importance of specific heads highlights the limitations of existing interpretability methods. To address these challenges, we introduce Gradient-Driven Multi-Head Attention Rollout (GMAR), a novel method that quantifies the importance of each attention head using gradient-based scores. These scores are normalized to derive a weighted aggregate attention score, effectively capturing the relative contributions of individual heads. GMAR clarifies the role of each head in the prediction process, enabling more precise interpretability at the head level. Experimental results demonstrate that GMAR consistently outperforms traditional attention rollout techniques. This work provides a practical contribution to transformer-based architectures, establishing"
```