# ARISE
# ARISE: RNA-Anchored Shared-Edge Topology and Hierarchical Fusion for Scalable Spatial Multi-Omics Integration

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" />
  <img src="https://img.shields.io/badge/PyTorch-1.12+-orange.svg" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
  <img src="https://img.shields.io/badge/Paper-bioRxiv-red.svg" />
</p>

<p align="center">
  <b>A</b>nchored <b>R</b>NA for <b>I</b>ntegrated <b>S</b>patial <b>E</b>mbedding
</p>

---

## Overview

**ARISE** is an RNA-anchored framework for spatial multi-omics integration. Rather than constructing independent modality-specific graphs, ARISE defines a **shared-edge topology** by intersecting an RNA feature-similarity graph with a spatial-proximity graph, retaining only edges supported by both transcriptional similarity and physical adjacency. Auxiliary modalities (ADT, ATAC, histone modifications) are encoded on this common scaffold, and an **inside-out hierarchical fusion** module integrates them into a unified latent representation.

<p align="center">
  <img src="main/ARISE/figure/model.png" width="850"/>
  <br>
  <em>Overview of the ARISE framework.</em>
</p>

**Key advantages:**
- Theoretically grounded: graph intersection minimizes false-positive edges across all k-of-r fusion rules (Theorems 1–3)
- Stable under perturbation: GNN encoder drift is provably linear in graph perturbation magnitude (Theorem 4)
- Modular: additional modalities can be incorporated without redefining the shared-edge topology
- Supports bi-modal (RNA+ADT, RNA+ATAC) and tri-modal (RNA+ATAC+Protein/Histone) settings

---

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12
- CUDA (recommended)

### Setup

```bash
git clone https://github.com/XiangxiangWang-code/ARISE.git
cd ARISE
conda create -n arise python=3.8
conda activate arise
pip install -r requirements.txt
```

### Dependencies

```
torch>=1.12.0
torch-geometric
scanpy>=1.9.0
anndata>=0.8.0
numpy
scipy
pandas
scikit-learn
matplotlib
seaborn
```

---

## Data Preparation

ARISE accepts input in [AnnData](https://anndata.readthedocs.io/) `.h5ad` format. Each modality should be stored as a separate AnnData object with spatial coordinates in `adata.obsm['spatial']`.

### Datasets Used in the Paper

| Dataset | Modalities | Spots | Source |
|---|---|---|---|
| Human Lymph Node (HLN) | RNA + ADT | 3,484 | [Long et al., 2024](https://www.nature.com/articles/s41592-024-02313-7) |
| Mouse Brain (P22) | RNA + ATAC | 9,215 | [Zhang et al., 2023](https://www.nature.com/articles/s41586-023-05795-1) |
| Mouse Thymus | RNA + ADT | 4,697 | [Liao et al., 2023](https://www.biorxiv.org/content/10.1101/2023.04.26.538404) |
| Mouse Embryo E13 | RNA + ATAC | 2,187 | [Zhang et al., 2023](https://www.nature.com/articles/s41586-023-05795-1) |
| Mouse Embryo (Spatial-Mux-seq) | RNA + ATAC + Protein | 10,000 | [Guo et al., 2024](https://doi.org/10.1101/2024.09.19.39345645) |
| Mouse Embryo (H3K27ac/H3K4me3/H3K27me3) | RNA + Histone | ~2,500–10,000 | [Guo et al., 2024](https://doi.org/10.1101/2024.09.19.39345645) |

### Data Format

```python
import anndata as ad

# RNA modality
adata_rna = ad.read_h5ad("rna.h5ad")         # shape: (n_spots, n_genes)

# Auxiliary modality (ADT or ATAC)
adata_adt = ad.read_h5ad("adt.h5ad")         # shape: (n_spots, n_proteins)

# Spatial coordinates must be stored in obsm
# adata_rna.obsm['spatial'] — shape: (n_spots, 2)
```

---

## Quick Start

```python
from arise import ARISE

# Initialize model
model = ARISE(
    modalities=["RNA", "ADT"],   # or ["RNA", "ATAC"] / ["RNA", "ATAC", "Protein"]
    n_neighbors=15,              # k for kNN graph construction
    latent_dim=64,               # embedding dimension
    alpha=10.0,                  # spatial regularization weight
)

# Train
model.fit(adata_rna, adata_adt)

# Get unified embedding
embedding = model.get_embedding()   # shape: (n_spots, latent_dim)

# Cluster
labels = model.cluster(n_clusters=8)
```

---

## Reproducing Paper Results

### RNA + ADT: Human Lymph Node

```bash
python scripts/run_hln.py \
    --rna_path data/HLN/rna.h5ad \
    --adt_path data/HLN/adt.h5ad \
    --n_clusters 8 \
    --n_neighbors 15 \
    --latent_dim 64 \
    --alpha 10
```

### RNA + ATAC: Mouse Brain

```bash
python scripts/run_mouse_brain.py \
    --rna_path data/MouseBrain/rna.h5ad \
    --atac_path data/MouseBrain/atac.h5ad \
    --n_clusters 14 \
    --n_neighbors 15 \
    --latent_dim 64 \
    --alpha 10
```

### RNA + ATAC: Embryo E13

```bash
python scripts/run_e13.py \
    --rna_path data/E13/rna.h5ad \
    --atac_path data/E13/atac.h5ad \
    --n_neighbors 15 \
    --latent_dim 64 \
    --alpha 10
```

### Tri-modal: Mouse Embryo (Spatial-Mux-seq)

```bash
python scripts/run_trimodal.py \
    --rna_path data/ME/rna.h5ad \
    --atac_path data/ME/atac.h5ad \
    --protein_path data/ME/protein.h5ad \
    --n_neighbors 15 \
    --latent_dim 64 \
    --alpha 10
```

---

## Results

### Bi-modal Benchmarks

| Dataset | Method | ARI | AMI | NMI |
|---|---|---|---|---|
| Mouse Brain (RNA+ATAC) | **ARISE** | **0.4064** | **0.4635** | **0.4657** |
| | SpatialGlue | 0.3213 | 0.3498 | 0.3712 |
| | STAGATE | 0.3085 | 0.3401 | 0.3623 |
| | TotalVI | 0.0183 | 0.0341 | 0.0512 |
| Human Lymph Node (RNA+ADT) | **ARISE** | **0.3427** | **0.4141** | **0.4182** |
| | SpatialGlue | 0.2981 | 0.3624 | 0.3891 |
| | PRAGA | 0.3012 | 0.3801 | 0.3914 |

### Tri-modal Benchmarks (SC ↑ / DB ↓)

| Dataset | ARISE | PRAGA | MISO |
|---|---|---|---|
| ME (RNA+ATAC+Protein) | **best** | moderate | poor |
| ME H3K27ac | **best** | unstable | near-collapse |
| ME H3K4me3 | **best** | unstable | near-collapse |
| ME H3K27me3 | **best** | unstable | near-collapse |

---

## Method

ARISE proceeds in four steps:

**1. Preprocessing**
- RNA: top 3,000 HVGs, total-count normalization, log1p, centering
- ADT: centered log-ratio normalization
- ATAC: TF-IDF normalization + latent semantic indexing

**2. RNA-Anchored Shared-Edge Topology**

$$A_{\text{com}} = A_{\text{feat}} \cap A_{\text{spatial}}$$

Only edges supported by both transcriptional similarity and spatial proximity are retained. This minimizes false-positive edges (Theorems 1–3).

**3. Inside-Out Hierarchical Fusion**
- Dual RNA embeddings (feature graph + spatial graph) are first consolidated into an anchor representation
- Auxiliary modality embeddings, encoded on $A_{\text{com}}$, are then progressively incorporated

**4. Training Objective**

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{spatial}} + \beta \mathcal{L}_{\text{recon}} + \gamma \mathcal{L}_{\text{L1,L2}}$$

---

## Citation

If you find ARISE useful in your research, please cite:

```bibtex
@article{wang2024arise,
  title     = {ARISE: RNA-Anchored Shared-Edge Topology and Hierarchical Fusion 
               for Scalable Spatial Multi-Omics Integration},
  author    = {Wang, Xiangxiang and Su, Yanchi and Hao, Gaoyang and 
               Wang, Meng and Wang, Yunhe and Li, Xiangtao},
  journal   = {Journal Title Here},
  year      = {2024},
  url       = {https://github.com/XiangxiangWang-code/ARISE}
}
```

---

## Contact

For questions or issues, please open a GitHub Issue or contact:

- Yunhe Wang: wangyh082@hebut.edu.cn  
- Xiangtao Li: lixt314@jlu.edu.cn
