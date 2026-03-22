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
  <img src="ARISE/figure/model.png" width="850"/>
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
import scanpy as sc
from process import normalize, Protein, build_dual_graph, preprocess_HLN
from train import initialize_model, train_model
import argparse

# Load data
adata_RNA = sc.read_h5ad('./data/HLN/adata_RNA.h5ad')
adata_ADT = sc.read_h5ad('./data/HLN/adata_ADT.h5ad')

# Preprocess
RNA_data, ADT_data = preprocess_HLN(adata_RNA, adata_ADT)
cell_positions = adata_RNA.obsm['spatial']

# Build RNA-anchored shared-edge graph
graph_data = build_dual_graph(RNA_data, ADT_data, cell_positions, device=device)

# Initialize and train
args = argparse.Namespace(
    hidden_dim=512, out_dim=64, num_clusters=10,
    beta=25, gamma=10, delta=1, dropout=0,
    lr=1e-3, epochs=350
)
model = initialize_model(graph_data, RNA_data.shape[1], ADT_data.shape[1], args)
model, embeddings, labels = train_model(model, graph_data, args, true_labels)
```

---

## Reproducing Paper Results

### RNA + ADT: Human Lymph Node
```bash
python run_HLN.py \
    --hidden_dim 512 \
    --out_dim 64 \
    --num_clusters 10 \
    --beta 25 \
    --gamma 10 \
    --delta 1 \
    --lr 1e-3 \
    --epochs 350
```

Data should be placed as:
```
data/
└── HLN/
    ├── adata_RNA.h5ad
    ├── adata_ADT.h5ad
    └── GT_labels.txt
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


---

## Contact

For questions or issues, please open a GitHub Issue or contact:

- Yunhe Wang: wangyh082@hebut.edu.cn  
- Xiangtao Li: lixt314@jlu.edu.cn
