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
## Setup
```bash
git clone https://github.com/XiangxiangWang-code/ARISE.git
cd ARISE
pip install -r requirements.txt
```
---

## Data Preparation

ARISE accepts input in [AnnData](https://anndata.readthedocs.io/) `.h5ad` format. Each modality should be stored as a separate AnnData object with spatial coordinates in `adata.obsm['spatial']`.


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

## Reproducing Paper Results

Training scripts are located in `ARISE/code/`.

### RNA + ADT: Human Lymph Node
```bash
python ARISE/code/HLN.py
```

---

## Contact

For questions or issues, please open a GitHub Issue or contact:

- Xiangxiang Wang: 2319659260@qq.com
