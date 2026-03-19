import numpy as np
import scanpy as sc
import scipy
import sklearn
import anndata
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Data

def clr_normalize_each_cell(adata, inplace=True):
    """
    Perform CLR (Centered Log-Ratio) normalization on protein (ADT) data per cell.

    Args:
        adata (AnnData): The input AnnData object containing the expression matrix.
        inplace (bool): Whether to modify the original AnnData object or return a copy.

    Returns:
        AnnData: The CLR-normalized AnnData object.
    """
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata



def pca(adata, use_reps=None, n_comps=10):
    """
    Perform PCA (Principal Component Analysis) on the input AnnData object.

    Args:
        adata (AnnData): The input data.
        use_reps (str, optional): Use a precomputed feature representation (e.g., adata.obsm['X_pca']).
        n_comps (int): Number of principal components.

    Returns:
        np.ndarray: The PCA-transformed matrix.
    """
    from sklearn.decomposition import PCA
    pca_model = PCA(n_components=n_comps)

    if use_reps is not None:
        feat_pca = pca_model.fit_transform(adata.obsm[use_reps])
    else:
        feat_pca = pca_model.fit_transform(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X)

    return feat_pca


def normalize(adata, highly_genes=3000):
    """
    Normalize RNA data and extract highly variable genes.

    Args:
        adata (AnnData): The input RNA data.
        highly_genes (int): Number of top highly variable genes to retain.

    Returns:
        AnnData: The processed AnnData object.
    """
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata



def Protein(adata):
    """
    Normalize ADT (Protein) data using CLR and scale it.

    Args:
        adata (AnnData): The input ADT data.

    Returns:
        AnnData: The processed ADT data.
    """
    adata = clr_normalize_each_cell(adata)
    sc.pp.scale(adata)
    return adata


def tfidf(X):
    """
    Compute TF-IDF matrix for input count matrix.

    Args:
        X (np.ndarray or sparse): Count matrix.

    Returns:
        np.ndarray or sparse: TF-IDF normalized matrix.
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf



def lsi(adata: anndata.AnnData, n_components, use_highly_variable: Optional[bool] = None, **kwargs) -> None:
    """
    Perform LSI (Latent Semantic Indexing) on AnnData.

    Args:
        adata (AnnData): The input data.
        n_components (int): Number of components.
        use_highly_variable (bool): Whether to use only highly variable genes.
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:, 1:]



def adata_preprocess_1(adata, min_cells=100, pca_n_comps=2000, HVG=3000):
    """
    Basic preprocessing pipeline for RNA modality.

    Args:
        adata (AnnData): RNA expression data.
        min_cells (int): Minimum cells per gene.
        pca_n_comps (int): Not used in current function.
        HVG (int): Number of highly variable genes.

    Returns:
        np.ndarray: Processed RNA expression matrix.
    """
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_counts=3)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=HVG)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata[:, adata.var['highly_variable']].X




class DualGraphData(Data):
    """
    Custom PyTorch Geometric Data structure for dual-graph learning.

    Attributes:
        x_RNA: RNA feature matrix.
        x_ADT: ADT feature matrix.
        sim_edge_index: Indices for similarity-based edges.
        sim_edge_weight: Weights for similarity edges.
        dist_edge_index: Indices for spatial distance-based edges.
        dist_edge_weight: Weights for spatial edges.
        common_edge_index: Overlap of similarity and spatial edges.
        common_edge_weight: Weights for common edges.
    """
    def __init__(self, x_RNA, x_ADT, sim_edge_index, sim_edge_weight,
                 dist_edge_index, dist_edge_weight, common_edge_index, common_edge_weight):
        super().__init__()
        self.x_RNA = x_RNA
        self.x_ADT = x_ADT
        self.sim_edge_index = sim_edge_index
        self.sim_edge_weight = sim_edge_weight
        self.dist_edge_index = dist_edge_index
        self.dist_edge_weight = dist_edge_weight
        self.common_edge_index = common_edge_index
        self.common_edge_weight = common_edge_weight



def build_dual_graph(RNA_expression, ADT_expression, cell_positions, device='cpu', num_neighbors=15):
    """
    Build a dual graph based on expression similarity and spatial proximity.

    Args:
        RNA_expression (np.ndarray): RNA expression matrix.
        ADT_expression (np.ndarray): ADT expression matrix.
        cell_positions (np.ndarray): 2D coordinates of cells.
        device (str): Device to place tensors ('cpu' or 'cuda').
        num_neighbors (int): Number of neighbors for KNN.

    Returns:
        DualGraphData: PyTorch Geometric-compatible graph object.
    """
    similarity_matrix = cosine_similarity(RNA_expression)

    nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='cosine').fit(RNA_expression)
    distances, indices = nbrs.kneighbors(RNA_expression)

    adjacency_matrix = np.zeros_like(similarity_matrix, dtype=int)
    for i in range(len(RNA_expression)):
        for j in indices[i][1:]:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1

    sim_edge_index = torch.tensor(np.array(np.nonzero(adjacency_matrix)), dtype=torch.long).to(device)
    sim_edge_weight = torch.tensor(similarity_matrix[adjacency_matrix > 0], dtype=torch.float).to(device)

    distance_matrix = cdist(cell_positions, cell_positions, metric='euclidean')
    knn_graph = kneighbors_graph(cell_positions, n_neighbors=num_neighbors, mode='distance', include_self=False)
    knn_graph = knn_graph.maximum(knn_graph.T)

    dist_edge_index = torch.tensor(knn_graph.nonzero(), dtype=torch.long).to(device)
    dist_edge_weight = torch.tensor(knn_graph.data, dtype=torch.float).to(device)

    sim_edges = set(zip(sim_edge_index[0].tolist(), sim_edge_index[1].tolist()))
    dist_edges = set(zip(dist_edge_index[0].tolist(), dist_edge_index[1].tolist()))
    common_edges = sim_edges.intersection(dist_edges)

    common_edge_index = torch.tensor(list(zip(*common_edges)), dtype=torch.long).to(device)
    common_edge_weight = torch.ones(common_edge_index.shape[1], dtype=torch.float).to(device)

    x_RNA = torch.tensor(RNA_expression, dtype=torch.float).to(device)
    x_ADT = torch.tensor(ADT_expression, dtype=torch.float).to(device)

    return DualGraphData(
        x_RNA=x_RNA,
        x_ADT=x_ADT,
        sim_edge_index=sim_edge_index,
        sim_edge_weight=sim_edge_weight,
        dist_edge_index=dist_edge_index,
        dist_edge_weight=dist_edge_weight,
        common_edge_index=common_edge_index,
        common_edge_weight=common_edge_weight
    )



def preprocess_HLN(adata_RNA, adata_ADT):
    """
    Preprocess the HLN dataset: load RNA/ADT data, filter, normalize, scale, and extract features.
    Args:
        rna_path (str): Path to RNA h5ad file
        adt_path (str): Path to ADT h5ad file
        gt_path (str): Path to ground truth labels (txt file)
    Returns:
        RNA_expression (np.ndarray): Processed RNA expression matrix
        ADT_expression (np.ndarray): Processed ADT expression matrix
        cell_positions (np.ndarray): Spatial coordinates of cells
        true_labels (List[int]): Ground truth cluster labels
    """

    sc.pp.filter_genes(adata_RNA, min_cells=10)
    sc.pp.highly_variable_genes(adata_RNA, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_RNA, target_sum=1e4)
    sc.pp.log1p(adata_RNA)
    sc.pp.scale(adata_RNA)

    adata_ADT = Protein(adata_ADT)

    RNA_expression = adata_RNA[:, adata_RNA.var['highly_variable']].X
    ADT_expression = adata_ADT.X

    return RNA_expression, ADT_expression

