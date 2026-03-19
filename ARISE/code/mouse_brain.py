import os
import random

import numpy as np
import torch
import scanpy as sc
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import warnings
from sklearn.preprocessing import quantile_transform

warnings.simplefilter(action='ignore', category=FutureWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 888
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed_all(seed)  # gpus
torch.backends.cudnn.deterministic = True
import time

start_time = time.time()

class DualGCN(nn.Module):
    # === GCN Encoder with dual view and fusion layers ===
    def __init__(self, in_channels, hidden_channels, out_channels,protein, dropout=0.5):
        super(DualGCN, self).__init__()
        self.x_RNA1 = GCNConv(in_channels, hidden_channels)
        self.x_RNA2 = GCNConv(in_channels, hidden_channels )
        self.protein3 = GCNConv(protein, out_channels)

        self.sim_conv = GCNConv(hidden_channels , out_channels)
        self.dist_conv = GCNConv(hidden_channels , out_channels)
        self.protein = GCNConv(hidden_channels, out_channels)

        # self.conv = GCNConv(out_channels*2, out_channels)
        self.fusion_layer1 = nn.Sequential(
            nn.Linear(2 * out_channels , out_channels)
        )
        self.fusion_layer2 = nn.Sequential(
            nn.Linear( 2*out_channels , out_channels)
        )

        self.dropout = dropout
        self.deconv1 = nn.Linear(out_channels, hidden_channels)
        self.deconv2 = nn.Linear(hidden_channels, in_channels )

        self.deconv4 = nn.Linear(hidden_channels, protein )
        self.deconv5 = nn.Linear(hidden_channels, protein + in_channels )


    def forward(self, x_RNA, x_ADT, sim_edge_index, sim_edge_weight, dist_edge_index, dist_edge_weight, common_edge_index , common_edge_weight):
        xs = F.relu(self.x_RNA1(x_RNA, sim_edge_index, sim_edge_weight))
        xs= F.dropout(xs, self.dropout, training=self.training)

        xd = F.relu(self.x_RNA2(x_RNA, dist_edge_index, dist_edge_weight))
        xd = F.dropout(xd, self.dropout, training=self.training)

        x_sim = self.sim_conv(xs, sim_edge_index, sim_edge_weight)
        x_dist = self.dist_conv(xd, dist_edge_index, dist_edge_weight)

        pro = self.protein3(x_ADT, common_edge_index, common_edge_weight)


        combined = torch.cat([x_sim, x_dist], dim=1)
        fused = self.fusion_layer1(combined)

        combined = torch.cat([fused, pro], dim=1)
        fused_pro = self.fusion_layer2(combined)
        return x_sim, x_dist, fused, fused_pro,pro

    def reconstruct(self, z):
        x_recon = F.relu(self.deconv1(z))
        x_recon = self.deconv2(x_recon)
        return x_recon

    def reconstruct2(self, z):
        x_recon = F.relu(self.deconv1(z))
        x_recon = self.deconv4(x_recon)
        return x_recon

    def reconstruct3(self, z):
        x_recon = F.relu(self.deconv1(z))
        x_recon = self.deconv5(x_recon)
        return x_recon


class DualSDMCC(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,protein, num_clusters,
                 beta, gamma, delta, dropout,
                 l1_lambda=1e-5, l2_lambda=1e-4):
        super(DualSDMCC, self).__init__()
        self.gcn = DualGCN(in_channels, hidden_channels, out_channels, protein,dropout=dropout)
        self.cluster_layer = nn.Parameter(torch.Tensor(num_clusters, out_channels))
        self.num_clusters = num_clusters
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.cluster_layer.data)

    def forward(self, x_RNA, x_ADT, sim_edge_index, sim_edge_weight, dist_edge_index, dist_edge_weight, common_edge_index , common_edge_weight):
        sim_z, dist_z, fused_z , fused_pro , pro= self.gcn(x_RNA, x_ADT, sim_edge_index, sim_edge_weight, dist_edge_index, dist_edge_weight, common_edge_index , common_edge_weight)
        return sim_z, dist_z, fused_z, fused_pro,pro

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def compute_regularization_loss(self):
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
            l2_loss += torch.sum(param.pow(2))
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss

    def compute_losses(self, x_RNA, x_ADT, sim_z, dist_z, fused_z ,fused_pro, combined_raw ,pro):
        l_reconstruction = F.mse_loss(combined_raw, self.gcn.reconstruct3(fused_pro))
        l_reconstruction_sim = F.mse_loss(x_RNA, self.gcn.reconstruct(sim_z))
        l_reconstruction_dist = F.mse_loss(x_RNA, self.gcn.reconstruct(dist_z))
        l_reconstruction_adt = F.mse_loss(x_ADT, self.gcn.reconstruct2(pro))

        l_spatial_reg = self.spatial_regularization_loss(fused_z, data.sim_edge_index, data.sim_edge_weight)

        reg_loss = self.compute_regularization_loss()

        total_loss = (2.5 * (l_reconstruction +l_reconstruction_sim+l_reconstruction_dist+l_reconstruction_adt)  + 10 * l_spatial_reg +reg_loss)

        return total_loss, l_reconstruction


    def cosine_similarity(self, emb):
        mat = torch.matmul(emb, emb.T)
        norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
        mat = torch.div(mat, torch.matmul(norm, norm.T))
        mat = torch.where(torch.isnan(mat), torch.zeros_like(mat), mat)
        mat = mat - torch.diag_embed(torch.diag(mat))
        return mat

    def spatial_regularization_loss(self, emb, dist_edge_index, dist_edge_weight):
        num_nodes = emb.size(0)
        graph_nei = torch.sparse_coo_tensor(
            dist_edge_index,
            torch.ones_like(dist_edge_weight),
            size=(num_nodes, num_nodes)
        ).to_dense()
        graph_neg = 1 - graph_nei

        mat = torch.sigmoid(self.cosine_similarity(emb))

        neigh_loss = torch.mul(graph_nei, torch.log(mat + 1e-10)).mean()
        neg_loss = torch.mul(graph_neg, torch.log(1 - mat + 1e-10)).mean()

        return -(neigh_loss + neg_loss) / 2

def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        sim_z, dist_z, fused_z ,fused_pro,pro = model(data.x_RNA, data.x_ADT, data.sim_edge_index, data.sim_edge_weight ,data.dist_edge_index, data.dist_edge_weight, data.common_edge_index , data.common_edge_weight)
    return fused_pro.cpu().numpy(), sim_z.cpu().numpy(), dist_z.cpu().numpy()

def evaluate_model_performance(predicted_labels, true_labels):

    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return ari, nmi

def cluster_embeddings(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    return kmeans.fit_predict(embeddings)
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_silhouette_score(embeddings, labels):
    silhouette_avg = silhouette_score(embeddings, labels)
    return silhouette_avg

def clr_normalize_each_cell(adata, inplace=True):
    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
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
    """Dimension reduction with PCA algorithm"""

    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)

    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca
def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)

    return adata

import anndata
import scipy
import sklearn
from typing import Optional
def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def lsi(
        adata: anndata.AnnData, n_components,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    print(n_components)
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    #X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]

current_dir = os.path.dirname(__file__)

adata_ATAC = sc.read_h5ad("your_path")
adata_RNA = sc.read_h5ad("your_path")

true_label = "your_path"
with open(true_label, 'r') as file:
    content = file.readlines()
true_labels = list(map(int, [line.strip() for line in content]))

sc.pp.filter_genes(adata_RNA, min_cells=10)
sc.pp.filter_cells(adata_RNA, min_genes=200)
sc.pp.highly_variable_genes(adata_RNA, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_RNA, target_sum=1e4)
sc.pp.log1p(adata_RNA)
sc.pp.scale(adata_RNA)
adata_omics1_high = adata_RNA[:, adata_RNA.var['highly_variable']]
adata_RNA.obsm['feat'] = pca(adata_omics1_high, n_comps=50)

adata_ATAC = adata_ATAC[adata_RNA.obs_names].copy()  # .obsm['X_lsi'] represents the dimension reduced feature
sc.pp.highly_variable_genes(adata_ATAC, flavor="seurat_v3", n_top_genes=3000)
lsi(adata_ATAC, use_highly_variable=False, n_components=51)
adata_ATAC.obsm['feat'] = adata_ATAC.obsm['X_lsi'].copy()


print(adata_ATAC.obsm['feat'].shape)
print(adata_RNA.obsm['feat'].shape)

ADT_expression =adata_ATAC.obsm['feat']
RNA_expression =adata_RNA.obsm['feat']
q = ADT_expression.shape[1]

cell_positions = adata_ATAC.obsm['spatial']
similarity_matrix = cosine_similarity(RNA_expression)

num_neighbors = 15
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

k = 15
include_self = False

knn_graph = kneighbors_graph(cell_positions, n_neighbors=k, mode='distance', include_self=include_self)
knn_graph = knn_graph.maximum(knn_graph.T)

dist_edge_index = torch.tensor(knn_graph.nonzero(), dtype=torch.long).to(device)
dist_edge_weight = torch.ones_like(torch.tensor(knn_graph.data, dtype=torch.float)).to(device)

sim_edges = set(zip(sim_edge_index[0].tolist(), sim_edge_index[1].tolist()))
dist_edges = set(zip(dist_edge_index[0].tolist(), dist_edge_index[1].tolist()))


common_edges = sim_edges.intersection(dist_edges)

common_edge_index = torch.tensor(list(zip(*common_edges)), dtype=torch.long).to(device)
common_edge_weight = torch.ones(common_edge_index.shape[1], dtype=torch.float).to(device)
#
num_nodes, num_features = RNA_expression.shape

x_RNA = torch.tensor(RNA_expression, dtype=torch.float).to(device)
x_ADT = torch.tensor(ADT_expression, dtype=torch.float).to(device)

class DualGraphData(Data):
    def __init__(self, x_RNA, x_ADT, sim_edge_index, sim_edge_weight,dist_edge_index, dist_edge_weight , common_edge_index,common_edge_weight):
        super().__init__()
        self.x_RNA = x_RNA
        self.x_ADT = x_ADT
        self.sim_edge_index = sim_edge_index
        self.sim_edge_weight = sim_edge_weight
        self.dist_edge_index = dist_edge_index
        self.dist_edge_weight = dist_edge_weight
        self.common_edge_index = common_edge_index
        self.common_edge_weight = common_edge_weight


data = DualGraphData(
    x_RNA=x_RNA,
    x_ADT=x_ADT,
    sim_edge_index=sim_edge_index,
    sim_edge_weight=sim_edge_weight,
    dist_edge_index=dist_edge_index,
    dist_edge_weight=dist_edge_weight,
    common_edge_index=common_edge_index,
    common_edge_weight=common_edge_weight,

)

ari_list = []
nmi_list = []

hidden_channels = 512
out_channels = 128

num_clusters = len(set(true_labels))
print(num_clusters)

learning_rate = 0.001
num_epochs = 350

model = DualSDMCC(
    in_channels=num_features,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    protein =q,
    num_clusters=num_clusters,
    beta=25,
    gamma=0.5,
    delta=0,
    dropout=0,
    l1_lambda=1e-4,
    l2_lambda=1e-3
).to(device)
#
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
best_ari = 0
best_nmi = 0
combined_raw = torch.cat([x_RNA, x_ADT], dim=1)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()


    sim_z, dist_z, fused_z,fused_pro,pro = model(data.x_RNA, data.x_ADT,
           data.sim_edge_index, data.sim_edge_weight, data.dist_edge_index, data.dist_edge_weight , data.common_edge_index , data.common_edge_weight)

    loss, l_reconstruction= model.compute_losses(data.x_RNA, data.x_ADT, sim_z, dist_z, fused_z , fused_pro,combined_raw , pro)

    loss.backward()
    optimizer.step()

    embeddings, _, _ = evaluate_model(model, data)
    predicted_labels = cluster_embeddings(embeddings, num_clusters)
    ari, nmi = evaluate_model_performance(predicted_labels, true_labels)

    print(f"Epoch {epoch} - ARI: {ari:.4f}, NMI: {nmi:.4f}")

    if ari > best_ari:
        best_ari = ari
        best_nmi = nmi
        best_embeddings = embeddings
        best_labels = predicted_labels
        best_epoch = epoch
    ari_list.append(ari)
    nmi_list.append(nmi)

print(f"best_ari: {best_ari:.4f}")
print(f"best_nmi: {best_nmi:.4f}")

