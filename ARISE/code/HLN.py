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

seed = 888
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed_all(seed)  # gpus
torch.backends.cudnn.deterministic = True
warnings.simplefilter(action='ignore', category=FutureWarning)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time

start_time = time.time()

class DualGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,q, dropout=0.5):
        super(DualGCN, self).__init__()
        self.x_RNA1 = GCNConv(in_channels, hidden_channels)
        self.x_RNA2 = GCNConv(in_channels, hidden_channels )
        self.protein3 = GCNConv(q, out_channels)

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

        self.deconv4 = nn.Linear(hidden_channels, q )
        self.deconv5 = nn.Linear(hidden_channels, q + in_channels )


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
    def __init__(self, in_channels, hidden_channels, out_channels,q, num_clusters,
                 beta, gamma, delta, dropout,
                 l1_lambda=1e-5, l2_lambda=1e-4):
        super(DualSDMCC, self).__init__()
        self.gcn = DualGCN(in_channels, hidden_channels, out_channels, q,dropout=dropout)
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

        total_loss = (self.beta * (l_reconstruction +l_reconstruction_sim+l_reconstruction_dist+l_reconstruction_adt)  + 10 * l_spatial_reg +reg_loss)

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

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def Protein(adata):
    # adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    # sc.pp.scale(adata, zero_center=False, max_value=10)
    adata_omics2 = clr_normalize_each_cell(adata)
    sc.pp.scale(adata_omics2)
    # adata.obsm['feat'] = pca(adata_omics2, n_comps=3000)
    return adata



adata_ADT = sc.read_h5ad('your_path')
adata_RNA = sc.read_h5ad('your_path')
metadata_file = os.path.join('your_path')
with open(metadata_file, 'r') as file:
    content = file.readlines()
true_labels = list(map(int, [line.strip() for line in content]))
print(len(true_labels))
print(adata_RNA.shape)
print(adata_ADT.shape)

sc.pp.filter_genes(adata_RNA, min_cells=10)
sc.pp.highly_variable_genes(adata_RNA, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_RNA, target_sum=1e4)
sc.pp.log1p(adata_RNA)
sc.pp.scale(adata_RNA)

adata_ADT = Protein(adata_ADT)

ADT_expression =adata_ADT.X
RNA_expression =adata_RNA[:, adata_RNA.var['highly_variable']].X
q = ADT_expression.shape[1]

cell_positions = adata_ADT.obsm['spatial']


similarity_matrix = cosine_similarity(RNA_expression)

num_neighbors = 15
nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='cosine').fit(RNA_expression)
distances, indices = nbrs.kneighbors(RNA_expression)

adjacency_matrix = np.zeros_like(similarity_matrix, dtype=int)
for i in range(RNA_expression.shape[0]):
    for j in indices[i][1:]:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1


sim_edge_index = torch.tensor(np.array(np.nonzero(adjacency_matrix)), dtype=torch.long).to(device)
sim_edge_weight = torch.tensor(similarity_matrix[adjacency_matrix > 0], dtype=torch.float).to(device)

x = cdist(cell_positions, cell_positions, metric='euclidean')

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
out_channels = 64

num_clusters = len(set(true_labels))
print(num_clusters)
print(num_clusters)

#
learning_rate = 0.001
num_epochs = 350

model = DualSDMCC(
    in_channels=num_features,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    q =q,
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
best_epoch = 0
scl=[]
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
