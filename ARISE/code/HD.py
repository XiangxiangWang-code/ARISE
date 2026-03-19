import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import warnings
import scanpy as sc
from sklearn.metrics import silhouette_score, davies_bouldin_score

warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class DualGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(DualGCN, self).__init__()
        self.shared_conv1 = GCNConv(in_channels, hidden_channels)
        self.shared_conv2 = GCNConv(in_channels, hidden_channels )
        self.sim_conv = GCNConv(hidden_channels , out_channels)
        self.dist_conv = GCNConv(hidden_channels , out_channels)

        # self.conv = GCNConv(out_channels*2, out_channels)
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels)
        )

        self.dropout = dropout
        self.deconv1 = nn.Linear(out_channels, hidden_channels)
        self.deconv2 = nn.Linear(hidden_channels, in_channels)

    def forward(self, x, sim_edge_index, sim_edge_weight, dist_edge_index, dist_edge_weight):
        xs = F.relu(self.shared_conv1(x, sim_edge_index, sim_edge_weight))
        xs= F.dropout(xs, self.dropout, training=self.training)

        xd = F.relu(self.shared_conv2(x, dist_edge_index, dist_edge_weight))
        xd = F.dropout(xd, self.dropout, training=self.training)

        x_sim = self.sim_conv(xs, sim_edge_index, sim_edge_weight)
        x_dist = self.dist_conv(xd, dist_edge_index, dist_edge_weight)


        combined = torch.cat([x_sim, x_dist], dim=1)
        fused = self.fusion_layer(combined)

        return x_sim, x_dist, fused

    def reconstruct(self, z):
        x_recon = F.relu(self.deconv1(z))
        x_recon = self.deconv2(x_recon)
        return x_recon


class DualSDMCC(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_clusters,
                 sim_edge_index, sim_edge_weight,
                 dist_edge_index, dist_edge_weight,
                 beta, gamma, delta, dropout,
                 l1_lambda=1e-5, l2_lambda=1e-4):
        super(DualSDMCC, self).__init__()
        self.gcn = DualGCN(in_channels, hidden_channels, out_channels, dropout=dropout)
        self.num_clusters = num_clusters
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda


    def forward(self, x, sim_edge_index, sim_edge_weight, dist_edge_index, dist_edge_weight):
        sim_z, dist_z, fused_z = self.gcn(x, sim_edge_index, sim_edge_weight, dist_edge_index, dist_edge_weight)
        return sim_z, dist_z, fused_z

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

    def compute_losses(self, x, sim_z, dist_z, fused_z):
        l_reconstruction = F.mse_loss(x, self.gcn.reconstruct(fused_z))
        l_reconstruction_sim = F.mse_loss(x, self.gcn.reconstruct(sim_z))
        l_reconstruction_dist = F.mse_loss(x, self.gcn.reconstruct(dist_z))

        l_embed_diff_sim = F.mse_loss(sim_z, fused_z)
        l_embed_diff_dist = F.mse_loss(dist_z, fused_z)

        l_spatial_reg = self.spatial_regularization_loss(fused_z, data.sim_edge_index, data.sim_edge_weight)


        total_loss = (self.beta * (l_reconstruction + l_reconstruction_sim + l_reconstruction_dist) +
                      self.gamma * (l_embed_diff_sim + l_embed_diff_dist) + 10 * l_spatial_reg )

        return total_loss, l_reconstruction, l_embed_diff_sim, l_embed_diff_dist

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


# Model evaluation function
def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        sim_z, dist_z, fused_z = model(data.x, data.sim_edge_index, data.sim_edge_weight,
                                       data.dist_edge_index, data.dist_edge_weight)
    return fused_z.cpu().numpy(), sim_z.cpu().numpy(), dist_z.cpu().numpy()

def evaluate_silhouette_score(embeddings, num_clusters):
    labels = cluster_embeddings(embeddings, num_clusters)
    silhouette_avg = silhouette_score(embeddings, labels)
    return silhouette_avg

def evaluate_davies_bouldin_index(embeddings, num_clusters):
    labels = cluster_embeddings(embeddings, num_clusters)
    dbi = davies_bouldin_score(embeddings, labels)
    return dbi


def cluster_embeddings(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    return kmeans.fit_predict(embeddings)



current_dir = os.path.dirname(__file__)
adata = sc.read_h5ad('your_path')
print(adata)


def adata_preprocess_1(adata, min_cells=100, pca_n_comps=2000, HVG=3000):
    print('===== 1 - Preprocessing Data ')
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_counts=3)
    print(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=HVG)

    sc.pp.normalize_total(adata, target_sum=1e4)

    sc.pp.log1p(adata)

    sc.pp.scale(adata, zero_center=False, max_value=10)

    return adata[:, adata.var['highly_variable']].X


gene_expression = adata_preprocess_1(adata, min_cells=100,pca_n_comps=2000,  HVG=3000)
print(gene_expression.shape)

similarity_matrix = cosine_similarity(gene_expression)


adjacency_matrix = (similarity_matrix > 0.9).astype(int)
sim_edge_index = torch.tensor(np.array(np.nonzero(adjacency_matrix)), dtype=torch.long).to(device)
sim_edge_weight = torch.tensor(similarity_matrix[adjacency_matrix > 0], dtype=torch.float).to(device)

k = 15
cell_positions = adata.obsm['spatial']

distance_matrix = cdist(cell_positions, cell_positions, metric='euclidean')

nearest_k_indices = np.argsort(distance_matrix, axis=1)[:, 1:k + 1]

# Construct adjacency matrix
distance_adj = np.zeros_like(distance_matrix)
for i in range(len(cell_positions)):
    distance_adj[i, nearest_k_indices[i]] = 1
np.fill_diagonal(distance_adj, 0)

dist_edge_index = torch.tensor(np.array(np.nonzero(distance_adj)), dtype=torch.long).to(device)
dist_edge_weight = torch.tensor(1 / (distance_matrix[distance_adj > 0] + 1e-8), dtype=torch.float).to(device)

num_nodes, num_features = gene_expression.shape
x = torch.tensor(gene_expression, dtype=torch.float).to(device)

ari_list = []
nmi_list = []

# Create dual-graph data object
class DualGraphData(Data):
    def __init__(self, x, sim_edge_index, sim_edge_weight, dist_edge_index, dist_edge_weight):
        super().__init__()
        self.x = x
        self.sim_edge_index = sim_edge_index
        self.sim_edge_weight = sim_edge_weight
        self.dist_edge_index = dist_edge_index
        self.dist_edge_weight = dist_edge_weight


data = DualGraphData(
    x=x,
    sim_edge_index=sim_edge_index,
    sim_edge_weight=sim_edge_weight,
    dist_edge_index=dist_edge_index,
    dist_edge_weight=dist_edge_weight
)

# Define hyperparameters
hidden_channels = 512
out_channels = 256
num_clusters = 10
learning_rate = 0.001
num_epochs = 500

model = DualSDMCC(
    in_channels=num_features,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    num_clusters=num_clusters,
    sim_edge_index=sim_edge_index,
    sim_edge_weight=sim_edge_weight,
    dist_edge_index=dist_edge_index,
    dist_edge_weight=dist_edge_weight,
    beta=3.5,
    gamma=0.5,
    delta=0,
    dropout=0.5,
    l1_lambda=1e-4,
    l2_lambda=1e-3
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
best_dbi = 4
best_labels = None

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    sim_z, dist_z, fused_z = model(data.x, data.sim_edge_index, data.sim_edge_weight,
                                   data.dist_edge_index, data.dist_edge_weight)

    loss, l_reconstruction, l_embed_diff_sim, l_embed_diff_dist= model.compute_losses(data.x, sim_z, dist_z, fused_z)

    loss.backward()

    torch.cuda.empty_cache()

    optimizer.step()

    if epoch % 10 == 0:
        embeddings, _, _ = evaluate_model(model, data)
        silhouette_avg = evaluate_silhouette_score(embeddings, num_clusters)
        dbi = evaluate_davies_bouldin_index(embeddings, num_clusters)
        print(f"Epoch {epoch} - sc: {silhouette_avg}, db: {dbi}")
        ari_list.append(silhouette_avg)
        nmi_list.append(dbi)

        if silhouette_avg > best_silhouette_avg:
            best_dbi = dbi
            best_silhouette_avg = silhouette_avg
            best_embeddings = embeddings
            best_labels = cluster_embeddings(embeddings, num_clusters)

print(f"Best sc: {best_silhouette_avg}")
print(f"Best db: {best_dbi}")


