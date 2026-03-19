import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DualGCN(nn.Module):
    """
    A dual-stream Graph Convolutional Network for multimodal representation learning.

    Args:
        in_channels (int): Number of input features for the RNA modality.
        hidden_channels (int): Number of hidden units in GCN layers.
        out_channels (int): Number of output embedding dimensions.
        q (int): Dimension of ADT modality.
        dropout (float): Dropout rate.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, q, dropout=0.5):
        super(DualGCN, self).__init__()

        # RNA stream: similarity-based and distance-based GCN branches
        self.x_RNA1 = GCNConv(in_channels, hidden_channels)
        self.x_RNA2 = GCNConv(in_channels, hidden_channels)

        # ADT stream: initial embedding
        self.protein3 = GCNConv(q, out_channels)

        # Project RNA branches to embedding space
        self.sim_conv = GCNConv(hidden_channels, out_channels)
        self.dist_conv = GCNConv(hidden_channels, out_channels)

        # Optional projection from RNA to ADT space
        self.protein = GCNConv(hidden_channels, out_channels)

        # Fusion layers
        self.fusion_layer1 = nn.Sequential(nn.Linear(2 * out_channels, out_channels))
        self.fusion_layer2 = nn.Sequential(nn.Linear(2 * out_channels, out_channels))

        # Decoder layers for reconstruction
        self.dropout = dropout
        self.deconv1 = nn.Linear(out_channels, hidden_channels)
        self.deconv2 = nn.Linear(hidden_channels, in_channels)
        self.deconv4 = nn.Linear(hidden_channels, q)
        self.deconv5 = nn.Linear(hidden_channels, q + in_channels)

    def forward(self, x_RNA, x_ADT, sim_edge_index, sim_edge_weight,
                dist_edge_index, dist_edge_weight, common_edge_index, common_edge_weight):
        """
        Forward pass for dual GCN.
        Returns multiple embeddings: sim_z, dist_z, fused_z, fused_with_adt_z, adt_z
        """
        xs = F.relu(self.x_RNA1(x_RNA, sim_edge_index, sim_edge_weight))
        xs = F.dropout(xs, self.dropout, training=self.training)

        xd = F.relu(self.x_RNA2(x_RNA, dist_edge_index, dist_edge_weight))
        xd = F.dropout(xd, self.dropout, training=self.training)

        x_sim = self.sim_conv(xs, sim_edge_index, sim_edge_weight)
        x_dist = self.dist_conv(xd, dist_edge_index, dist_edge_weight)
        pro = self.protein3(x_ADT, common_edge_index, common_edge_weight)

        combined = torch.cat([x_sim, x_dist], dim=1)
        fused = self.fusion_layer1(combined)

        combined_protein = torch.cat([fused, pro], dim=1)
        fused_pro = self.fusion_layer2(combined_protein)

        return x_sim, x_dist, fused, fused_pro, pro

    # RNA reconstruction
    def reconstruct(self, z):
        x_recon = F.relu(self.deconv1(z))
        return self.deconv2(x_recon)

    # ADT reconstruction
    def reconstruct2(self, z):
        x_recon = F.relu(self.deconv1(z))
        return self.deconv4(x_recon)

    # Joint RNA+ADT reconstruction
    def reconstruct3(self, z):
        x_recon = F.relu(self.deconv1(z))
        return self.deconv5(x_recon)

class Dual(nn.Module):
    """
    Dual Spatially-regularized Deep Multimodal Clustering with Consistency (DualSDMCC).

    Args:
        in_channels (int): Dimension of RNA input features.
        hidden_channels (int): Hidden layer dimension in GCN.
        out_channels (int): Output embedding size.
        q (int): Number of ADT features.
        num_clusters (int): Number of target clusters.
        beta (float): Weight for reconstruction loss.
        gamma (float): [Not used here] placeholder for other potential losses.
        delta (float): [Not used here] placeholder for additional losses.
        dropout (float): Dropout rate.
        l1_lambda (float): L1 regularization weight.
        l2_lambda (float): L2 regularization weight.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, q, num_clusters,
                 beta, gamma, delta, dropout,
                 l1_lambda=1e-4, l2_lambda=1e-3):
        super(Dual, self).__init__()
        self.gcn = DualGCN(in_channels, hidden_channels, out_channels, q, dropout)
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

    def forward(self, x_RNA, x_ADT, sim_edge_index, sim_edge_weight,
                dist_edge_index, dist_edge_weight, common_edge_index, common_edge_weight):
        return self.gcn(x_RNA, x_ADT, sim_edge_index, sim_edge_weight,
                        dist_edge_index, dist_edge_weight, common_edge_index, common_edge_weight)

    def target_distribution(self, q):
        """
        Soft clustering target distribution for deep clustering loss.
        """
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def compute_regularization_loss(self):
        l1_loss = sum(torch.sum(torch.abs(p)) for p in self.parameters())
        l2_loss = sum(torch.sum(p ** 2) for p in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss

    def compute_losses(self, x_RNA, x_ADT, sim_z, dist_z, fused_z, fused_pro, combined_raw, pro):
        """
        Compute total loss including reconstruction and spatial regularization.
        """
        l_rec = F.mse_loss(combined_raw, self.gcn.reconstruct3(fused_pro))
        l_sim = F.mse_loss(x_RNA, self.gcn.reconstruct(sim_z))
        l_dist = F.mse_loss(x_RNA, self.gcn.reconstruct(dist_z))
        l_adt = F.mse_loss(x_ADT, self.gcn.reconstruct2(pro))

        l_spatial = self.spatial_regularization_loss(fused_z, dist_edge_index=self.gcn_input.dist_edge_index,
                                                     dist_edge_weight=self.gcn_input.dist_edge_weight)

        reg_loss = self.compute_regularization_loss()
        total_loss = self.beta * (l_rec + l_sim + l_dist + l_adt) + self.gamma * l_spatial + self.delta * reg_loss

        return total_loss, l_rec

    def cosine_similarity(self, emb):
        mat = torch.matmul(emb, emb.T)
        norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
        mat = torch.div(mat, torch.matmul(norm, norm.T))
        mat = torch.where(torch.isnan(mat), torch.zeros_like(mat), mat)
        mat = mat - torch.diag_embed(torch.diag(mat))
        return mat

    def spatial_regularization_loss(self, emb, dist_edge_index, dist_edge_weight):
        """
        Spatial regularization to encourage local smoothness and contrastive separation.
        """
        num_nodes = emb.size(0)
        graph_nei = torch.sparse_coo_tensor(dist_edge_index, torch.ones_like(dist_edge_weight),
                                            size=(num_nodes, num_nodes)).to_dense()
        graph_neg = 1 - graph_nei
        sim_mat = torch.sigmoid(self.cosine_similarity(emb))

        neigh_loss = torch.mul(graph_nei, torch.log(sim_mat + 1e-10)).mean()
        neg_loss = torch.mul(graph_neg, torch.log(1 - sim_mat + 1e-10)).mean()

        return -(neigh_loss + neg_loss) / 2
