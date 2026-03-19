# train.py
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
from model import Dual
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def evaluate_model(model, data):
    """
    Run the model in evaluation mode and return fused predictions and embeddings.

    Args:
        model (Dual): The trained model.
        data (DualGraphData): Graph data with features and edges.

    Returns:
            - fused_pro: fused probability (output of final prediction layer),
            - sim_z: RNA similarity graph embedding,
            - dist_z: spatial distance graph embedding.
    """
    model.eval()
    with torch.no_grad():
        sim_z, dist_z, fused_z ,fused_pro,pro = model(data.x_RNA, data.x_ADT, data.sim_edge_index, data.sim_edge_weight ,data.dist_edge_index, data.dist_edge_weight, data.common_edge_index , data.common_edge_weight)
    return fused_pro.cpu().numpy(), sim_z.cpu().numpy(), dist_z.cpu().numpy()

def cluster_embeddings(embeddings, num_clusters):
    """
    Perform KMeans clustering on the embeddings.

    Args:
        embeddings (np.ndarray): Feature embeddings.
        num_clusters (int): Number of clusters.

    Returns:
         Predicted cluster labels.
    """
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    return kmeans.fit_predict(embeddings)

def evaluate_model_performance(predicted_labels, true_labels):
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return ari, nmi


def initialize_model(data, input_dim, adt_dim, args):
    """
       Initialize the Dual model with hyperparameters.

       Args:
           data (DualGraphData): Input graph data.
           input_dim (int): Input dimension of RNA features.
           adt_dim (int): Input dimension of ADT features.
           args (Namespace): Hyperparameters and settings.

       Returns:
           Dual: Initialized model instance.
       """
    model = Dual(
        in_channels=input_dim,
        hidden_channels=args.hidden_dim,
        out_channels=args.out_dim,
        q=adt_dim,
        num_clusters=args.num_clusters,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        dropout=args.dropout
    ).to(args.device)

    return model


def train_model(model, data, args ,true_labels):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    combined_raw = torch.cat([data.x_RNA, data.x_ADT], dim=1)
    ari_list = []
    nmi_list = []
    best_ari = 0

    for epoch in range(args.epochs):
        optimizer.zero_grad()

        sim_z, dist_z, fused_z, fused_pro, pro = model(
            data.x_RNA, data.x_ADT,
            data.sim_edge_index, data.sim_edge_weight,
            data.dist_edge_index, data.dist_edge_weight,
            data.common_edge_index, data.common_edge_weight
        )


        model.gcn_input = data  # for spatial regularization

        loss, l_rec = model.compute_losses(
            data.x_RNA, data.x_ADT,
            sim_z, dist_z, fused_z, fused_pro, combined_raw, pro
        )

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}, Total Loss: {loss.item():.4f}, Recon Loss: {l_rec.item():.4f}")

        # Evaluate clustering performance
        embeddings, _, _ = evaluate_model(model, data)

        predicted_labels = cluster_embeddings(embeddings, args.num_clusters)
        ari, nmi = evaluate_model_performance(predicted_labels, true_labels)
        # sc = evaluate_silhouette_score(embeddings,predicted_labels)
        # scl.append(sc)
        print(f"Epoch {epoch} - ARI: {ari:.4f}, NMI: {nmi:.4f}")

        if ari > best_ari:
            best_ari = ari
            best_nmi = nmi
            best_embeddings = embeddings
            best_labels = predicted_labels

        ari_list.append(ari)
        nmi_list.append(nmi)
    print("ari : " , best_ari)
    print("nmi : " , best_nmi)
    return model,best_embeddings,best_labels
