# run_HLN.py
import argparse
import os
import torch
from process import normalize, Protein, build_dual_graph,preprocess_HLN
from train import initialize_model, train_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--beta', type=float, default=25)
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=350)
    return parser.parse_args()


def main():
    import scanpy as sc
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    current_dir = os.path.dirname(__file__)
    adata_ADT = sc.read_h5ad('./data/HLN/adata_ADT.h5ad')
    adata_RNA = sc.read_h5ad('./data/HLN/adata_RNA.h5ad')
    metadata_file = os.path.join(current_dir, './data/HLN/GT_labels.txt')
    with open(metadata_file, 'r') as file:
        content = file.readlines()
    true_labels = list(map(int, [line.strip() for line in content]))
    RNA_data , ADT_data = preprocess_HLN(adata_RNA,adata_ADT)


    cell_positions = adata_RNA.obsm['spatial']
    graph_data = build_dual_graph(RNA_data, ADT_data, cell_positions, device=device)


    model = initialize_model(graph_data, RNA_data.shape[1], ADT_data.shape[1], args)
    model,best_embeddings,best_labels = train_model(model, graph_data, args , true_labels)
    import scanpy as sc
    import matplotlib.pyplot as plt
    import pandas as pd
    # 确保 adata_RNA 中有空间坐标
    assert 'spatial' in adata_RNA.obsm.keys(), "Spatial coordinates not found in adata_RNA.obsm['spatial']"

    # 将聚类结果添加到 adata_RNA 的 obs 中
    adata_RNA.obs['cluster'] = pd.Categorical(best_labels)

    # 设置图形大小
    plt.figure(figsize=(8, 8))

    # 使用 scanpy 的 spatial 绘图
    sc.pl.spatial(adata_RNA, color='cluster', size=1.0,spot_size=1.2, img_key=None, show=False)

    # 显示图像
    plt.show()

if __name__ == '__main__':
    main()
