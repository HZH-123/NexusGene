import numpy as np
import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
#from UNGSL_test import data_loader
torch.manual_seed(42)
np.random.seed(42)
def create_cluster(data, k=4):
    # 确保 train_mask 是 PyTorch Tensor
    if isinstance(data.train_mask, np.ndarray):
        data.train_mask = torch.from_numpy(data.train_mask).bool()
    elif not torch.is_tensor(data.train_mask):
        raise TypeError("train_mask must be a tensor or numpy array")

    # 获取训练节点的特征（仅训练节点用于聚类）
    train_mask = data.train_mask  # 现在是 Tensor
    embedding = data.x[train_mask].detach().cpu().numpy()  # [n_train, dim]
    kmeans = KMeans(n_clusters=k, random_state=0)
    k_labels = kmeans.fit_predict(embedding)  # shape: [n_train]
    dataset = []
    train_indices = torch.where(train_mask)[0]  # ✅ 现在 train_mask 是 Tensor，没问题！
    for exclude in range(k):
        # mask: 在训练节点中，不属于 cluster `exclude` 的节点
        mask = (k_labels != exclude)  # numpy array of bool, length = n_train
        mask = torch.from_numpy(mask)  # 转为 Tensor
        # 从训练节点中选出保留的节点（全局索引）
        sub_nodes_global = train_indices[mask]  # 全局节点 ID（来自全图）
        # 构建子图（relabel_nodes=True → 局部 ID）
        sub_edge_index, _ = subgraph(sub_nodes_global, data.edge_index, relabel_nodes=True)
        # 构造子图 Data 对象（使用局部特征和标签）
        sub_data = Data(
            x=data.x[sub_nodes_global],
            edge_index=sub_edge_index,
            y=data.y[sub_nodes_global]
        )
        sub_data.orig_node_idx = sub_nodes_global  # 👈 保存全局 ID！
        dataset.append(sub_data)

    return dataset

# def create_cluster(data, k=7):
#     # 类转导学习：保留全量节点和边，仅按聚类划分标签
#     all_nodes = torch.arange(data.num_nodes, device=data.x.device)  # 所有节点可见
#     embedding = data.x.detach().cpu().numpy()  # 用所有节点的特征做聚类（而非仅训练节点）
#
#     # 聚类所有节点（包括训练和测试节点）
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     k_labels = kmeans.fit_predict(embedding)  # 每个节点的聚类标签（0~k-1）
#
#     dataset = []
#     for cluster_id in range(k):
#         # 子图训练标签：排除当前聚类簇（用其他簇的标签训练）
#         train_mask = torch.from_numpy(k_labels != cluster_id).to(data.x.device)
#         # 子图验证标签：仅用当前聚类簇（模拟未标注节点）
#         test_mask = torch.from_numpy(k_labels == cluster_id).to(data.x.device)
#
#         # 构造包含全量节点和边的子图（仅标签划分不同）
#         sub_data = Data(
#             x=data.x,  # 全量节点特征（类转导：所有节点可见）
#             edge_index=data.edge_index,  # 全量边结构（类转导：所有边可见）
#             y=data.y,  # 全量标签（但训练时仅用train_mask部分）
#             train_mask=train_mask,
#             test_mask=test_mask
#         )
#         dataset.append(sub_data)
#     return dataset