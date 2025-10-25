import numpy as np
import torch
from networkx import eigenvector_centrality

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, recall_score, accuracy_score
)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse, degree, coalesce
from UNGSL_test.cluster import create_cluster
from UNGSL_test.data_h5_loader import read_h5file
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# === 新增：4个组学数据拆分函数（仅作用于输入的节点子集）===
def SNV_split(node_subset, x_full):
    # 仅对node_subset中的节点，用SNV特征（前16列）划分
    x_np = x_full.cpu().numpy()
    # 提取子集节点的SNV特征
    x_snv_subset = x_np[node_subset, :16]
    row_mean = np.mean(x_snv_subset, axis=1)
    # 对子集节点按均值排序（索引对应子集内部）
    sorted_subset_idx = np.argsort(row_mean)
    split_point = int(sorted_subset_idx.shape[0] * 0.2)
    # 映射回原始节点ID：新训练集（80%）、新验证集（20%）
    new_train_idx = node_subset[sorted_subset_idx[split_point:]]
    new_val_idx = node_subset[sorted_subset_idx[:split_point]]
    return new_train_idx, new_val_idx


def GE_split(node_subset, x_full):
    # 仅对node_subset中的节点，用GE特征（32-48列）划分
    x_np = x_full.cpu().numpy()
    x_ge_subset = x_np[node_subset, 32:48]
    row_mean = np.mean(x_ge_subset, axis=1)
    sorted_subset_idx = np.argsort(row_mean)
    split_point = int(sorted_subset_idx.shape[0] * 0.2)
    new_train_idx = node_subset[sorted_subset_idx[split_point:]]
    new_val_idx = node_subset[sorted_subset_idx[:split_point]]
    return new_train_idx, new_val_idx


def METH_split(node_subset, x_full):
    # 仅对node_subset中的节点，用METH特征（16-32列）划分
    x_np = x_full.cpu().numpy()
    x_meth_subset = x_np[node_subset, 16:32]
    row_mean = np.mean(x_meth_subset, axis=1)
    sorted_subset_idx = np.argsort(row_mean)
    split_point = int(sorted_subset_idx.shape[0] * 0.2)
    new_train_idx = node_subset[sorted_subset_idx[split_point:]]
    new_val_idx = node_subset[sorted_subset_idx[:split_point]]
    return new_train_idx, new_val_idx


def CNA_split(node_subset, x_full):
    # 仅对node_subset中的节点，用CNA特征（32列后）划分
    x_np = x_full.cpu().numpy()
    x_cna_subset = x_np[node_subset, 48:]
    row_mean = np.mean(x_cna_subset, axis=1)
    sorted_subset_idx = np.argsort(row_mean)
    split_point = int(sorted_subset_idx.shape[0] * 0.2)
    new_train_idx = node_subset[sorted_subset_idx[split_point:]]
    new_val_idx = node_subset[sorted_subset_idx[:split_point]]
    return new_train_idx, new_val_idx


# === 设备与数据加载（完全保留原始数据集的train/val/test划分，后续仅调整训练集内部）===
primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = read_h5file("networks/LTG_multiomics.h5")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_np = data.x.cpu().numpy()
x_scaled = scaler.fit_transform(x_np)
data.x = torch.tensor(x_scaled, dtype=torch.float)
data_full = data.to(primary_device)
data_full.y = data_full.y.float()
dataset = create_cluster(data.cpu())

# 打印原始数据集基本信息（确认原始划分未被修改）
print("=== 原始数据集信息 ===")
for i, sub in enumerate(dataset):
    pos_ratio = sub.y.float().mean().item()
    print(f"[Subgraph {i}] Nodes: {sub.num_nodes}, Pos ratio: {pos_ratio:.4f}")
print(f"原始训练集节点数: {data_full.train_mask.sum().item()}")
print(f"原始验证集节点数: {data_full.val_mask.sum().item()}")
print(f"原始测试集节点数: {data_full.test_mask.sum().item()}")

config = {
    'gsl_hidden': 256,
    'cls_hidden': 128,
    'dropout': 0.5,
    'threshold': 0.01,
    'alpha': 0.8,
    'reg_weight': 1e-5,
    'topk': 10,
    'top_p': 0.1,
    'aff_weight': 0.1,
}


# === 模型定义（保持不变）===
class EnhancedGSL(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, hidden_channel),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channel, hidden_channel),
        )

    def forward(self, x, original_adj):
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=1)
        sim = torch.mm(x, x.t())
        adj = config['alpha'] * sim + (1 - config['alpha']) * original_adj
        return adj


class EnhancedClassifier(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.conv1 = SAGEConv(in_channel, hidden_channel)
        self.bn1 = nn.LayerNorm(hidden_channel)
        self.conv2 = SAGEConv(hidden_channel, hidden_channel)
        self.conv3 = SAGEConv(hidden_channel, hidden_channel)
        self.dropout = config['dropout']
        self.conv4 = SAGEConv(hidden_channel, out_channel)
        self.residual = nn.Linear(in_channel, hidden_channel) if in_channel != hidden_channel else None

    def forward(self, x, edge_index):
        identity = x
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        if self.residual is not None:
            x = x + self.residual(identity)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index)
        return x.squeeze(-1)


class UNGSLayer(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.conf_weight = nn.Parameter(torch.ones(num_nodes) * 0.5)

    def forward(self, s, confidence):
        conf_mat = (confidence.unsqueeze(0) + confidence.unsqueeze(1)) / 2
        adj = torch.sigmoid(s) * torch.sigmoid(conf_mat)
        adj = (adj + adj.t()) / 2
        return adj


class FullGraphGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1):
        super().__init__()
        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim // 2,
            heads=1,  # 减少头数以节省显存
            concat=False,
            dropout=0.3,
            add_self_loops=True
        )
        self.sage = SAGEConv(in_dim, hidden_dim // 2)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_weight=None):
        x1 = self.gat(x, edge_index)
        x2 = self.sage(x, edge_index, edge_weight)
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.classifier(x)
        return x.squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


def compute_confidence(logits):
    prob = torch.sigmoid(logits)
    entropy = - (prob * torch.log(prob + 1e-10) + (1 - prob) * torch.log(1 - prob + 1e-10))
    confidence = 1.0 - entropy
    return confidence.clamp(0.0, 1.0)


# === 子图训练函数（显存优化，保持不变）===
def train_model(subgraph):
    device = primary_device
    subgraph = subgraph.to(device)
    n = subgraph.num_nodes
    x = subgraph.x
    y = subgraph.y.float()
    edge_index = subgraph.edge_index
    x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    adj_dense = to_dense_adj(edge_index, max_num_nodes=n)[0].to(device)

    pretrain_gsl = EnhancedGSL(subgraph.num_features, config['gsl_hidden']).to(device)
    pretrain_cls = EnhancedClassifier(subgraph.num_features, config['cls_hidden'], 1).to(device)
    pretrain_optim = torch.optim.AdamW([
        {'params': pretrain_gsl.parameters(), 'lr': 1e-3, 'weight_decay': 1e-3},
        {'params': pretrain_cls.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4}
    ])
    criterion = nn.BCEWithLogitsLoss()

    def affinity_loss(S, y, margin=0.5):
        y = y.float()
        pos_mask = (y.unsqueeze(0) * y.unsqueeze(1)) > 0
        neg_mask = (y.unsqueeze(0) + y.unsqueeze(1)) == 0
        loss = 0.0
        if pos_mask.any():
            loss += F.relu(margin - S[pos_mask]).mean()
        if neg_mask.any():
            loss += F.relu(S[neg_mask] - margin).mean()
        return loss

    for epoch in range(1, 151):
        pretrain_gsl.train()
        pretrain_cls.train()
        pretrain_optim.zero_grad()
        S = pretrain_gsl(x, adj_dense)
        aff_loss = config['aff_weight'] * affinity_loss(S, y)
        edge_index_S = torch.nonzero(S > config['threshold']).t()
        logits = pretrain_cls(x, edge_index_S).squeeze(-1)
        cls_loss = criterion(logits, y)
        loss = cls_loss + aff_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pretrain_gsl.parameters(), max_norm=2.0)
        torch.nn.utils.clip_grad_norm_(pretrain_cls.parameters(), max_norm=2.0)
        pretrain_optim.step()

    model_gsl = EnhancedGSL(subgraph.num_features, config['gsl_hidden']).to(device)
    model_cls = EnhancedClassifier(subgraph.num_features, config['cls_hidden'], 1).to(device)
    model_ungsl = UNGSLayer(n).to(device)
    model_gsl.load_state_dict(pretrain_gsl.state_dict())
    model_cls.load_state_dict(pretrain_cls.state_dict())

    model_cls.requires_grad_(False)
    finetune_optim = torch.optim.AdamW([
        {'params': model_gsl.parameters(), 'lr': 1e-3},
        {'params': model_ungsl.parameters(), 'lr': 5e-3}
    ], weight_decay=1e-5)

    for epoch in range(1, 201):
        model_gsl.train()
        model_ungsl.train()
        if epoch == 100:
            model_cls.requires_grad_(True)
            finetune_optim = torch.optim.AdamW([
                {'params': model_gsl.parameters(), 'lr': 1e-4},
                {'params': model_ungsl.parameters(), 'lr': 5e-4},
                {'params': model_cls.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}
            ], weight_decay=1e-5)
        finetune_optim.zero_grad()
        S = model_gsl(x, adj_dense)
        with torch.no_grad():
            cls_out = model_cls(x, edge_index)
            confidence = compute_confidence(cls_out)
        S_hat = model_ungsl(S, confidence)
        edge_index_finetune = torch.nonzero(S_hat > config['threshold']).t()
        out = model_cls(x, edge_index_finetune).squeeze(-1)
        cls_loss = criterion(out, y)
        reg_loss = torch.norm(S_hat, p=1)
        reg_weight = config['reg_weight'] * (1 + epoch / 500)
        total_loss = cls_loss + reg_weight * reg_loss
        total_loss.backward()
        finetune_optim.step()

    # 显存释放
    orig_idx = subgraph.orig_node_idx.clone()
    del adj_dense, S, S_hat, edge_index_S, edge_index_finetune, subgraph
    torch.cuda.empty_cache()
    return model_gsl, model_cls, model_ungsl, orig_idx


# === Step 1: 训练所有子图模型（保持不变）===
models = []
for sub in dataset:
    print(f"\nTraining subgraph {len(models) + 1}/{len(dataset)}...")
    models.append(train_model(sub))

# === Step 2: 结构集成（稀疏化，保持不变）===
N = data_full.num_nodes
all_edges = []
all_weights = []

for i, (model_gsl, model_cls, model_ungsl, orig_idx) in enumerate(models):
    print(f"\nProcessing subgraph {i + 1}/{len(models)} for structure ensemble...")
    model_gsl.eval()
    model_ungsl.eval()
    model_cls.eval()
    subgraph = dataset[i].to(primary_device)
    sub_nodes = orig_idx.to(primary_device)
    n_sub = sub_nodes.size(0)
    global_to_local = torch.full((N,), -1, dtype=torch.long, device=primary_device)
    global_to_local[sub_nodes] = torch.arange(n_sub, device=primary_device)
    edge_index_global = subgraph.edge_index.to(primary_device)
    edge_index_local = global_to_local[edge_index_global]
    edge_index_local = edge_index_local[:, (edge_index_local[0] >= 0) & (edge_index_local[1] >= 0)]
    adj_sub = to_dense_adj(edge_index_local, max_num_nodes=n_sub)[0]
    x_sub = subgraph.x.to(primary_device)
    with torch.no_grad():
        S_sub = model_gsl(x_sub, adj_sub)
        cls_out_sub = model_cls(x_sub, edge_index_local).squeeze(-1)
        confidence_sub = compute_confidence(cls_out_sub)
        S_hat_sub = model_ungsl(S_sub, confidence_sub)

        triu_mask = torch.triu(torch.ones_like(S_hat_sub), diagonal=1).bool()
        values = S_hat_sub[triu_mask]
        if values.numel() > 0:
            k = max(5, int(config['top_p'] * values.numel()))
            if k < values.numel():
                threshold_val = torch.kthvalue(values, values.numel() - k + 1).values
            else:
                threshold_val = values.min()
            mask = S_hat_sub >= threshold_val
            mask = mask | mask.t()
        else:
            mask = torch.zeros_like(S_hat_sub, dtype=torch.bool)

        rows, cols = torch.nonzero(mask, as_tuple=True)
        if rows.numel() > 0:
            global_rows = sub_nodes[rows]
            global_cols = sub_nodes[cols]
            weights = S_hat_sub[rows, cols]
            all_edges.append(torch.stack([global_rows, global_cols], dim=0))
            all_weights.append(weights)

    # 显存清理
    del S_sub, S_hat_sub, mask, adj_sub, x_sub, subgraph
    torch.cuda.empty_cache()

# 合并所有边（去重 + 权重聚合）
if all_edges:
    edge_index_ensemble = torch.cat(all_edges, dim=1)
    edge_weight_ensemble = torch.cat(all_weights, dim=0)
    # 去重：保留最大权重
    edge_index_ensemble, inverse = torch.unique(edge_index_ensemble, dim=1, return_inverse=True)
    final_weights = torch.zeros(edge_index_ensemble.size(1), device=primary_device)
    final_weights.scatter_reduce_(0, inverse, edge_weight_ensemble, reduce="max", include_self=False)
else:
    edge_index_ensemble = torch.empty((2, 0), dtype=torch.long, device=primary_device)
    final_weights = torch.empty(0, device=primary_device)

print(f"\n✅ Ensemble graph has {edge_index_ensemble.size(1)} edges.")


# === 核心：评估函数（支持传入“训练集内部划分的新mask”，完全保留原始测试集）===
def train_and_evaluate(edge_index, edge_weight, x_full, y_full,
                       new_train_mask, new_val_mask, original_test_mask,  # 关键：用新训练/验证mask，原始测试mask
                       pos_weight, device, model_save_path,
                       epochs=500, lr=0.01, weight_decay=5e-4,
                       max_patience=30):
    model = FullGraphGNN(in_dim=x_full.size(1), hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_val_aupr = 0.0
    patience = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # 训练：仅用“训练集内部划分的新训练集”
        out = model(x_full, edge_index, edge_weight)
        loss = criterion(out[new_train_mask], y_full[new_train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            # 验证：仅用“训练集内部划分的新验证集”
            val_out = model(x_full, edge_index, edge_weight)
            val_probs = torch.sigmoid(val_out[new_val_mask]).cpu().numpy()
            val_labels = y_full[new_val_mask].cpu().numpy()

            if val_labels.sum() == 0:
                val_aupr = 0.0
            else:
                precision, recall, _ = precision_recall_curve(val_labels, val_probs)
                val_aupr = auc(recall, precision)

        scheduler.step(val_aupr)

        if val_aupr > best_val_aupr:
            best_val_aupr = val_aupr
            patience = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience += 1

        if patience >= max_patience:
            break

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # 最终验证（选阈值）：仍用新验证集
        val_out = model(x_full, edge_index, edge_weight)
        val_probs = torch.sigmoid(val_out[new_val_mask]).cpu().numpy()
        val_labels = y_full[new_val_mask].cpu().numpy()

        best_thr = 0.5
        if val_labels.sum() > 0:
            precision, recall, thresholds = precision_recall_curve(val_labels, val_probs)
            best_metric = 0.0
            for i, thr in enumerate(thresholds):
                if recall[i] >= 0.60:
                    f1_local = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)
                    if f1_local > best_metric:
                        best_metric = f1_local
                        best_thr = thr
            if best_metric == 0.0:
                best_thr = 0.5

        # 测试：完全用原始测试集（不修改）
        test_out = model(x_full, edge_index, edge_weight)
        test_probs = torch.sigmoid(test_out[original_test_mask]).cpu().numpy()
        test_labels = y_full[original_test_mask].cpu().numpy()

        pred_test = (test_probs > best_thr).astype(int)
        acc = accuracy_score(test_labels, pred_test)
        auc_score = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.5
        precision, recall, _ = precision_recall_curve(test_labels, test_probs)
        aupr_score = auc(recall, precision)
        f1 = f1_score(test_labels, pred_test)
        prec = precision_score(test_labels, pred_test) if pred_test.sum() > 0 else 0.0
        rec = recall_score(test_labels, pred_test)

    return {
        'val_aupr': best_val_aupr,
        'test_acc': acc,
        'test_auc': auc_score,
        'test_aupr': aupr_score,
        'test_f1': f1,
        'test_prec': prec,
        'test_rec': rec,
        'best_thr': best_thr
    }


# === 核心：分阶段评估函数（适配精炼图，同样保留原始测试集）===
def train_and_evaluate_a(edge_index_train, edge_index_infer, x_full, y_full,
                         new_train_mask, new_val_mask, original_test_mask,
                         pos_weight, device, model_save_path,
                         epochs=500, lr=0.01, weight_decay=5e-4,
                         max_patience=30):
    model = FullGraphGNN(in_dim=x_full.size(1), hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_val_aupr = 0.0
    patience = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # 训练：用训练专用边 + 新训练集
        out = model(x_full, edge_index_train)
        loss = criterion(out[new_train_mask], y_full[new_train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            # 验证：用推理专用边 + 新验证集
            val_out = model(x_full, edge_index_infer)
            val_probs = torch.sigmoid(val_out[new_val_mask]).cpu().numpy()
            val_labels = y_full[new_val_mask].cpu().numpy()

            if val_labels.sum() == 0:
                val_aupr = 0.0
            else:
                precision, recall, _ = precision_recall_curve(val_labels, val_probs)
                val_aupr = auc(recall, precision)

        scheduler.step(val_aupr)

        if val_aupr > best_val_aupr:
            best_val_aupr = val_aupr
            patience = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience += 1

        if patience >= max_patience:
            break

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # 选阈值：新验证集
        val_out = model(x_full, edge_index_infer)
        val_probs = torch.sigmoid(val_out[new_val_mask]).cpu().numpy()
        val_labels = y_full[new_val_mask].cpu().numpy()

        best_thr = 0.5
        if val_labels.sum() > 0:
            precision, recall, thresholds = precision_recall_curve(val_labels, val_probs)
            best_metric = 0.0
            for i, thr in enumerate(thresholds):
                if recall[i] >= 0.60:
                    f1_local = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)
                    if f1_local > best_metric:
                        best_metric = f1_local
                        best_thr = thr
            if best_metric == 0.0:
                best_thr = 0.5

        # 测试：原始测试集
        test_out = model(x_full, edge_index_infer)
        test_probs = torch.sigmoid(test_out[original_test_mask]).cpu().numpy()
        test_labels = y_full[original_test_mask].cpu().numpy()

        pred_test = (test_probs > best_thr).astype(int)
        acc = accuracy_score(test_labels, pred_test)
        auc_score = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.5
        precision, recall, _ = precision_recall_curve(test_labels, test_probs)
        aupr_score = auc(recall, precision)
        f1 = f1_score(test_labels, pred_test)
        prec = precision_score(test_labels, pred_test) if pred_test.sum() > 0 else 0.0
        rec = recall_score(test_labels, pred_test)

    return {
        'val_aupr': best_val_aupr,
        'test_acc': acc,
        'test_auc': auc_score,
        'test_aupr': aupr_score,
        'test_f1': f1,
        'test_prec': prec,
        'test_rec': rec,
        'best_thr': best_thr
    }


# === 核心：准备基础数据（原始图、精炼图边，完全保留原始测试集）===
# 1. 原始图边（Baseline用）
orig_adj_dense = to_dense_adj(data_full.edge_index.to(primary_device), max_num_nodes=N)[0]
edge_index_orig, _ = dense_to_sparse(orig_adj_dense)
edge_index_orig = edge_index_orig.to(primary_device)

# 2. 精炼图边（Refined Graph用）
orig_edge_set = set(map(tuple, edge_index_orig.t().cpu().numpy()))
orig_edge_count = edge_index_orig.size(1)
candidate_edges = []
candidate_weights = []

for i in range(edge_index_ensemble.size(1)):
    u, v = edge_index_ensemble[:, i].cpu().tolist()
    w = final_weights[i].item()
    # 跳过已存在的边
    if (u, v) in orig_edge_set or (v, u) in orig_edge_set:
        continue
    # 相似度阈值过滤
    if w <= 0.6:
        continue
    candidate_edges.append([u, v])
    candidate_weights.append(w)

# 处理候选边（无候选边则用原始图）
if not candidate_edges:
    edge_index_train_refined = edge_index_orig
    edge_index_full_refined = edge_index_orig
else:
    candidate_edge_index = torch.tensor(candidate_edges, dtype=torch.long).t().contiguous().to(primary_device)
    candidate_edge_weight = torch.tensor(candidate_weights, dtype=torch.float).to(primary_device)

    # 精炼图（推理用）：原始边 + 新候选边
    max_total_new = int(orig_edge_count * 0.2)  # 最多新增20%边
    if candidate_edge_index.size(1) > max_total_new:
        topk_total = torch.topk(candidate_edge_weight, max_total_new)
        candidate_edge_index = candidate_edge_index[:, topk_total.indices]
    edge_index_full_refined = torch.cat([edge_index_orig, candidate_edge_index], dim=1)
    edge_index_full_refined, _ = coalesce(edge_index_full_refined, None, num_nodes=N)

    # 训练用精炼图：后续按“新训练集节点”过滤
    edge_index_train_refined = edge_index_full_refined

# 3. 提取原始训练集节点（核心：仅对这些节点用4个函数划分）
original_train_nodes = torch.where(data_full.train_mask)[0].cpu().numpy()
# 提取原始测试集mask（全程不变）
if isinstance(data_full.test_mask, np.ndarray):
    data_full.test_mask = torch.from_numpy(data_full.test_mask)
original_test_mask = data_full.test_mask.clone()

print(f"\n=== 训练集内部划分配置 ===")
print(f"原始训练集节点总数: {len(original_train_nodes)}")
print(f"每种划分后新训练集节点数: ~{int(len(original_train_nodes) * 0.8)}")
print(f"每种划分后新验证集节点数: ~{int(len(original_train_nodes) * 0.2)}")
print(f"原始测试集节点数: {original_test_mask.sum().item()}（全程不变）")

# === 核心：批量用4个函数划分“原始训练集节点”，并测试性能 ===
# 定义4种划分方式（输入：原始训练集节点 + 全量特征）
split_functions = [
    ('SNV', lambda nodes: SNV_split(nodes, data_full.x)),
    ('GE', lambda nodes: GE_split(nodes, data_full.x)),
    ('METH', lambda nodes: METH_split(nodes, data_full.x)),
    ('CNA', lambda nodes: CNA_split(nodes, data_full.x))
]

# 存储所有结果
all_results = []

# 遍历每种划分方式
for split_name, split_func in split_functions:
    print(f"\n" + "=" * 70)
    print(f"📝 正在测试：用{split_name}特征划分原始训练集节点")
    print(f"=" * 70)

    # 步骤1：对原始训练集节点，用当前函数划分为“新训练集”和“新验证集”
    new_train_idx, new_val_idx = split_func(original_train_nodes)

    # 步骤2：转换为mask（适配模型输入）
    new_train_mask = torch.zeros(N, dtype=torch.bool, device=primary_device)
    new_train_mask[new_train_idx] = True
    new_val_mask = torch.zeros(N, dtype=torch.bool, device=primary_device)
    new_val_mask[new_val_idx] = True

    # 步骤3：计算正样本权重（基于新训练集）
    data_full.y = data_full.y.to(new_train_mask.device)
    train_labels = data_full.y[new_train_mask]
    pos_weight = (1 - train_labels.mean()) / train_labels.mean() if train_labels.mean() > 0 else torch.tensor(1.0,
                                                                                                              device=primary_device)

    # 打印当前划分的节点数量（验证正确性）
    print(f"当前划分节点数：新训练集={new_train_mask.sum().item()}, 新验证集={new_val_mask.sum().item()}")

    # ---------------------- 实验1：原始图（Baseline）----------------------
    print(f"\n🧪 实验1：原始图模型")
    data_full.y=data_full.y.to(primary_device)
    data_full.x=data_full.x.to(primary_device)
    edge_index_orig=edge_index_orig.to(primary_device)

    metrics_orig = train_and_evaluate(
        edge_index_orig, None,
        data_full.x, data_full.y,
        new_train_mask, new_val_mask, original_test_mask,  # 关键：新训练/验证，原始测试
        pos_weight, primary_device,
        model_save_path=f'best_orig_{split_name}.pth'
    )
    all_results.append({
        '划分方式': split_name,
        '模型类型': '原始图',
        '验证集AUPR': metrics_orig['val_aupr'],
        '测试集ACC': metrics_orig['test_acc'],
        '测试集AUC': metrics_orig['test_auc'],
        '测试集AUPR': metrics_orig['test_aupr'],
        '测试集F1': metrics_orig['test_f1'],
        '测试集Precision': metrics_orig['test_prec'],
        '测试集Recall': metrics_orig['test_rec'],
        '最优阈值': metrics_orig['best_thr']
    })
    print(f"✅ 原始图 | 测试集AUPR: {metrics_orig['test_aupr']:.4f}, F1: {metrics_orig['test_f1']:.4f}")

    # ---------------------- 实验2：精炼图（Refined Graph）----------------------
    print(f"\n🧪 实验2：精炼图模型")
    # 过滤精炼图训练边：仅保留“新训练集节点”之间的边（避免数据泄露）
    new_train_nodes_set = set(new_train_idx)
    u_list = edge_index_train_refined[0].cpu().tolist()
    v_list = edge_index_train_refined[1].cpu().tolist()
    u_in_train = torch.tensor([u in new_train_nodes_set for u in u_list], device=primary_device)
    v_in_train = torch.tensor([v in new_train_nodes_set for v in v_list], device=primary_device)
    train_edge_mask = u_in_train & v_in_train
    edge_index_train_current = edge_index_train_refined[:, train_edge_mask]

    metrics_refined = train_and_evaluate_a(
        edge_index_train=edge_index_train_current,
        edge_index_infer=edge_index_full_refined,
        x_full=data_full.x, y_full=data_full.y,
        new_train_mask=new_train_mask, new_val_mask=new_val_mask, original_test_mask=original_test_mask,
        pos_weight=pos_weight, device=primary_device,
        model_save_path=f'best_refined_{split_name}.pth'
    )
    all_results.append({
        '划分方式': split_name,
        '模型类型': '精炼图',
        '验证集AUPR': metrics_refined['val_aupr'],
        '测试集ACC': metrics_refined['test_acc'],
        '测试集AUC': metrics_refined['test_auc'],
        '测试集AUPR': metrics_refined['test_aupr'],
        '测试集F1': metrics_refined['test_f1'],
        '测试集Precision': metrics_refined['test_prec'],
        '测试集Recall': metrics_refined['test_rec'],
        '最优阈值': metrics_refined['best_thr']
    })
    print(f"✅ 精炼图 | 测试集AUPR: {metrics_refined['test_aupr']:.4f}, F1: {metrics_refined['test_f1']:.4f}")

    # ---------------------- 实验3：集成预测（Ensemble）----------------------
    print(f"\n🧪 实验3：集成模型（原始图+精炼图）")
    # 加载两种模型
    model_orig = FullGraphGNN(in_dim=data_full.x.size(1), hidden_dim=64).to(primary_device)
    model_orig.load_state_dict(torch.load(f'best_orig_{split_name}.pth', map_location=primary_device))
    model_orig.eval()

    model_refined = FullGraphGNN(in_dim=data_full.x.size(1), hidden_dim=64).to(primary_device)
    model_refined.load_state_dict(torch.load(f'best_refined_{split_name}.pth', map_location=primary_device))
    model_refined.eval()

    with torch.no_grad():
        # 用新验证集选阈值
        val_out_orig = model_orig(data_full.x, edge_index_orig)[new_val_mask]
        val_out_refined = model_refined(data_full.x, edge_index_full_refined)[new_val_mask]
        val_ens_logits = (val_out_orig + val_out_refined) / 2.0
        val_ens_probs = torch.sigmoid(val_ens_logits).cpu().numpy()
        val_ens_labels = data_full.y[new_val_mask].cpu().numpy()

        best_ens_thr = 0.5
        if val_ens_labels.sum() > 0:
            precision, recall, thresholds = precision_recall_curve(val_ens_labels, val_ens_probs)
            best_metric = 0.0
            for i, thr in enumerate(thresholds):
                if recall[i] >= 0.60:
                    f1_local = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)
                    if f1_local > best_metric:
                        best_metric = f1_local
                        best_ens_thr = thr
            if best_metric == 0.0:
                best_ens_thr = 0.5

        # 用原始测试集评估
        test_out_orig = model_orig(data_full.x, edge_index_orig)[original_test_mask]
        test_out_refined = model_refined(data_full.x, edge_index_full_refined)[original_test_mask]
        test_ens_logits = (test_out_orig + test_out_refined) / 2.0
        test_ens_probs = torch.sigmoid(test_ens_logits).cpu().numpy()
        test_ens_labels = data_full.y[original_test_mask].cpu().numpy()

    # 计算集成模型指标
    pred_ens = (test_ens_probs > best_ens_thr).astype(int)
    ens_acc = accuracy_score(test_ens_labels, pred_ens)
    ens_auc = roc_auc_score(test_ens_labels, test_ens_probs) if len(np.unique(test_ens_labels)) > 1 else 0.5
    ens_prec, ens_rec, _ = precision_recall_curve(test_ens_labels, test_ens_probs)
    ens_aupr = auc(ens_rec, ens_prec)
    ens_f1 = f1_score(test_ens_labels, pred_ens)
    ens_precision = precision_score(test_ens_labels, pred_ens) if pred_ens.sum() > 0 else 0.0
    ens_recall = recall_score(test_ens_labels, pred_ens)

    all_results.append({
        '划分方式': split_name,
        '模型类型': '集成模型',
        '验证集AUPR': 0.0,  # 集成模型无单独验证AUPR
        '测试集ACC': ens_acc,
        '测试集AUC': ens_auc,
        '测试集AUPR': ens_aupr,
        '测试集F1': ens_f1,
        '测试集Precision': ens_precision,
        '测试集Recall': ens_recall,
        '最优阈值': best_ens_thr
    })
    print(f"✅ 集成模型 | 测试集AUPR: {ens_aupr:.4f}, F1: {ens_f1:.4f}")

# === 输出最终汇总结果 ===
print(f"\n" + "=" * 120)
print("📊 最终性能汇总（仅划分原始训练集，原始测试集不变）")
print("=" * 120)
df_results = pd.DataFrame(all_results)
# 按“划分方式”和“测试集AUPR”排序（每种划分下最优模型在前）
df_results_sorted = df_results.sort_values(by=['划分方式', '测试集AUPR'], ascending=[True, False])
print(df_results_sorted.to_string(index=False, float_format="%.4f"))

# 统计每种划分方式下的最优模型（按测试集AUPR）
print(f"\n" + "=" * 80)
print("🏆 每种划分方式下的最优模型（按测试集AUPR排序）")
print("=" * 80)
best_per_split = []
for split_name in ['SNV', 'GE', 'METH', 'CNA']:
    split_data = df_results[df_results['划分方式'] == split_name]
    best_row = split_data.loc[split_data['测试集AUPR'].idxmax()]
    best_per_split.append(best_row)

df_best = pd.DataFrame(best_per_split)
print(df_best[['划分方式', '模型类型', '测试集AUPR', '测试集F1', '测试集ACC']].to_string(index=False,
                                                                                         float_format="%.4f"))

# 保存结果到CSV（便于后续分析）
df_results_sorted.to_csv('train_subset_split_results.csv', index=False, float_format="%.4f")
print(f"\n💾 结果已保存到 'train_subset_split_results.csv'")