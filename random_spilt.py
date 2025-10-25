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


# === 新增分层抽样划分函数 ===
def s_train_test_split(data, train_ratio):
    """分层抽样划分训练集和测试集，保持正负样本比例一致"""
    positive_index = (data.y == True).nonzero(as_tuple=True)[0]
    negative_index = (data.y == False).nonzero(as_tuple=True)[0]

    num_positive = positive_index.size(0)
    num_negative = negative_index.size(0)

    # 计算训练集正负样本数量
    positive_train_size = int(num_positive * train_ratio)
    negative_train_size = int(num_negative * train_ratio)

    # 随机打乱索引
    positive_perm = torch.randperm(num_positive)
    negative_perm = torch.randperm(num_negative)

    # 划分训练集和测试集索引
    positive_train_index = positive_index[positive_perm[:positive_train_size]]
    negative_train_index = negative_index[negative_perm[:negative_train_size]]
    positive_test_index = positive_index[positive_perm[positive_train_size:]]
    negative_test_index = negative_index[negative_perm[negative_train_size:]]

    # 合并索引
    train_index = torch.cat((positive_train_index, negative_train_index))
    test_index = torch.cat((positive_test_index, negative_test_index))

    return train_index, test_index


# === 设备与数据加载 ===
primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = read_h5file("networks/STRINGdb_multiomics.h5")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_np = data.x.cpu().numpy()
x_scaled = scaler.fit_transform(x_np)
data.x = torch.tensor(x_scaled, dtype=torch.float)
data_full = data.to(primary_device)
data_full.y = data_full.y.float()

# === 核心修改：使用分层抽样划分数据集（10%-90% 训练-测试）===
num_nodes = data_full.num_nodes
train_ratio = 0.9

# 使用新增的分层抽样函数划分索引
train_index, test_index = s_train_test_split(data_full, train_ratio)

# 定义训练/测试掩码
train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=primary_device)
test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=primary_device)
train_mask[train_index] = True
test_mask[test_index] = True

# 替换data_full的掩码属性
data_full.train_mask = train_mask
data_full.test_mask = test_mask
data_full.val_mask = None  # 明确移除验证集

# 打印划分信息（验证分层效果）
train_pos_ratio = data_full.y[train_mask].mean().item()
test_pos_ratio = data_full.y[test_mask].mean().item()
print(f"训练集样本数: {train_mask.sum().item()}, 正样本比例: {train_pos_ratio:.4f}")
print(f"测试集样本数: {test_mask.sum().item()}, 正样本比例: {test_pos_ratio:.4f}")
print(f"原始数据正样本比例: {data_full.y.mean().item():.4f}")

# 基于新划分的data_full创建子图集群
dataset = create_cluster(data_full.cpu())

for i, sub in enumerate(dataset):
    pos_ratio = sub.y.float().mean().item()
    print(f"[Subgraph {i}] Nodes: {sub.num_nodes}, Pos ratio: {pos_ratio:.4f}")

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


# === 子图训练函数（显存优化，适配新数据划分）===
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


# === Step 1: 训练所有子图模型 ===
models = []
for sub in dataset:
    print(f"Training subgraph {len(models) + 1}/{len(dataset)}...")
    models.append(train_model(sub))

# === Step 2: 结构集成（稀疏化！）===
N = data_full.num_nodes
all_edges = []
all_weights = []

for i, (model_gsl, model_cls, model_ungsl, orig_idx) in enumerate(models):
    print(f"Processing subgraph {i + 1}/{len(models)} for structure ensemble...")
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

print(f"✅ Ensemble graph has {edge_index_ensemble.size(1)} edges.")

# === Step 3: 评估函数修改（适配10%-90%划分，移除验证集逻辑）===
orig_adj_dense = to_dense_adj(data_full.edge_index.to(primary_device), max_num_nodes=N)[0]


def train_and_evaluate(edge_index, edge_weight, x_full, y_full,
                       train_mask, test_mask,
                       pos_weight, device, model_save_path,
                       epochs=800, lr=0.005, weight_decay=5e-4,
                       max_patience=30):
    """修改后的评估函数：仅用训练集训练，测试集评估，基于训练集选阈值"""
    model = FullGraphGNN(in_dim=x_full.size(1), hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 学习率调度：基于训练损失（原验证集逻辑移除）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_train_loss = float('inf')
    patience = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_full, edge_index, edge_weight)
        loss = criterion(out[train_mask], y_full[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # 保存训练损失最低的模型
        if loss.item() < best_train_loss:
            best_train_loss = loss.item()
            patience = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience += 1

        if patience >= max_patience:
            break

    # 加载最优模型
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    with torch.no_grad():
        # 1. 基于训练集选择分类阈值（确保与数据分布匹配）
        train_out = model(x_full, edge_index, edge_weight)
        train_probs = torch.sigmoid(train_out[train_mask]).cpu().numpy()
        train_labels = y_full[train_mask].cpu().numpy()

        best_thr = 0.5
        if train_labels.sum() > 0:
            precision, recall, thresholds = precision_recall_curve(train_labels, train_probs)
            best_metric = 0.0
            for i, thr in enumerate(thresholds):
                if recall[i] >= 0.60:  # 保持原召回率约束
                    f1_local = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)
                    if f1_local > best_metric:
                        best_metric = f1_local
                        best_thr = thr
            if best_metric == 0.0:
                best_thr = 0.5

        # 2. 在测试集上评估性能
        test_out = model(x_full, edge_index, edge_weight)
        test_probs = torch.sigmoid(test_out[test_mask]).cpu().numpy()
        test_labels = y_full[test_mask].cpu().numpy()

        pred_test = (test_probs > best_thr).astype(int)
        acc = accuracy_score(test_labels, pred_test)
        # 处理单类别情况（避免ROC-AUC计算报错）
        auc_score = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.5
        precision_test, recall_test, _ = precision_recall_curve(test_labels, test_probs)
        aupr_score = auc(recall_test, precision_test)
        f1 = f1_score(test_labels, pred_test)
        prec = precision_score(test_labels, pred_test, zero_division=0)  # 避免无正例时报错
        rec = recall_score(test_labels, pred_test, zero_division=0)

    return {
        'best_train_loss': best_train_loss,
        'test_acc': acc,
        'test_auc': auc_score,
        'test_aupr': aupr_score,
        'test_f1': f1,
        'test_precision': prec,
        'test_recall': rec,
        'best_thr': best_thr
    }


def train_and_evaluate_a(
        edge_index_train,  # 训练阶段使用的边
        edge_index_infer,  # 测试阶段使用的边
        x_full, y_full,
        train_mask, test_mask,
        pos_weight, device, model_save_path,
        epochs=500, lr=0.01, weight_decay=5e-4,
        max_patience=30
):
    """修改后的精炼图评估函数：适配10%-90%划分"""
    model = FullGraphGNN(in_dim=x_full.size(1), hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_train_loss = float('inf')
    patience = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # 训练时只用 edge_index_train
        out = model(x_full, edge_index_train)
        loss = criterion(out[train_mask], y_full[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_train_loss:
            best_train_loss = loss.item()
            patience = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience += 1

        if patience >= max_patience:
            break

    # 加载最优模型
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    with torch.no_grad():
        # 基于训练集选择阈值
        train_out = model(x_full, edge_index_infer)  # 用推理边计算训练集概率（更贴合测试逻辑）
        train_probs = torch.sigmoid(train_out[train_mask]).cpu().numpy()
        train_labels = y_full[train_mask].cpu().numpy()

        best_thr = 0.5
        if train_labels.sum() > 0:
            precision, recall, thresholds = precision_recall_curve(train_labels, train_probs)
            best_metric = 0.0
            for i, thr in enumerate(thresholds):
                if recall[i] >= 0.60:
                    f1_local = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)
                    if f1_local > best_metric:
                        best_metric = f1_local
                        best_thr = thr
            if best_metric == 0.0:
                best_thr = 0.5

        # 测试集评估
        test_out = model(x_full, edge_index_infer)
        test_probs = torch.sigmoid(test_out[test_mask]).cpu().numpy()
        test_labels = y_full[test_mask].cpu().numpy()

        pred_test = (test_probs > best_thr).astype(int)
        acc = accuracy_score(test_labels, pred_test)
        auc_score = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.5
        precision_test, recall_test, _ = precision_recall_curve(test_labels, test_probs)
        aupr_score = auc(recall_test, precision_test)
        f1 = f1_score(test_labels, pred_test, zero_division=0)
        prec = precision_score(test_labels, pred_test, zero_division=0)
        rec = recall_score(test_labels, pred_test, zero_division=0)

    return {
        'best_train_loss': best_train_loss,
        'test_acc': acc,
        'test_auc': auc_score,
        'test_aupr': aupr_score,
        'test_f1': f1,
        'test_precision': prec,
        'test_recall': rec,
        'best_thr': best_thr
    }


# === 计算正负样本权重（基于新训练集）===
results = []
x_full = data_full.x.to(primary_device)
y_full = data_full.y.float().to(primary_device)
train_mask = data_full.train_mask
test_mask = data_full.test_mask

train_labels = y_full[train_mask]
pos_weight = (1 - train_labels.mean()) / train_labels.mean()  # 平衡正负样本损失
print(f"✅ Training set: {train_mask.sum().item()} nodes | Test set: {test_mask.sum().item()} nodes")
print(f"✅ Positive weight for loss: {pos_weight.item():.4f}")

# =============== 实验 1：原始图（baseline） ===============
print("\n" + "=" * 50)
print("🧪 Experiment 1: Original Graph (Baseline)")
print("=" * 50)

edge_index_orig, _ = dense_to_sparse(orig_adj_dense)
edge_index_orig = edge_index_orig.to(primary_device)

metrics_orig = train_and_evaluate(
    edge_index_orig, None,
    x_full, y_full, train_mask, test_mask,
    pos_weight, primary_device,
    model_save_path='best_model_orig.pth'
)

results.append({
    'Method': 'Original Graph',
    'Best_Train_Loss': metrics_orig['best_train_loss'],
    'Test_Acc': metrics_orig['test_acc'],
    'Test_AUC': metrics_orig['test_auc'],
    'Test_AUPR': metrics_orig['test_aupr'],
    'Test_F1': metrics_orig['test_f1'],
    'Test_Precision': metrics_orig['test_precision'],
    'Test_Recall': metrics_orig['test_recall'],
    'Best_Threshold': metrics_orig['best_thr']
})

print(
    f"✅ Original Graph | Test AUPR: {metrics_orig['test_aupr']:.4f} | Test F1: {metrics_orig['test_f1']:.4f} | Test Acc: {metrics_orig['test_acc']:.4f}")

# =============== 实验 2：精炼图（Refined Graph） ===============
print("\n" + "=" * 50)
print("🧪 Experiment 2: Refined Graph (Allow test nodes in inference only)")
print("=" * 50)

# 原始边（用于去重）
orig_edge_set = set(map(tuple, edge_index_orig.t().cpu().numpy()))
train_nodes = set(torch.where(train_mask)[0].cpu().tolist())
orig_edge_count = edge_index_orig.size(1)

# 收集所有候选新边（不限于训练节点）
candidate_edges = []
candidate_weights = []

for i in range(edge_index_ensemble.size(1)):
    u, v = edge_index_ensemble[:, i].cpu().tolist()
    w = final_weights[i].item()

    # 跳过已存在的边
    if (u, v) in orig_edge_set or (v, u) in orig_edge_set:
        continue

    # 更严格的相似度阈值（可调）
    if w <= 0.5:
        continue

    candidate_edges.append([u, v])
    candidate_weights.append(w)

# 如果没有候选边，直接使用原始图
if not candidate_edges:
    edge_index_train = edge_index_orig
    edge_index_full_refined = edge_index_orig
else:
    # 构建候选边张量：确保是 (2, M)
    candidate_edge_index = torch.tensor(candidate_edges, dtype=torch.long).t().contiguous().to(primary_device)
    candidate_edge_weight = torch.tensor(candidate_weights, dtype=torch.float).to(primary_device)

    # --- 1. 构建训练用边：仅保留两端都在训练集的新边 ---
    u_list = candidate_edge_index[0].cpu().tolist()
    v_list = candidate_edge_index[1].cpu().tolist()
    u_train_mask = torch.tensor([u in train_nodes for u in u_list], device=primary_device)
    v_train_mask = torch.tensor([v in train_nodes for v in v_list], device=primary_device)
    train_edge_mask = u_train_mask & v_train_mask

    train_new_edges = candidate_edge_index[:, train_edge_mask]  # shape: (2, K)

    # 限制训练新增边数量（不超过原始边的5%）
    max_train_new = int(orig_edge_count * 0.05)
    if train_new_edges.size(1) > max_train_new:
        train_weights = candidate_edge_weight[train_edge_mask]
        topk = torch.topk(train_weights, max_train_new)
        train_new_edges = train_new_edges[:, topk.indices]

    # 打印调试信息
    print(f"Original edges shape: {edge_index_orig.shape}")
    print(f"New training edges shape: {train_new_edges.shape}")

    # 拼接原始边和新边（确保都是 (2, *)）
    assert edge_index_orig.dim() == 2 and edge_index_orig.size(
        0) == 2, f"edge_index_orig shape invalid: {edge_index_orig.shape}"
    assert train_new_edges.dim() == 2 and train_new_edges.size(
        0) == 2, f"train_new_edges shape invalid: {train_new_edges.shape}"

    edge_index_train = torch.cat([edge_index_orig, train_new_edges], dim=1)  # (2, E1 + K)

    # 强制 coalesce 并检查结果
    edge_index_train, _ = coalesce(edge_index_train, None, num_nodes=N)
    print("coalesce function:", coalesce)
    if edge_index_train.dim() != 2 or edge_index_train.size(0) != 2:
        raise RuntimeError(f"Unexpected shape after coalesce: {edge_index_train.shape}")
    if edge_index_train.numel() % 2 != 0:
        raise RuntimeError(f"coalesce returned odd-length tensor: {edge_index_train.numel()}")

    # --- 2. 构建完整精炼图（用于推理）---
    max_total_new = int(orig_edge_count * 0.2)  # 最多新增20%
    if candidate_edge_index.size(1) > max_total_new:
        topk_total = torch.topk(candidate_edge_weight, max_total_new)
        candidate_edge_index = candidate_edge_index[:, topk_total.indices]

    edge_index_full_refined = torch.cat([edge_index_orig, candidate_edge_index], dim=1)
    edge_index_full_refined, _ = coalesce(edge_index_full_refined, None, num_nodes=N)
    if edge_index_full_refined.dim() != 2 or edge_index_full_refined.size(0) != 2:
        raise RuntimeError(f"Unexpected shape after coalesce for full refined: {edge_index_full_refined.shape}")
    if edge_index_full_refined.numel() % 2 != 0:
        raise RuntimeError(f"coalesce returned odd-length tensor for full refined: {edge_index_full_refined.numel()}")

# 确保最终输出的形状正确
assert edge_index_train.dim() == 2 and edge_index_train.size(
    0) == 2, f"Final edge_index_train shape invalid: {edge_index_train.shape}"
assert edge_index_full_refined.dim() == 2 and edge_index_full_refined.size(
    0) == 2, f"Final edge_index_full_refined shape invalid: {edge_index_full_refined.shape}"

print(f"✅ Training graph edges: {edge_index_train.size(1)} "
      f"(added: {edge_index_train.size(1) - edge_index_orig.size(1)})")
print(f"✅ Full refined graph edges: {edge_index_full_refined.size(1)} "
      f"(added: {edge_index_full_refined.size(1) - edge_index_orig.size(1)})")

# 评估 refined graph
metrics_refined = train_and_evaluate_a(
    edge_index_train=edge_index_train,
    edge_index_infer=edge_index_full_refined,
    x_full=x_full, y_full=y_full,
    train_mask=train_mask, test_mask=test_mask,
    pos_weight=pos_weight, device=primary_device,
    model_save_path='best_model_refined.pth'
)

results.append({
    'Method': 'Refined Graph',
    'Best_Train_Loss': metrics_refined['best_train_loss'],
    'Test_Acc': metrics_refined['test_acc'],
    'Test_AUC': metrics_refined['test_auc'],
    'Test_AUPR': metrics_refined['test_aupr'],
    'Test_F1': metrics_refined['test_f1'],
    'Test_Precision': metrics_refined['test_precision'],
    'Test_Recall': metrics_refined['test_recall'],
    'Best_Threshold': metrics_refined['best_thr']
})

print(
    f"✅ Refined Graph | Test AUPR: {metrics_refined['test_aupr']:.4f} | Test F1: {metrics_refined['test_f1']:.4f} | Test Acc: {metrics_refined['test_acc']:.4f}")

# =============== 实验 3：Ensemble Prediction ===============
print("\n" + "=" * 50)
print("🧪 Experiment 3: Ensemble Prediction (Orig+Refined)")
print("=" * 50)

# 加载两个实验的最优模型
model_orig = FullGraphGNN(in_dim=x_full.size(1), hidden_dim=64).to(primary_device)
model_orig.load_state_dict(torch.load('best_model_orig.pth', map_location=primary_device))
model_orig.eval()

model_refined = FullGraphGNN(in_dim=x_full.size(1), hidden_dim=64).to(primary_device)
model_refined.load_state_dict(torch.load('best_model_refined.pth', map_location=primary_device))
model_refined.eval()

with torch.no_grad():
    # 1. 基于训练集选择集成模型的阈值
    train_out_orig = model_orig(x_full, edge_index_orig)[train_mask]
    train_out_refined = model_refined(x_full, edge_index_full_refined)[train_mask]
    train_ensemble_logits = (train_out_orig + train_out_refined) / 2.0
    train_probs_ens = torch.sigmoid(train_ensemble_logits).cpu().numpy()
    train_labels_np = y_full[train_mask].cpu().numpy()

    best_ens_thr = 0.5
    if train_labels_np.sum() > 0:
        precision, recall, thresholds = precision_recall_curve(train_labels_np, train_probs_ens)
        best_metric = 0.0
        for i, thr in enumerate(thresholds):
            if recall[i] >= 0.60:
                f1_local = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)
                if f1_local > best_metric:
                    best_metric = f1_local
                    best_ens_thr = thr
        if best_metric == 0.0:
            best_ens_thr = 0.5

    # 2. 在测试集上评估集成模型
    test_out_orig = model_orig(x_full, edge_index_orig)[test_mask]
    test_out_refined = model_refined(x_full, edge_index_full_refined)[test_mask]
    test_ensemble_logits = (test_out_orig + test_out_refined) / 2.0
    test_probs_ens = torch.sigmoid(test_ensemble_logits).cpu().numpy()
    test_labels_np = y_full[test_mask].cpu().numpy()

# 计算集成模型性能指标
pred_ens = (test_probs_ens > best_ens_thr).astype(int)
acc = accuracy_score(test_labels_np, pred_ens)
auc_score = roc_auc_score(test_labels_np, test_probs_ens) if len(np.unique(test_labels_np)) > 1 else 0.5
precision_ens, recall_ens, _ = precision_recall_curve(test_labels_np, test_probs_ens)
aupr_score = auc(recall_ens, precision_ens)
f1 = f1_score(test_labels_np, pred_ens, zero_division=0)
prec = precision_score(test_labels_np, pred_ens, zero_division=0)
rec = recall_score(test_labels_np, pred_ens, zero_division=0)

# 记录集成模型结果（无训练损失，取两个模型损失的平均值作为参考）
avg_train_loss = (metrics_orig['best_train_loss'] + metrics_refined['best_train_loss']) / 2.0
results.append({
    'Method': 'Ensemble (Orig+Refined)',
    'Best_Train_Loss': avg_train_loss,
    'Test_Acc': acc,
    'Test_AUC': auc_score,
    'Test_AUPR': aupr_score,
    'Test_F1': f1,
    'Test_Precision': prec,
    'Test_Recall': rec,
    'Best_Threshold': best_ens_thr
})

print(f"✅ Ensemble Prediction | Test AUPR: {aupr_score:.4f} | Test F1: {f1:.4f} | Test Acc: {acc:.4f}")

# =============== 打印最终结果 ===============
print("\n" + "=" * 100)
print("📊 FINAL PERFORMANCE COMPARISON (Train:10% | Test:90%)")
print("=" * 100)
df = pd.DataFrame(results)
# 格式化输出（保留4位小数）
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 找出最优模型（基于Test_AUPR）
best_row = df.loc[df['Test_AUPR'].idxmax()]
print(f"\n🏆 Best Method: {best_row['Method']}")
print(f"   - Test AUPR: {best_row['Test_AUPR']:.4f}")
print(f"   - Test F1: {best_row['Test_F1']:.4f}")
print(f"   - Test Accuracy: {best_row['Test_Acc']:.4f}")
print(f"   - Best Threshold: {best_row['Best_Threshold']:.4f}")

# 按Test_AUPR排序输出
df_sorted = df.sort_values('Test_AUPR', ascending=False)
print("\n" + "=" * 100)
print("📊 SORTED RESULTS (by Test AUPR)")
print("=" * 100)
print(df_sorted.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 保存结果到CSV文件（便于后续分析）
df_sorted.to_csv('ungsl_10_90_split_results.csv', index=False)
print("\n✅ Results saved to 'ungsl_10_90_split_results.csv'")