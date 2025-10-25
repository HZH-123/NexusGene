import numpy as np
import torch
import torch_geometric
from networkx import eigenvector_centrality

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, recall_score, accuracy_score
)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse, degree, coalesce, subgraph
from UNGSL_test.cluster import create_cluster
from UNGSL_test.data_h5_loader import read_h5file
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
dataset = create_cluster(data.cpu())

for i, sub in enumerate(dataset):
    pos_ratio = sub.y.float().mean().item()
    print(f"[Subgraph {i}] Nodes: {sub.num_nodes}, Pos ratio: {pos_ratio:.4f}")

config = {
    'gsl_hidden': 256,
    'cls_hidden': 128,
    'dropout': 0.1,
    'threshold': 0.01,
    'alpha': 0.8,
    'reg_weight': 1e-5,
    'topk': 5,
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
            heads=1,
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

# === 子图训练函数（显存优化）===
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

    orig_idx = subgraph.orig_node_idx.clone()
    del adj_dense, S, S_hat, edge_index_S, edge_index_finetune, subgraph
    torch.cuda.empty_cache()
    return model_gsl, model_cls, model_ungsl, orig_idx

# === Step 1: 训练所有子图模型 ===
models = []
for sub in dataset:
    print(f"Training subgraph {len(models)+1}/{len(dataset)}...")
    models.append(train_model(sub))

# === Step 2: 结构集成（稀疏化！）===
N = data_full.num_nodes
all_edges = []
all_weights = []

for i, (model_gsl, model_cls, model_ungsl, orig_idx) in enumerate(models):
    print(f"Processing subgraph {i+1}/{len(models)} for structure ensemble...")
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

    del S_sub, S_hat_sub, mask, adj_sub, x_sub, subgraph
    torch.cuda.empty_cache()

if all_edges:
    edge_index_ensemble = torch.cat(all_edges, dim=1)
    edge_weight_ensemble = torch.cat(all_weights, dim=0)
    edge_index_ensemble, inverse = torch.unique(edge_index_ensemble, dim=1, return_inverse=True)
    final_weights = torch.zeros(edge_index_ensemble.size(1), device=primary_device)
    final_weights.scatter_reduce_(0, inverse, edge_weight_ensemble, reduce="max", include_self=False)
else:
    edge_index_ensemble = torch.empty((2, 0), dtype=torch.long, device=primary_device)
    final_weights = torch.empty(0, device=primary_device)

print(f"✅ Ensemble graph has {edge_index_ensemble.size(1)} edges.")

# === 构建训练子图（Inductive 设置）===
train_mask = data_full.train_mask
val_mask = data_full.val_mask
test_mask = data_full.test_mask

# 获取训练节点
train_nodes = torch.where(train_mask)[0]
# 提取训练子图的边（仅训练节点之间的边）
edge_index_train_only, _ = torch_geometric.utils.subgraph(train_nodes, data_full.edge_index, relabel_nodes=False)

# 同样处理精炼图：训练阶段只用训练节点之间的新边
orig_edge_set = set(map(tuple, edge_index_train_only.t().cpu().numpy()))

candidate_edges = []
candidate_weights = []

for i in range(edge_index_ensemble.size(1)):
    u, v = edge_index_ensemble[:, i].cpu().tolist()
    w = final_weights[i].item()
    if (u, v) in orig_edge_set or (v, u) in orig_edge_set:
        continue
    if w <= 0.5:
        continue
    # 仅保留两端都在训练集的新边（用于训练）
    if u in train_nodes and v in train_nodes:
        candidate_edges.append([u, v])
        candidate_weights.append(w)

if candidate_edges:
    candidate_edge_index = torch.tensor(candidate_edges, dtype=torch.long).t().to(primary_device)
    candidate_edge_weight = torch.tensor(candidate_weights, dtype=torch.float).to(primary_device)

    max_train_new = int(edge_index_train_only.size(1) * 0.05)
    if candidate_edge_index.size(1) > max_train_new:
        topk = torch.topk(candidate_edge_weight, max_train_new)
        candidate_edge_index = candidate_edge_index[:, topk.indices]
    edge_index_train_only=edge_index_train_only.to(primary_device)
    candidate_edge_index = candidate_edge_index.to(primary_device)
    edge_index_train_refined = torch.cat([edge_index_train_only, candidate_edge_index], dim=1)
    edge_index_train_refined, _ = coalesce(edge_index_train_refined, None, num_nodes=N)
else:
    edge_index_train_refined = edge_index_train_only

# 完整推理图（用于验证/测试）：原始图 + 所有高质量新边（不限于训练节点）
orig_adj_dense = to_dense_adj(data_full.edge_index, max_num_nodes=N)[0]
edge_index_orig_full, _ = dense_to_sparse(orig_adj_dense)
edge_index_orig_full = edge_index_orig_full.to(primary_device)

# 构建完整精炼图（用于推理）
all_candidate_edges = []
all_candidate_weights = []
for i in range(edge_index_ensemble.size(1)):
    u, v = edge_index_ensemble[:, i].cpu().tolist()
    w = final_weights[i].item()
    if w <= 0.6:
        continue
    if (u, v) in set(map(tuple, edge_index_orig_full.t().cpu().numpy())):
        continue
    all_candidate_edges.append([u, v])
    all_candidate_weights.append(w)

if all_candidate_edges:
    all_candidate_edge_index = torch.tensor(all_candidate_edges, dtype=torch.long).t().to(primary_device)
    max_total_new = int(edge_index_orig_full.size(1) * 0.2)
    if all_candidate_edge_index.size(1) > max_total_new:
        weights_all = torch.tensor(all_candidate_weights, device=primary_device)
        topk = torch.topk(weights_all, max_total_new)
        all_candidate_edge_index = all_candidate_edge_index[:, topk.indices]
    edge_index_full_refined = torch.cat([edge_index_orig_full, all_candidate_edge_index], dim=1)
    edge_index_full_refined, _ = coalesce(edge_index_full_refined, None, num_nodes=N)
else:
    edge_index_full_refined = edge_index_orig_full

print(f"✅ Inductive training edges (orig): {edge_index_train_only.size(1)}")
print(f"✅ Inductive training edges (refined): {edge_index_train_refined.size(1)}")
print(f"✅ Full graph for inference: {edge_index_full_refined.size(1)} edges")

# === 评估函数（Inductive）===
def train_and_evaluate_inductive(
    edge_index_train,        # 仅训练节点的边
    edge_index_infer,        # 完整图（含 val/test 节点）
    x_full, y_full,
    train_mask, val_mask, test_mask,
    pos_weight, device, model_save_path,
    epochs=500, lr=0.01, weight_decay=5e-4,
    max_patience=30
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 移动数据
    x_full = data_full.x.to(device)
    edge_index_train = edge_index_train.to(device)
    model = FullGraphGNN(in_dim=x_full.size(1), hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_val_aupr = 0.0
    patience = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_full, edge_index_train)  # 🟢 训练只用训练边
        loss = criterion(out[train_mask], y_full[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(x_full, edge_index_infer)  # 🟢 推理用完整图
            val_probs = torch.sigmoid(val_out[val_mask]).cpu().numpy()
            val_labels = y_full[val_mask].cpu().numpy()

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
        test_out = model(x_full, edge_index_infer)
        test_probs = torch.sigmoid(test_out[test_mask]).cpu().numpy()
        test_labels = y_full[test_mask].cpu().numpy()

        val_out = model(x_full, edge_index_infer)
        val_probs = torch.sigmoid(val_out[val_mask]).cpu().numpy()
        val_labels = y_full[val_mask].cpu().numpy()

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

        pred_test = (test_probs > best_thr).astype(int)
        acc = accuracy_score(test_labels, pred_test)
        auc_score = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.5
        precision, recall, _ = precision_recall_curve(test_labels, test_probs)
        aupr_score = auc(recall, precision)
        f1 = f1_score(test_labels, pred_test)
        prec = precision_score(test_labels, pred_test)
        rec = recall_score(test_labels, pred_test)

    return {
        'val_aupr': best_val_aupr,
        'acc': acc,
        'auc': auc_score,
        'aupr': aupr_score,
        'f1': f1,
        'prec': prec,
        'rec': rec,
        'best_thr': best_thr
    }

# === 准备标签和权重 ===
x_full = data_full.x.to(primary_device)
y_full = data_full.y.float().to(primary_device)
train_labels = y_full[train_mask]
pos_weight = (1 - train_labels.mean()) / train_labels.mean()

results = []

# =============== 实验 1: Original Graph (Inductive) ===============
print("\n" + "="*50)
print("🧪 Experiment 1: Original Graph (Inductive)")
print("="*50)

metrics_orig = train_and_evaluate_inductive(
    edge_index_train=edge_index_train_only,
    edge_index_infer=edge_index_orig_full,
    x_full=x_full, y_full=y_full,
    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    pos_weight=pos_weight, device=primary_device,
    model_save_path='best_model_orig.pth'
)


print(f"✅ Original Graph | Test AUPR: {metrics_orig['aupr']:.4f} | F1: {metrics_orig['f1']:.4f}")

# =============== 实验 2: Refined Graph (Inductive) ===============
print("\n" + "="*50)
print("🧪 Experiment 2: Refined Graph (Inductive)")
print("="*50)

metrics_refined = train_and_evaluate_inductive(
    edge_index_train=edge_index_train_refined,
    edge_index_infer=edge_index_full_refined,
    x_full=x_full, y_full=y_full,
    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    pos_weight=pos_weight, device=primary_device,
    model_save_path='best_model_refined.pth'
)



print(f"✅ Refined Graph | Test AUPR: {metrics_refined['aupr']:.4f} | F1: {metrics_refined['f1']:.4f}")

# =============== 实验 3: Ensemble Prediction ===============
print("\n" + "="*50)
print("🧪 Experiment 3: Ensemble Prediction (Inductive)")
print("="*50)

model_orig = FullGraphGNN(in_dim=x_full.size(1), hidden_dim=64).to(primary_device)
model_orig.load_state_dict(torch.load('best_model_orig.pth', map_location=primary_device))
model_orig.eval()

model_refined = FullGraphGNN(in_dim=x_full.size(1), hidden_dim=64).to(primary_device)
model_refined.load_state_dict(torch.load('best_model_refined.pth', map_location=primary_device))
model_refined.eval()

with torch.no_grad():
    val_out_orig = model_orig(x_full, edge_index_orig_full)[val_mask]
    val_out_refined = model_refined(x_full, edge_index_full_refined)[val_mask]
    val_ensemble_logits = (val_out_orig + val_out_refined) / 2.0
    val_probs_ens = torch.sigmoid(val_ensemble_logits).cpu().numpy()
    val_labels_np = y_full[val_mask].cpu().numpy()

best_ens_thr = 0.5
if val_labels_np.sum() > 0:
    precision, recall, thresholds = precision_recall_curve(val_labels_np, val_probs_ens)
    best_metric = 0.0
    for i, thr in enumerate(thresholds):
        if recall[i] >= 0.60:
            f1_local = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)
            if f1_local > best_metric:
                best_metric = f1_local
                best_ens_thr = thr
    if best_metric == 0.0:
        best_ens_thr = 0.5

with torch.no_grad():
    test_out_orig = model_orig(x_full, edge_index_orig_full)[test_mask]
    test_out_refined = model_refined(x_full, edge_index_full_refined)[test_mask]
    test_ensemble_logits = (test_out_orig + test_out_refined) / 2.0
    test_probs_ens = torch.sigmoid(test_ensemble_logits).cpu().numpy()
    test_labels_np = y_full[test_mask].cpu().numpy()

pred_ens = (test_probs_ens > best_ens_thr).astype(int)
acc = accuracy_score(test_labels_np, pred_ens)
auc_score = roc_auc_score(test_labels_np, test_probs_ens) if len(np.unique(test_labels_np)) > 1 else 0.5
precision, recall, _ = precision_recall_curve(test_labels_np, test_probs_ens)
aupr_score = auc(recall, precision)
f1 = f1_score(test_labels_np, pred_ens)
prec = precision_score(test_labels_np, pred_ens)
rec = recall_score(test_labels_np, pred_ens)

results.append({
    'Method': 'Ensemble (Orig+Refined, Inductive)',
    'Val_Best_AUPR': 0.0,
    'Test_Acc': acc,
    'Test_AUC': auc_score,
    'Test_AUPR': aupr_score,
    'Test_F1': f1,
    'Test_Precision': prec,
    'Test_Recall': rec,
    'Best_Threshold': best_ens_thr
})

print(f"✅ Ensemble Prediction | Test AUPR: {aupr_score:.4f} | F1: {f1:.4f}")

# =============== 打印结果 ===============
print("\n" + "="*80)
print("📊 FINAL COMPARISON (INDUCTIVE SETTING)")
print("="*80)
df = pd.DataFrame(results)
print(df.to_string(index=False, float_format="%.4f"))

best_row = df.loc[df['Test_AUPR'].idxmax()]
print(f"\n🏆 Best Method: {best_row['Method']}, Test AUPR = {best_row['Test_AUPR']:.4f}")

df_sorted = df.sort_values('Test_AUPR', ascending=False)
print("\n" + "="*80)
print("📊 FINAL RESULTS (sorted by Test AUPR)")
print("="*80)
print(df_sorted.to_string(index=False, float_format="%.4f"))