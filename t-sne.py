import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, recall_score, accuracy_score
)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from UNGSL_test.cluster import create_cluster
from UNGSL_test.data_h5_loader import read_h5file
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

# === 设备与数据加载 ===
primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = read_h5file("networks/LTG_multiomics.h5")
data_full = data.to(primary_device)
data_full.y = data_full.y.float()
dataset = create_cluster(data.cpu())

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
        # 残差连接（可选但推荐）
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
        # 第一层：GAT + SAGE，各输出 hidden_dim // 2
        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim // 2,
            heads=2,
            concat=False,  # 输出维度 = hidden_dim // 2（平均多头）
            dropout=0.3,
            add_self_loops=True  # 默认添加自环（推荐）
        )
        self.sage = SAGEConv(in_dim, hidden_dim // 2)

        self.bn1 = nn.LayerNorm(hidden_dim)
        # 第二层：SAGEConv（可继续使用 edge_weight）
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_weight=None):
        # GAT 分支（不使用 edge_weight，但可加自环）
        x1 = self.gat(x, edge_index)  # [N, hidden_dim//2]
        # SAGE 分支（支持 edge_weight）
        x2 = self.sage(x, edge_index, edge_weight)  # [N, hidden_dim//2]

        x = torch.cat([x1, x2], dim=-1)  # [N, hidden_dim]
        x = F.relu(self.bn1(x))
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight)  # [N, hidden_dim]
        x = F.relu(self.bn2(x))
        x = self.dropout(x)

        x = self.classifier(x)  # [N, out_dim]
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

# === 子图训练函数（保持不变）===
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
        loss = cls_loss + aff_loss  # ✅ 合并损失
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
        {'params': model_gsl.parameters(), 'lr': 1e-2},
        {'params': model_ungsl.parameters(), 'lr': 5e-2}
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

    return model_gsl, model_cls, model_ungsl, subgraph.orig_node_idx

# === Step 1: 训练所有子图模型 ===
models = []
for sub in dataset:
    print(f"Training subgraph {len(models)+1}/{len(dataset)}...")
    models.append(train_model(sub))

# === Step 2: 结构集成（保持不变）===
N = data_full.num_nodes
ensemble_adj = torch.zeros((N, N), device=primary_device)

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

        # ✅ 改进：自适应 top-p 稀疏化（替代固定 topk）
        triu_mask = torch.triu(torch.ones_like(S_hat_sub), diagonal=1).bool()
        values = S_hat_sub[triu_mask]
        if values.numel() > 0:
            k = max(5, int(config['top_p'] * values.numel()))  # 至少保留5条
            if k >= values.numel():
                threshold_val = values.min()
            else:
                threshold_val = torch.kthvalue(values, values.numel() - k + 1).values
            mask = S_hat_sub >= threshold_val
            mask = mask | mask.t()  # 对称化
        else:
            mask = torch.zeros_like(S_hat_sub, dtype=torch.bool)

        rows, cols = torch.nonzero(mask, as_tuple=True)
        full_adj_i = torch.zeros_like(ensemble_adj)
        full_adj_i[sub_nodes[rows], sub_nodes[cols]] = S_hat_sub[rows, cols]
        ensemble_adj += full_adj_i

#ensemble_adj = ensemble_adj / len(models)

# === Step 3: 自动调融合权重 λ 并评估 ===
orig_adj_dense = to_dense_adj(data_full.edge_index.to(primary_device), max_num_nodes=N)[0]

def train_and_evaluate(edge_index, edge_weight, x_full, y_full,
                       train_mask, val_mask, test_mask,
                       pos_weight, device, model_save_path,
                       epochs=500, lr=0.01, weight_decay=5e-4,
                       max_patience=30):
    model = FullGraphGNN(in_dim=x_full.size(1), hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_val_aupr = 0.0
    patience = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_full, edge_index, edge_weight)
        loss = criterion(out[train_mask], y_full[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(x_full, edge_index, edge_weight)
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

    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path))

    # 验证集选阈值
    model.eval()
    with torch.no_grad():
        val_out = model(x_full, edge_index, edge_weight)
        val_probs = torch.sigmoid(val_out[val_mask]).cpu().numpy()
        val_labels = y_full[val_mask].cpu().numpy()

        # 替换原来的 F1 阈值搜索
        best_aupr_metric = 0.0
        best_thr = 0.5

        if val_labels.sum() > 0:
            precision, recall, thresholds = precision_recall_curve(val_labels, val_probs)
            # 计算每个阈值对应的 F1 和 AUPR 局部指标
            for i, thr in enumerate(thresholds):
                # 只考虑 Recall >= 0.6 的区域（AUPR 敏感区）
                if recall[i] >= 0.60:
                    f1_local = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)
                    # 用 F1 作为代理，但限制在高 Recall 区
                    if f1_local > best_aupr_metric:
                        best_aupr_metric = f1_local
                        best_thr = thr
            # 如果没找到，回退到 0.5
            if best_aupr_metric == 0.0:
                best_thr = 0.5
        else:
            best_thr = 0.5

        # 测试
        test_out = model(x_full, edge_index, edge_weight)
        test_probs = torch.sigmoid(test_out[test_mask]).cpu().numpy()
        test_labels = y_full[test_mask].cpu().numpy()

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
results = []

x_full = data_full.x.to(primary_device)
y_full = data_full.y.float().to(primary_device)
train_mask = data_full.train_mask
val_mask = data_full.val_mask
test_mask = data_full.test_mask

# 计算 pos_weight（用于 BCEWithLogitsLoss）
train_labels = y_full[train_mask]
pos_weight = (1 - train_labels.mean()) / train_labels.mean()


results = []

# =============== 实验 1：原始图（baseline） ===============
print("\n" + "="*50)
print("🧪 Experiment 1: Original Graph (Baseline)")
print("="*50)

edge_index_orig, edge_weight_orig = dense_to_sparse(orig_adj_dense)
edge_index_orig = edge_index_orig.to(primary_device)
edge_weight_orig = edge_weight_orig.to(primary_device)

# 训练函数（后面定义）
metrics_orig = train_and_evaluate(
    edge_index_orig, None,
    x_full, y_full, train_mask, val_mask, test_mask,
    pos_weight, primary_device,
    model_save_path='best_model_orig.pth'
)

results.append({
    'Method': 'Original Graph',
    'Val_Best_AUPR': metrics_orig['val_aupr'],
    'Test_Acc': metrics_orig['acc'],
    'Test_AUC': metrics_orig['auc'],
    'Test_AUPR': metrics_orig['aupr'],
    'Test_F1': metrics_orig['f1'],
    'Test_Precision': metrics_orig['prec'],
    'Test_Recall': metrics_orig['rec'],
    'Best_Threshold': metrics_orig['best_thr']
})

print(f"✅ Original Graph | Test AUPR: {metrics_orig['aupr']:.4f} | F1: {metrics_orig['f1']:.4f}")

# =============== 实验 2：精炼图（Refined Graph） ===============
print("\n" + "="*50)
print("🧪 Experiment 2: Refined Graph (Original + Top New Edges from Ensemble)")
print("="*50)

# 先用原始图训练一个快速分类器，获取全图置信度
quick_model = FullGraphGNN(in_dim=x_full.size(1), hidden_dim=128).to(primary_device)
quick_opt = torch.optim.Adam(quick_model.parameters(), lr=0.01, weight_decay=5e-4)
quick_criterion = FocalLoss()

for _ in range(100):  # 快速训练
    quick_model.train()
    quick_opt.zero_grad()
    out = quick_model(x_full, edge_index_orig)
    loss = quick_criterion(out[train_mask], y_full[train_mask])
    loss.backward()
    quick_opt.step()

quick_model.eval()
with torch.no_grad():
    quick_logits = quick_model(x_full, edge_index_orig)
    quick_pred = torch.sigmoid(quick_logits) > 0.5  # 二值预测
    same_label_mask = quick_pred.unsqueeze(0) == quick_pred.unsqueeze(1)  # 同预测标签

# 图精炼
fused_adj = orig_adj_dense.clone()
candidate_mask = orig_adj_dense < 0.01  # 放宽阈值
# 只保留：集成图中强边 + 同预测标签
#valid_new = (ensemble_adj > 0.1) & candidate_mask & same_label_mask.to(primary_device)
valid_new = (ensemble_adj > 0.23) & candidate_mask

num_new = int(valid_new.sum().item())
if num_new > 8000:
    scores = ensemble_adj * valid_new.float()
    threshold = torch.kthvalue(scores.view(-1), scores.numel() - 8000).values
    final_mask = (scores >= threshold) & valid_new
else:
    final_mask = valid_new

fused_adj = orig_adj_dense.clone()
fused_adj[final_mask] = ensemble_adj[final_mask]
print(f"✅ Added {final_mask.sum().item()} high-quality semantic-consistent edges.")

# 转稀疏（忽略 edge_weight，因 SAGE 不用）
edge_index_refined, edge_weight_refined = dense_to_sparse(fused_adj)
edge_index_refined = edge_index_refined.to(primary_device)
edge_weight_refined = edge_weight_refined.to(primary_device)  # 虽然不用，但定义它
# 训练评估
metrics_refined = train_and_evaluate(
    edge_index_refined, None,
    x_full, y_full, train_mask, val_mask, test_mask,
    pos_weight, primary_device,
    model_save_path='best_model_refined.pth'
)
results.append({
    'Method': 'Refined Graph',
    'Val_Best_AUPR': metrics_refined['val_aupr'],
    'Test_Acc': metrics_refined['acc'],
    'Test_AUC': metrics_refined['auc'],
    'Test_AUPR': metrics_refined['aupr'],
    'Test_F1': metrics_refined['f1'],
    'Test_Precision': metrics_refined['prec'],
    'Test_Recall': metrics_refined['rec'],
    'Best_Threshold': metrics_refined['best_thr']
})

print(f"✅ Refined Graph | Test AUPR: {metrics_refined['aupr']:.4f} | F1: {metrics_refined['f1']:.4f}")
# =============== 实验 3：Ensemble Prediction (Original + Refined) ===============
print("\n" + "="*50)
print("🧪 Experiment 3: Ensemble Prediction (Original + Refined Models)")
print("="*50)

# 加载两个训练好的模型
model_orig = FullGraphGNN(in_dim=x_full.size(1), hidden_dim=128).to(primary_device)
model_orig.load_state_dict(torch.load('best_model_orig.pth', map_location=primary_device))
model_orig.eval()

model_refined = FullGraphGNN(in_dim=x_full.size(1), hidden_dim=128).to(primary_device)
model_refined.load_state_dict(torch.load('best_model_refined.pth', map_location=primary_device))
model_refined.eval()

# 获取验证集 logits 用于选阈值
with torch.no_grad():
    val_out_orig = model_orig(x_full, edge_index_orig)[val_mask]
    val_out_refined = model_refined(x_full, edge_index_refined)[val_mask]
    val_ensemble_logits = (val_out_orig + val_out_refined) / 2.0
    val_probs_ens = torch.sigmoid(val_ensemble_logits).cpu().numpy()
    val_labels_np = y_full[val_mask].cpu().numpy()

# 在验证集上选择 AUPR 友好阈值（Recall >= 0.6）
best_ens_thr = 0.5
if val_labels_np.sum() > 0:
    precision, recall, thresholds = precision_recall_curve(val_labels_np, val_probs_ens)
    best_metric = 0.0
    for i, thr in enumerate(thresholds):
        if recall[i] >= 0.60:  # 关注高 Recall 区域
            f1_local = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)
            if f1_local > best_metric:
                best_metric = f1_local
                best_ens_thr = thr
    if best_metric == 0.0:  # 回退
        best_ens_thr = 0.5

# 测试集预测
with torch.no_grad():
    test_out_orig = model_orig(x_full, edge_index_orig)[test_mask]
    test_out_refined = model_refined(x_full, edge_index_refined)[test_mask]
    test_ensemble_logits = (test_out_orig + test_out_refined) / 2.0
    test_probs_ens = torch.sigmoid(test_ensemble_logits).cpu().numpy()
    test_labels_np = y_full[test_mask].cpu().numpy()

# 评估
pred_ens = (test_probs_ens > best_ens_thr).astype(int)
acc = accuracy_score(test_labels_np, pred_ens)
auc_score = roc_auc_score(test_labels_np, test_probs_ens) if len(np.unique(test_labels_np)) > 1 else 0.5
precision, recall, _ = precision_recall_curve(test_labels_np, test_probs_ens)
aupr_score = auc(recall, precision)
f1 = f1_score(test_labels_np, pred_ens)
prec = precision_score(test_labels_np, pred_ens)
rec = recall_score(test_labels_np, pred_ens)

# 保存结果
results.append({
    'Method': 'Ensemble (Orig+Refined)',
    'Val_Best_AUPR': 0.0,  # 未单独训练，无法准确获取
    'Test_Acc': acc,
    'Test_AUC': auc_score,
    'Test_AUPR': aupr_score,
    'Test_F1': f1,
    'Test_Precision': prec,
    'Test_Recall': rec,
    'Best_Threshold': best_ens_thr
})

print(f"✅ Ensemble Prediction | Test AUPR: {aupr_score:.4f} | F1: {f1:.4f} | Recall: {rec:.4f}")

# =============== 打印最终结果 ===============
print("\n" + "="*80)
print("📊 FINAL COMPARISON")
print("="*80)
df = pd.DataFrame(results)
print(df.to_string(index=False, float_format="%.4f"))

best_row = df.loc[df['Test_AUPR'].idxmax()]
print(f"\n🏆 Best Method: {best_row['Method']}, Test AUPR = {best_row['Test_AUPR']:.4f}")

# 可选：排序后打印
print("\n" + "="*80)
print("📊 FINAL RESULTS (sorted by Test AUPR)")
print("="*80)
df_sorted = df.sort_values('Test_AUPR', ascending=False)
print(df_sorted.to_string(index=False, float_format="%.4f"))

# ==============================
# 🧪 t-SNE Visualization
# ==============================
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def extract_embeddings(model, x, edge_index, device):
    """提取模型最后一层之前的节点嵌入（即分类前的 hidden 表示）"""
    model.eval()
    with torch.no_grad():
        # GAT 分支
        x1 = model.gat(x, edge_index)  # [N, hidden_dim//2]
        # SAGE 分支
        x2 = model.sage(x, edge_index)  # [N, hidden_dim//2]
        x = torch.cat([x1, x2], dim=-1)  # [N, hidden_dim]
        x = F.relu(model.bn1(x))
        x = model.dropout(x)
        x = model.conv2(x, edge_index)  # [N, hidden_dim]
        x = F.relu(model.bn2(x))
        return x.cpu().numpy()

print("\n" + "="*50)
print("🎨 t-SNE Visualization of Refined Graph Embeddings")
print("="*50)

# 使用 refined 模型提取嵌入
embeddings = extract_embeddings(model_refined, x_full, edge_index_refined, primary_device)

# 仅可视化测试集节点（避免数据泄露）
test_mask_tensor = test_mask if isinstance(test_mask, torch.Tensor) else torch.tensor(test_mask)
test_indices = torch.where(test_mask_tensor)[0].cpu().numpy()
test_embeddings = embeddings[test_indices]
test_labels = y_full[test_mask].cpu().numpy()

# t-SNE 降维
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca')
embeddings_2d = tsne.fit_transform(test_embeddings)

# 绘图：分别绘制两类
plt.figure(figsize=(8, 6))

# 分离两类索引
idx_pan_cancer = test_labels == 1
idx_non_pan_cancer = test_labels == 0

plt.scatter(
    embeddings_2d[idx_pan_cancer, 0],
    embeddings_2d[idx_pan_cancer, 1],
    c='red', s=15, alpha=0.8, label='Cancer Gene'
)

plt.scatter(
    embeddings_2d[idx_non_pan_cancer, 0],
    embeddings_2d[idx_non_pan_cancer, 1],
    c='blue', s=15, alpha=0.8, label='Non-cancer Gene'
)

plt.legend(title="类别", fontsize=12, title_fontsize=12)
plt.title('t-SNE of Node Embeddings (Refined Graph) - Test Set', fontsize=14)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.tight_layout()
plt.savefig('tsne_refined_graph.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ t-SNE plot saved as 'tsne_refined_graph.png'")