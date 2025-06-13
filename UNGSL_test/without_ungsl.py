from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from UNGSL_test.cluster import create_cluster
from UNGSL_test.data_h5_loader import read_h5file
import shap

torch.manual_seed(42)
np.random.seed(42)
data = read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/STRINGdb_multiomics.h5")  # string-db
#data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/IREF_multiomics.h5")#IREF v9
#data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/CPDB_multiomics.h5")#CPDB
#data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/MULTINET_multiomics.h5")#MULTINET
#data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/PCNET_multiomics.h5")#PCNET  lr:0.0005 dropout:0.4
#异质
#data=read_h5file(r'/root/autodl-tmp/UNGSL/UNGSL_test/network/MTG_multiomics.h5')#MTG   threshold=0.5
#data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/LTG_multiomics.h5")#LTG
primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = data.to(primary_device)
data.y = data.y.float()

config = {
    'gsl_hidden': 256,
    'cls_hidden': 32,
    'lr': 0.003,
    'dropout': 0.8,
    'epochs': 500,
    'threshold': 0.7,
    'alpha': 0.8,
    'reg_weight': 1e-4,
    'pretrain_lr': 1e-4
}


# 增强版图结构学习模块
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
        # 混合原始图结构信息
        current_device = x.device
        adj = original_adj.to(current_device)
        sim = config['alpha'] * sim + (1 - config['alpha']) * adj
        return sim


class EnhancedClassifier(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.conv1 = GCNConv(in_channel, hidden_channel)
        self.bn1 = nn.LayerNorm(hidden_channel)
        self.conv2 = GCNConv(hidden_channel, hidden_channel)
        self.conv3 = GCNConv(hidden_channel, hidden_channel)
        self.dropout = config['dropout']
        self.conv4 = GCNConv(hidden_channel, out_channel)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index)
        return x.squeeze(-1)


def pretrain_epoch(pretrain_gsl, pretrain_cls, pretrain_optim, criterion, original_adj, data):
    pretrain_gsl.train()
    pretrain_cls.train()
    pretrain_optim.zero_grad()
    # 生成原始邻接矩阵
    S = pretrain_gsl(data.x, original_adj)
    edge_index = torch.nonzero(S > config['threshold']).t()
    # 分类任务
    out = pretrain_cls(data.x, edge_index)
    targets = data.y[data.train_mask].float()
    loss = criterion(out[data.train_mask], targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pretrain_gsl.parameters(), max_norm=2)
    torch.nn.utils.clip_grad_norm_(pretrain_cls.parameters(), max_norm=2)
    pretrain_optim.step()
    return loss.item()


def finetune_epoch_without_ungsl(criterion, model_gsl, model_cls, epoch, finetune_optim, original_adj, data):
    model_gsl.train()
    model_cls.train()
    finetune_optim.zero_grad()
    # 生成原始邻接矩阵
    S = model_gsl(data.x, original_adj)
    edge_index = torch.nonzero(S > config['threshold']).t()
    # 分类任务
    out = model_cls(data.x, edge_index)
    targets = data.y[data.train_mask].float()
    loss = criterion(out[data.train_mask], targets)
    loss.backward()
    finetune_optim.step()
    return loss.item()


def test_without_ungsl(model_gsl, model_cls, original_adj, data):
    model_gsl.eval()
    model_cls.eval()
    with torch.no_grad():
        # 使用学习到的图结构
        sim = model_gsl(data.x, original_adj)
        edge_index = (sim > config['threshold']).nonzero().t()
        out = model_cls(data.x, edge_index)
        pred = (torch.sigmoid(out) > 0.5).long()
        acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
        pred_probs = torch.sigmoid(out[data.test_mask])
        test_labels = data.y[data.test_mask].cpu().numpy()
        test_pred_probs = pred_probs.cpu().numpy()
        # 计算AUC
        auc_score = roc_auc_score(test_labels, test_pred_probs)
        # 计算AUPR
        precision, recall, _ = precision_recall_curve(test_labels, test_pred_probs)
        aupr_score = auc(recall, precision)
    return acc.item(), auc_score, aupr_score


def train_model_without_ungsl_single_fold(data, train_idx, test_idx):

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask[test_idx] = True

    train_x = data.x[data.train_mask].cpu().numpy()
    train_y = data.y[data.train_mask].cpu().numpy()
    smote = SMOTE()
    train_x_resampled, train_y_resampled = smote.fit_resample(train_x, train_y)
    new_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    for idx in torch.where(data.train_mask)[0].cpu().numpy():
        if data.y[idx].cpu().numpy() in train_y_resampled:
            new_train_mask[idx] = True
    data.train_mask = new_train_mask

    data.y = data.y.float()
    print("Stage 1: Pretraining GSL and Classifier without UNGSL")
    original_adj = torch.zeros((data.num_nodes, data.num_nodes), device=primary_device)
    original_adj[data.edge_index[0], data.edge_index[1]] = 1

    # 初始化模型（不使用UnGSL）
    pretrain_gsl = EnhancedGSL(data.num_features, config['gsl_hidden'])
    pretrain_cls = EnhancedClassifier(data.num_features, config['cls_hidden'], 1)
    pretrain_gsl = pretrain_gsl.to(primary_device)
    pretrain_cls = pretrain_cls.to(primary_device)

    pretrain_optim = torch.optim.AdamW([
        {'params': pretrain_gsl.parameters(), 'lr': 1e-4, 'weight_decay': 5e-4},
        {'params': pretrain_cls.parameters(), 'lr': 0.01, 'weight_decay': 1e-4}
    ])
    torch.nn.utils.clip_grad_norm_(pretrain_gsl.parameters(), max_norm=1.0)
    criterion = BCEWithLogitsLoss()

    # 预训练循环
    for epoch in range(1, 101):
        loss = pretrain_epoch(pretrain_gsl, pretrain_cls, pretrain_optim, criterion, original_adj, data)
        if epoch % 10 == 0:
            print(f'Pretrain Epoch: {epoch:03d}, Loss: {loss:.4f}')
        del loss
        torch.cuda.empty_cache()
    print("\nStage 2: Finetuning without UNGSL")
    # 初始化微调模型（加载预训练参数）
    model_gsl = EnhancedGSL(data.num_features, config['gsl_hidden'])
    model_gsl.load_state_dict(pretrain_gsl.state_dict())
    model_cls = EnhancedClassifier(data.num_features, config['cls_hidden'], 1)
    model_cls.load_state_dict(pretrain_cls.state_dict())
    model_gsl = model_gsl.to(primary_device)
    model_cls = model_cls.to(primary_device)
    finetune_optim = torch.optim.AdamW([
        {'params': model_gsl.parameters(), 'lr': 1e-4},
        {'params': model_cls.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)
    # 微调循环
    best_acc = 0
    for epoch in range(1, 201):
        loss = finetune_epoch_without_ungsl(criterion, model_gsl, model_cls, epoch, finetune_optim, original_adj, data)
        if epoch % 10 == 0:
            with torch.no_grad():
                acc, _, _ = test_without_ungsl(model_gsl, model_cls, original_adj, data)
            if acc > best_acc:
                best_acc = acc
            print(f'Finetune Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
        del loss
        torch.cuda.empty_cache()
    print(f'Best Test Accuracy: {best_acc:.4f}')
    return model_gsl, model_cls


def perform_10_fold_cv_without_ungsl(data):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_models = []

    for fold, (train_index, test_index) in enumerate(kf.split(data.x[data.train_mask])):
        print(f"Fold {fold + 1}")
        model_gsl, model_cls = train_model_without_ungsl_single_fold(data, train_index, test_index)
        all_models.append((model_gsl, model_cls))

    return all_models
def train_model_without_ungsl(data):
    data=data.to(primary_device)
    indices = torch.randperm(data.num_nodes)
    train_idx = indices[:int(0.8 * data.num_nodes)]
    test_idx = indices[int(0.8 * data.num_nodes):]
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask[test_idx] = True
    data.y = data.y.float()
    print("Stage 1: Pretraining GSL and Classifier without UNGSL")
    original_adj = torch.zeros((data.num_nodes, data.num_nodes), device=primary_device)
    original_adj[data.edge_index[0], data.edge_index[1]] = 1

    # 初始化模型（不使用UnGSL）
    pretrain_gsl = EnhancedGSL(data.num_features, config['gsl_hidden'])
    pretrain_cls = EnhancedClassifier(data.num_features, config['cls_hidden'], 1)
    pretrain_gsl = pretrain_gsl.to(primary_device)
    pretrain_cls = pretrain_cls.to(primary_device)

    pretrain_optim = torch.optim.AdamW([
        {'params': pretrain_gsl.parameters(), 'lr': 5e-4, 'weight_decay': 5e-4},#5e-4
        {'params': pretrain_cls.parameters(), 'lr': 1, 'weight_decay': 1e-4}#1e-3
    ])
    torch.nn.utils.clip_grad_norm_(pretrain_gsl.parameters(), max_norm=1.0)
    criterion = BCEWithLogitsLoss()

    # 预训练循环
    for epoch in range(1, 101):
        loss = pretrain_epoch(pretrain_gsl, pretrain_cls, pretrain_optim, criterion, original_adj, data)
        if epoch % 10 == 0:
            print(f'Pretrain Epoch: {epoch:03d}, Loss: {loss:.4f}')

    print("\nStage 2: Finetuning without UNGSL")
    # 初始化微调模型（加载预训练参数）
    model_gsl = EnhancedGSL(data.num_features, config['gsl_hidden'])
    model_gsl.load_state_dict(pretrain_gsl.state_dict())
    model_cls = EnhancedClassifier(data.num_features, config['cls_hidden'], 1)
    model_cls.load_state_dict(pretrain_cls.state_dict())
    model_gsl = model_gsl.to(primary_device)
    model_cls = model_cls.to(primary_device)
    finetune_optim = torch.optim.AdamW([
        {'params': model_gsl.parameters(), 'lr': 1e-4},
        {'params': model_cls.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)
    # 微调循环
    best_acc = 0
    for epoch in range(1, 201):
        loss = finetune_epoch_without_ungsl(criterion, model_gsl, model_cls, epoch, finetune_optim, original_adj, data)
        if epoch % 10 == 0:
            acc, _, _ = test_without_ungsl(model_gsl, model_cls, original_adj, data)
            if acc > best_acc:
                best_acc = acc
            print(f'Finetune Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
    print(f'Best Test Accuracy: {best_acc:.4f}')
    return model_gsl, model_cls

def train_on_data_with_trained_models(data, all_models):
    model_gsl_data, model_cls_data = train_model_without_ungsl(data)
    model_cls_data=model_cls_data.to(primary_device)
    model_gsl_data = model_gsl_data.to(primary_device)
    data.x = data.x.to(primary_device)
    original_adj = torch.zeros((data.num_nodes, data.num_nodes), device=primary_device)
    original_adj[data.edge_index[0], data.edge_index[1]] = 1
    outs = []
    for model_gsl, model_cls in all_models:
        # 将模型移动到 primary_device
        model_gsl = model_gsl.to(primary_device)
        model_cls = model_cls.to(primary_device)

        model_gsl.eval()
        model_cls.eval()
        with torch.no_grad():
            adj = model_gsl(data.x, original_adj)
            edge_index = (adj > config['threshold']).nonzero().t()
            out = model_cls_data(data.x, edge_index)
            outs.append(out.cpu().detach().numpy())

    outs = np.array(outs)
    outs_mean = np.sum(outs, axis=0) / len(all_models)
    outs_mean = torch.tensor(outs_mean).to(primary_device)

    pred = (torch.sigmoid(outs_mean) > 0.5).long()
    data.y=data.y.to(primary_device)
    acc = (pred == data.y).float().mean()
    pred_probs = torch.sigmoid(outs_mean)
    test_labels = data.y.cpu().numpy()
    test_pred_probs = pred_probs.cpu().numpy()

    # 计算AUC
    auc_score = roc_auc_score(test_labels, test_pred_probs)
    # 计算AUPR
    precision, recall, _ = precision_recall_curve(test_labels, test_pred_probs)
    aupr_score = auc(recall, precision)

    print(f"在 data 上的评估结果 - acc:{acc:.4f}, auc:{auc_score:.4f}, aupr:{aupr_score:.4f}")
    return acc, auc_score, aupr_score


dataset = create_cluster(data.cpu())
all_models = []
for sub_data in dataset:
    sub_data = sub_data.to(primary_device)
    indices = torch.randperm(sub_data.num_nodes)
    train_idx = indices[:int(0.8 * sub_data.num_nodes)]
    test_idx = indices[int(0.8 * sub_data.num_nodes):]
    sub_data.train_mask = torch.zeros(sub_data.num_nodes, dtype=torch.bool)
    sub_data.test_mask = torch.zeros(sub_data.num_nodes, dtype=torch.bool)
    sub_data.train_mask[train_idx] = True
    sub_data.test_mask[test_idx] = True
    models = perform_10_fold_cv_without_ungsl(sub_data)
    all_models.extend(models)
# 使用训练好的模型在 data 上进行训练和评估
torch.cuda.empty_cache()
train_on_data_with_trained_models(data, all_models)
