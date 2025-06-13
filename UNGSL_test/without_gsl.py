import torch
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from torch.nn import BCEWithLogitsLoss

from torch_geometric.nn import GCNConv
from imblearn.over_sampling import SMOTE

import torch.nn as nn
import torch.nn.functional as F

from UNGSL_test.cluster import create_cluster
from UNGSL_test.data_h5_loader import read_h5file

torch.manual_seed(42)
np.random.seed(42)




#data = read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/STRINGdb_multiomics.h5")  # string-db
#data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/IREF_multiomics.h5")#IREF v9
#data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/CPDB_multiomics.h5")#CPDB
#data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/MULTINET_multiomics.h5")#MULTINET
data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/PCNET_multiomics.h5")#PCNET  lr:0.0005 dropout:0.4
#异质
#data=read_h5file(r'/root/autodl-tmp/UNGSL/UNGSL_test/network/MTG_multiomics.h5')#MTG   threshold=0.5
#data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/LTG_multiomics.h5")#LTG
primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = data.to(primary_device)
data.y = data.y.float()

dataset = create_cluster(data.cpu())

config = {
    'cls_hidden': 128,
    'lr': 0.003,
    'dropout': 0.8,
    'epochs': 500,
    'threshold': 0.4,
    'alpha': 0.8,
    'reg_weight': 1e-4,
    'pretrain_lr': 1e-4
}


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


def train_model_without_gsl_single_fold(data, train_idx, test_idx):
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
    print("Stage 1: Pretraining Classifier without GSL")
    original_adj = torch.zeros((data.num_nodes, data.num_nodes), device=primary_device)
    original_adj[data.edge_index[0], data.edge_index[1]] = 1

    model_cls = EnhancedClassifier(data.num_features, config['cls_hidden'], 1)
    model_cls = model_cls.to(primary_device)
    pretrain_optim = torch.optim.AdamW([
        {'params': model_cls.parameters(), 'lr': 1, 'weight_decay': 1e-4}
    ])
    torch.nn.utils.clip_grad_norm_(model_cls.parameters(), max_norm=1.0)
    criterion = BCEWithLogitsLoss()

    for epoch in range(1, 151):
        model_cls.train()
        pretrain_optim.zero_grad()
        edge_index = data.edge_index
        out = model_cls(data.x, edge_index)
        targets = data.y[data.train_mask].float()
        loss = criterion(out[data.train_mask], targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_cls.parameters(), max_norm=2)
        pretrain_optim.step()
        if epoch % 10 == 0:
            print(f'Pretrain Epoch: {epoch:03d}, Loss: {loss:.4f}')

    print("\nStage 2: Finetuning without GSL")
    best_acc = 0
    for epoch in range(1, 201):
        model_cls.train()
        pretrain_optim.zero_grad()
        edge_index = data.edge_index
        out = model_cls(data.x, edge_index)
        targets = data.y[data.train_mask].float()
        loss = criterion(out[data.train_mask], targets)
        loss.backward()
        pretrain_optim.step()
        if epoch % 10 == 0:
            acc, _, _ = test_without_gsl(model_cls, data)
            if acc > best_acc:
                best_acc = acc
            print(f'Finetune Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
    print(f'Best Test Accuracy: {best_acc:.4f}')
    return model_cls


def test_without_gsl(model_cls, data):
    model_cls.eval()
    with torch.no_grad():
        edge_index = data.edge_index
        out = model_cls(data.x, edge_index)
        pred = (torch.sigmoid(out) > 0.5).long()
        acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        pred_probs = torch.sigmoid(out[data.test_mask])
        test_labels = data.y[data.test_mask].cpu().numpy()
        test_pred_probs = pred_probs.cpu().numpy()
        auc_score = roc_auc_score(test_labels, test_pred_probs)
        precision, recall, _ = precision_recall_curve(test_labels, test_pred_probs)
        aupr_score = auc(recall, precision)
    return acc.item(), auc_score, aupr_score


def perform_10_fold_cv_without_gsl(data):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_models = []

    for fold, (train_index, test_index) in enumerate(kf.split(data.x)):
        print(f"Fold {fold + 1}")
        model_cls = train_model_without_gsl_single_fold(data, train_index, test_index)
        all_models.append(model_cls)

    return all_models


def integrate_models(all_models, data):

    outs = []
    for model_cls in all_models:


        out = model_cls(data.x, data.edge_index)
        outs.append(out.cpu().detach().numpy())
    outs = np.array(outs)
    outs_mean = np.sum(outs, axis=0) / len(all_models)
    outs_mean = torch.tensor(outs_mean)

    pred = (torch.sigmoid(outs_mean) > 0.5).long()
    pred = pred.to(primary_device)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    pred_probs = torch.sigmoid(outs_mean[data.test_mask])
    test_labels = data.y[data.test_mask].cpu().numpy()
    test_pred_probs = pred_probs.cpu().numpy()

    # 计算AUC
    auc_score = roc_auc_score(test_labels, test_pred_probs)
    # 计算AUPR
    precision, recall, _ = precision_recall_curve(test_labels, test_pred_probs)
    aupr_score = auc(recall, precision)

    print(f"集成学习结果 - acc:{acc:.4f}, auc:{auc_score:.4f}, aupr:{aupr_score:.4f}")
    return acc, auc_score, aupr_score


# 对数据集中的每个数据样本执行十折交叉验证和集成学习
all_results = []
for data in dataset:
    data = data.to(primary_device)
    all_models = perform_10_fold_cv_without_gsl(data)
    # acc, auc_score, aupr_score = integrate_models(all_models, data)
    # all_results.append((acc, auc_score, aupr_score))
acc_score, auc_score, aupr_score = integrate_models(all_models, data)
print(f"所有数据样本的平均结果 - acc:{acc_score:.4f}, auc:{auc_score:.4f}, aupr:{aupr_score:.4f}")


