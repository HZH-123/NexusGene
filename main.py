
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.nn import BCEWithLogitsLoss
from torch_geometric.nn import GCNConv
from UNGSL_test.cluster import create_cluster
from UNGSL_test.data_h5_loader import read_h5file
torch.manual_seed(42)
np.random.seed(42)
data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/STRINGdb_multiomics.h5")
primary_device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data=data.to(primary_device)
data.y = data.y.float()
gene_names=data.gene_names
dataset=create_cluster(data.cpu())
config = {
    'gsl_hidden': 256,
    'cls_hidden': 128,
    'dropout': 0.8,#0.7
    'threshold': 0.4,#0.4
    'alpha': 0.8,
    'reg_weight': 0.5
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

    def forward(self, x,original_adj):
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=1)
        sim = torch.mm(x, x.t())
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
        self.conv3 = GCNConv(hidden_channel, hidden_channel)  # 新增一层卷积层
        self.dropout = config['dropout']
        self.conv4 = GCNConv(hidden_channel, out_channel)
    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))  # 新增层的前向传播
        x = F.dropout(x, p=self.dropout, training=self.training)
        x=self.conv4(x,edge_index)
        return x.squeeze(-1)
# 不确定性感知层
class UNGSLayer(nn.Module):
    def __init__(self, num_nodes,zeta,beta):
        super().__init__()
        # 基于节点度初始化
        self.beta = beta
        self.zeta = zeta
        self.l = nn.Parameter(torch.ones(num_nodes) * 0.5)
    def act(self,x):
        mask=(x>0).float()
        act=mask*self.zeta*nn.functional.sigmoid(x)+(1-mask)*self.beta-self.l
        return act
    def forward(self, s, confidence):
        # 基于置信度调整相似度
        x = torch.exp(-confidence.unsqueeze(0)) - self.l.unsqueeze(1)
        x=self.act(x)
        adj = torch.sigmoid(s) * x
        adj = adj + adj.t()  # 保持对称性
        return adj
def compute_confidence(logits):
    prob = torch.sigmoid(logits)
    entropy = - (prob * torch.log(prob + 1e-10) + (1-prob) * torch.log(1-prob + 1e-10))
    confidence = 1.0 - entropy
    return confidence.clamp(0.0, 1.0)
def pretrain_epoch(pretrain_gsl,pretrain_cls,pretrain_optim,criterion,original_adj,data):
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
    #torch.cuda.empty_cache()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pretrain_gsl.parameters(), max_norm=2)
    torch.nn.utils.clip_grad_norm_(pretrain_cls.parameters(), max_norm=2)
    pretrain_optim.step()
    return loss.item()
def train_model(data):
    #data=data.to(primary_device)
    indices = torch.randperm(data.num_nodes)
    train_idx = indices[:int(0.8 * data.num_nodes)]
    test_idx = indices[int(0.8 * data.num_nodes):]
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask[test_idx] = True
    data.y = data.y.float()
    print("Stage 1: Pretraining GSL and Classifier")
    original_adj = torch.zeros((data.num_nodes, data.num_nodes), device=primary_device)
    original_adj[data.edge_index[0], data.edge_index[1]] = 1
    # 初始化模型（不使用UnGSL）
    pretrain_gsl = EnhancedGSL(data.num_features, config['gsl_hidden'])
    pretrain_cls = EnhancedClassifier(data.num_features, config['cls_hidden'], 1)
    pretrain_gsl = pretrain_gsl.to(primary_device)
    pretrain_cls = pretrain_cls.to(primary_device)

    pretrain_optim = torch.optim.AdamW([
        {'params': pretrain_gsl.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3},#1e-4
        {'params': pretrain_cls.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}#1e-3
    ])
    torch.nn.utils.clip_grad_norm_(pretrain_gsl.parameters(), max_norm=1.0)
    criterion=BCEWithLogitsLoss()
    # 预训练循环
    for epoch in range(1, 101):
        loss = pretrain_epoch(pretrain_gsl,pretrain_cls,pretrain_optim,criterion,original_adj,data)
        if epoch % 10 == 0:
            print(f'Pretrain Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("\nStage 2: Finetuning with UnGSL")
    # 初始化微调模型（加载预训练参数）
    model_gsl = EnhancedGSL(data.num_features, config['gsl_hidden'])
    model_gsl.load_state_dict(pretrain_gsl.state_dict())
    model_cls = EnhancedClassifier(data.num_features, config['cls_hidden'], 1)
    model_cls.load_state_dict(pretrain_cls.state_dict())
    model_ungsl = UNGSLayer(data.num_nodes, zeta=0.5, beta=0.3)
    model_gsl = model_gsl.to(primary_device)
    model_cls = model_cls.to(primary_device)
    model_ungsl = model_ungsl.to(primary_device)
    model_cls.requires_grad_(False)  # 冻结分类器参数
    finetune_optim = torch.optim.AdamW([
        {'params': model_gsl.parameters(), 'lr': 1e-2},
        {'params': model_ungsl.parameters(), 'lr': 5e-2}
    ], weight_decay=1e-5)
    # 微调循环
    best_acc = 0
    for epoch in range(1, 201):
        loss = finetune_epoch(criterion,model_gsl,model_cls,model_ungsl,epoch,finetune_optim,original_adj,data)
        if epoch % 10 == 0:
            acc, _, _ = test(model_gsl,model_cls,model_ungsl,original_adj,data)
            if acc > best_acc:
                best_acc = acc
            print(f'Finetune Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
    print(f'Best Test Accuracy: {best_acc:.4f}')
    return model_ungsl, model_cls,model_gsl
def train_model_single_fold(data,train_idx,test_idx):
    #data=data.to(primary_device)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask[test_idx] = True

    data.y = data.y.float()
    print("Stage 1: Pretraining GSL and Classifier")
    original_adj = torch.zeros((data.num_nodes, data.num_nodes), device=primary_device)
    original_adj[data.edge_index[0], data.edge_index[1]] = 1
    # 初始化模型（不使用UnGSL）
    pretrain_gsl = EnhancedGSL(data.num_features, config['gsl_hidden'])
    pretrain_cls = EnhancedClassifier(data.num_features, config['cls_hidden'], 1)
    pretrain_gsl = pretrain_gsl.to(primary_device)
    pretrain_cls = pretrain_cls.to(primary_device)

    pretrain_optim = torch.optim.AdamW([
        {'params': pretrain_gsl.parameters(), 'lr': 5e-4, 'weight_decay': 5e-4},
        {'params': pretrain_cls.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4}
    ])
    torch.nn.utils.clip_grad_norm_(pretrain_gsl.parameters(), max_norm=1.0)
    criterion=BCEWithLogitsLoss()
    # 预训练循环
    for epoch in range(1, 101):
        loss = pretrain_epoch(pretrain_gsl,pretrain_cls,pretrain_optim,criterion,original_adj,data)
        if epoch % 10 == 0:
            print(f'Pretrain Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("\nStage 2: Finetuning with UnGSL")
    # 初始化微调模型（加载预训练参数）
    model_gsl = EnhancedGSL(data.num_features, config['gsl_hidden'])
    model_gsl.load_state_dict(pretrain_gsl.state_dict())
    model_cls = EnhancedClassifier(data.num_features, config['cls_hidden'], 1)
    model_cls.load_state_dict(pretrain_cls.state_dict())
    model_ungsl = UNGSLayer(data.num_nodes, zeta=0.5, beta=0.01)
    model_gsl = model_gsl.to(primary_device)
    model_cls = model_cls.to(primary_device)
    model_ungsl = model_ungsl.to(primary_device)
    model_cls.requires_grad_(False)  # 冻结分类器参数
    finetune_optim = torch.optim.AdamW([
        {'params': model_gsl.parameters(), 'lr': 1e-4},
        {'params': model_ungsl.parameters(), 'lr': 5e-4}
    ], weight_decay=1e-5)
    # 微调循环
    best_acc = 0
    for epoch in range(1, 201):
        loss = finetune_epoch(criterion,model_gsl,model_cls,model_ungsl,epoch,finetune_optim,original_adj,data)
        if epoch % 10 == 0:
            acc, _, _ = test(model_gsl,model_cls,model_ungsl,original_adj,data)
            if acc > best_acc:
                best_acc = acc
            print(f'Finetune Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
    print(f'Best Test Accuracy: {best_acc:.4f}')
    return model_ungsl, model_cls,model_gsl
def finetune_epoch(criterion,model_gsl,model_cls,model_ungsl,epoch,finetune_optim,original_adj,data):
    model_gsl.train()
    model_ungsl.train()
    model_cls.conv3.requires_grad_(True)
    if epoch == 100:
        model_cls.requires_grad_(True)  # 第二阶段解冻分类器
        finetune_optim.add_param_group({'params': model_cls.parameters()})
    finetune_optim.zero_grad()
    # 生成原始邻接矩阵
    S = model_gsl(data.x,original_adj)
    # 使用预训练分类器计算置信度（不计算梯度）
    with torch.no_grad():
        cls_out = model_cls(data.x, data.edge_index)
        if isinstance(cls_out, tuple):
            cls_out = cls_out[0]
        confidence=compute_confidence(cls_out)
    # 应用UnGSL调整
    S_hat = model_ungsl(S, confidence)
    edge_index = torch.nonzero(S_hat > config['threshold']).t()
    # 使用冻结分类器进行预测
    out = model_cls(data.x, edge_index)
    targets = data.y[data.train_mask].float()
    # 损失计算（仅包含正则项）
    reg_loss = torch.norm(S_hat, p=1)
    reg_weight = config['reg_weight'] * (1 + epoch / 500)
    total_loss = criterion(out[data.train_mask], targets) + reg_weight * reg_loss
    #torch.cuda.empty_cache()
    total_loss.backward()
    finetune_optim.step()
    return total_loss.item()
def test(model_gsl,model_cls,model_ungsl,original_adj,data):
    model_gsl.eval()
    model_ungsl.eval()
    model_cls.eval()
    with torch.no_grad():
        # 使用学习到的图结构
        sim = model_gsl(data.x,original_adj)
        cls_out = model_cls(data.x, data.edge_index)
        confidence=compute_confidence(cls_out)
        adj = model_ungsl(sim,confidence)
        edge_index=(adj>config['threshold']).nonzero().t()
        out = model_cls(data.x, edge_index)

        pred = (torch.sigmoid(out) > 0.5).long()
        acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        pred_probs = torch.sigmoid(out[data.test_mask])
        test_labels = data.y[data.test_mask].cpu().numpy()
        test_pred_probs = pred_probs.cpu().numpy()
        #smooth_pred_probs=(test_pred_probs+np.mean(test_pred_probs))/2
        # 计算AUC
        auc_score = roc_auc_score(test_labels, test_pred_probs)
        # 计算AUPR
        precision, recall, _ = precision_recall_curve(test_labels, test_pred_probs)
        aupr_score = auc(recall, precision)
    return acc.item(),auc_score,aupr_score

def perform_5_fold_cv(data):
    # train_idx,test_idx=s_train_test_split(data,0.9)
    indices = torch.randperm(data.num_nodes)
    train_idx = indices[:int(0.8 * data.num_nodes)]
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_models = []
    for fold, (train_fold_idx, val_fold_idx) in enumerate(kf.split(train_idx)):
        print(f"Fold {fold + 1}")
        train_fold_full_idx = train_idx[train_fold_idx]
        val_fold_full_idx = train_idx[val_fold_idx]
        model_gsl, model_cls, model_ungsl = train_model_single_fold(data, train_fold_full_idx, val_fold_full_idx)
        all_models.append((model_gsl, model_cls, model_ungsl))
    return all_models

models=[]
for data in dataset:
    data=data.to(primary_device)
    model_ungsl, model_cls,model_gsl =train_model(data)
    models.append((model_ungsl, model_cls, model_gsl))
model_ungsl,model_cls_data,model_gsl=train_model(data)
original_adj = torch.zeros((data.num_nodes, data.num_nodes), device=primary_device)
original_adj[data.edge_index[0], data.edge_index[1]] = 1
outs = []
for model in models:
    _, model_cls,_ = model
    sim = model_gsl(data.x, original_adj)
    cls_out = model_cls(data.x, data.edge_index)
    confidence = compute_confidence(cls_out)
    adj = model_ungsl(sim, confidence)
    edge_index = (adj > config['threshold']).nonzero().t()
    out = model_cls(data.x, edge_index)
    outs.append(out.cpu().detach().numpy())
outs=np.array(outs)
outs_mean=np.sum(outs,axis=0)/len(models)
outs_mean=torch.tensor(outs_mean)
pred = (torch.sigmoid(outs_mean) > 0.5).long()
pred=pred.to(primary_device)
acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
pred_probs = torch.sigmoid(outs_mean[data.test_mask])
test_labels = data.y[data.test_mask].cpu().numpy()
test_pred_probs = pred_probs.cpu().numpy()
# 计算AUC
auc_score = roc_auc_score(test_labels, test_pred_probs)
# 计算AUPR
precision, recall, _ = precision_recall_curve(test_labels, test_pred_probs)
aupr_score = auc(recall, precision)
print(f"acc:{acc:.4f},auc:{auc_score:.4f},aupr:{aupr_score:.4f}")