import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch_geometric.nn import GCNConv
from UNGSL_test.data_h5_loader import read_h5file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)
#data = read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/STRINGdb_multiomics.h5")  # string-db
#data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/IREF_multiomics.h5")#IREF v9
#data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/CPDB_multiomics.h5")#CPDB
#data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/MULTINET_multiomics.h5")#MULTINET
#data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/PCNET_multiomics.h5")#PCNET  lr:0.0005 dropout:0.4
#异质
#data=read_h5file(r'/root/autodl-tmp/UNGSL/UNGSL_test/network/MTG_multiomics.h5')#MTG   threshold=0.5
data=read_h5file(r"/root/autodl-tmp/UNGSL/UNGSL_test/network/LTG_multiomics.h5")#LTG
data = data.to(device)
data.y = data.y.float()
class EnhancedClassifier(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.conv1 = GCNConv(in_channel, hidden_channel)
        self.bn1 = nn.LayerNorm(hidden_channel)
        self.conv2 = GCNConv(hidden_channel, hidden_channel)
        self.conv3 = GCNConv(hidden_channel, hidden_channel)  # 新增一层卷积层
        self.dropout = 0.8
        self.conv4 = GCNConv(hidden_channel, out_channel)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))  # 新增层的前向传播
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index)
        return x.squeeze(-1)

model = EnhancedClassifier(data.num_features, 128, 1).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.05)
criterion = BCEWithLogitsLoss()
epochs = 200

def train():
    model.train()
    optim.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optim.step()
    return loss.item()


for epoch in range(epochs):
    loss = train()
    if epoch % 10 == 0:
        print(f"epoch:{epoch},loss: {loss}")
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred_probs = torch.sigmoid(out)
    pred = (pred_probs > 0.5).float()
    acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    test_labels = data.y[data.test_mask].cpu().numpy()
    test_pred_probs = pred_probs[data.test_mask].detach().cpu().numpy()
    # 计算AUC
    auc_score = roc_auc_score(test_labels, test_pred_probs)
    # 计算AUPR
    precision, recall, _ = precision_recall_curve(test_labels, test_pred_probs)
    aupr_score = auc(recall, precision)
    print(f"acc:{acc:.4f},auc:{auc_score:.4f},aupr:{aupr_score:.4f}")
test()



