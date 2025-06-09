import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from UNGSL_test import data_loader

def create_cluster(data,k=4):
    embedding=data.x[data.train_mask].detach().cpu().numpy()
    kmeans = KMeans(n_clusters=k, random_state=0)
    k_labels=kmeans.fit_predict(embedding)
    dataset=[]
    for exclude in range(k):
        mask=(k_labels!=exclude)
        mask=torch.tensor(mask)
        sub_nodes=torch.where(mask)[0]
        sub_edge_index,_=subgraph(sub_nodes,data.edge_index,relabel_nodes=True)
        # sub_data=Data(x=data.x[mask],edge_index=sub_edge_index,y=data.y[mask],mask=data.mask[mask])
        sub_data = Data(x=data.x[data.train_mask][mask], edge_index=sub_edge_index, y=data.y[data.train_mask][mask])
        dataset.append(sub_data)
    return dataset

# data = data_loader.load_graph_data('/root/autodl-tmp/UNGSL/UNGSL_test/PathNet/dataset_PathNet.pkl')
# dataset=create_cluster(data)
# for data in dataset:
#     print(data.edge_index)
