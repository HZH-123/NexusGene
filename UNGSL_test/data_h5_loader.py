import numpy as np
import h5py
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data

def read_h5file(path, network_name='network', feature_name='features'):
    with h5py.File(path, 'r') as f:
        network = f[network_name][:]
        features = f[feature_name][:]
        y_train = f['y_train'][:]
        gene_names = f['gene_names'][:]
        y_test = f['y_test'][:]
        if 'y_val' in f:
            y_val = f['y_val'][:]
        else:
            y_val = None
        y_train_val = np.logical_or(y_train, y_val)
        y_train_val_test = np.logical_or(y_train_val, y_test)
        train_mask = f['mask_train'][:]
        test_mask = f['mask_test'][:]
        if 'mask_val' in f:
            val_mask = f['mask_val'][:]
        else:
            val_mask = None
        adj = torch.tensor(network).nonzero().t()
        scaler = StandardScaler()
        features = torch.FloatTensor(scaler.fit_transform(features))
        y = []
        for i in y_train_val_test:
            y.append(i.item())
        y=torch.tensor(y)
        train_mask=train_mask.astype(bool)
        val_mask=val_mask.astype(bool)
        test_mask=test_mask.astype(bool)
    return Data(
        x=features,
        y=y,
        edge_index=adj,
        train_mask=train_mask,
        test_mask=test_mask,
        val_mask=val_mask,
        gene_names=gene_names
    )








