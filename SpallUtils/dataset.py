import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
import numpy as np
import pandas as pd
import os
import pickle as pkl


class SpiderDataset(InMemoryDataset):
    def __init__(self, root, raw_data_root,transform=None, pre_transform=None, force_reload=True):
        self.raw_data_root = raw_data_root
        super(SpiderDataset, self).__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['dataset.dat']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        PIK = os.path.join(self.raw_data_root, 'dataset.dat')
        with open(PIK, "rb") as f:
            objects = pkl.load(f)

        p_data_train, real_data, p_data_test, p_label_train, real_label, p_label_test, ori_idx_train, ori_idx_test = tuple(objects)
    
        all_datas_train = np.array(pd.concat([p_data_train, real_data]))
        all_datas_test = np.array(p_data_test)
        all_labels_train = np.array(pd.concat([p_label_train, real_label]))
        all_labels_test = np.array(p_label_test)

        features = np.vstack((all_datas_train, all_datas_test))
        labels = np.vstack([all_labels_train, all_labels_test])

        M = len(p_data_train)
        idx_train = range(M)
        idx_pred = range(M, len(all_labels_train))
        idx_test = range(len(all_labels_train), len(all_labels_train) + len(all_labels_test))

        train_mask = self.sample_mask(idx_train, labels.shape[0])
        pred_mask = self.sample_mask(idx_pred, labels.shape[0])
        test_mask = self.sample_mask(idx_test, labels.shape[0])

        y_train = self.get_binary_labels(labels, train_mask)
        y_pred = self.get_binary_labels(labels, pred_mask)
        y_test = self.get_binary_labels(labels, test_mask)


        adj = np.load(os.path.join(self.raw_data_root, 'adjcent.npy'))
        np.fill_diagonal(adj, 1)
        adj[adj < 0.2] = 0
        edge_index, edge_weight = dense_to_sparse(torch.tensor(adj))
        
        data_list = [
            Data(
            x = torch.from_numpy(features).float(), 
            edge_index = edge_index, 
            edge_attr = edge_weight.float(),
            y = torch.from_numpy(labels).float(), 
            train_mask = train_mask, 
            test_mask = test_mask,
            pred_mask = pred_mask,
            y_train = torch.from_numpy(y_train).float(),
            y_pred = torch.from_numpy(y_pred).float(),
            y_test = torch.from_numpy(y_test).float(),
            train_idx = ori_idx_train,
            test_idx = ori_idx_test,
            ),
        ]
        self.save(data_list, self.processed_paths[0])

    def sample_mask(self, idx, l):
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool_)

    def get_binary_labels(self, labels, mask):
        labels_binary = np.zeros(labels.shape)
        labels_binary[mask, :] = labels[mask, :]
        return labels_binary
    

if __name__ == "__main__":
    data_type = "PDAC-A"
    data_root = f"./data/{data_type}/torch_data"
    raw_data_root = f"./data/{data_type}/Infor_Data"
    dataset = SpiderDataset(root=data_root, raw_data_root=raw_data_root)
    print(dataset[0])
    print(dataset[0].pred_mask)