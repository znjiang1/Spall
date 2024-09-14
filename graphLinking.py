import pandas as pd
import os

import numpy as np
import pandas as pd
import random
from warnings import simplefilter
import pickle as pkl
from sklearn.model_selection import train_test_split
from scipy import sparse as sp
import argparse
simplefilter(action='ignore', category=FutureWarning)


def l2norm(mat):
    stat = np.sqrt(np.sum(mat**2, axis=1))
    cols = mat.columns
    mat[cols] = mat[cols].div(stat, axis=0)
    mat[np.isinf(mat)] = 0
    mat = mat.fillna(0)
    return mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Graph Linking")
    parser.add_argument("--data_root", type=str, default="./data/PDAC-A")
    args = parser.parse_args()
    data_root = args.data_root
    outputdir = f"{data_root}/Infor_Data"
    dataDir = f"{data_root}/Datadir"


    count_data1 = pd.read_csv(os.path.join(outputdir, 'ST_count/ST_count_1.csv'), index_col=0)
    count_data2 = pd.read_csv(os.path.join(outputdir, 'ST_count/ST_count_2.csv'), index_col=0)


    cell_embedding = pd.read_csv(os.path.join(outputdir, 'integrated.csv'), index_col=0)
    LabelsPath1 = os.path.join(outputdir, 'ST_label/ST_label_1.csv')
    LabelsPath2 = os.path.join(outputdir, 'ST_label/ST_label_2.csv')

    norm_embedding = cell_embedding
    norm_count_data1 = l2norm(count_data1)
    norm_count_data2 = l2norm(count_data2)

    embedding_spots1 = norm_count_data1
    embedding_spots2 = norm_count_data2
    
    p_data=embedding_spots1
    real_data=embedding_spots2
    p_label = pd.read_csv(LabelsPath1, header=0, index_col=0, sep=',')
    real_label = pd.read_csv(LabelsPath2, header=0, index_col=0, sep=',')

    # split the p_data into training and testing sets
    random.seed(123)
    indices = p_data.index.to_numpy()

    p_data_train, p_data_test, p_label_train, p_label_test, idx_train, idx_test = train_test_split(
    p_data, p_label, indices, test_size=0.2, random_state=42
    )

    # save objects
    PIK = os.path.join(outputdir, 'dataset.dat')
    res = [
        p_data_train, real_data, p_data_test, p_label_train, real_label, p_label_test, idx_train, idx_test
        ]
    with open(PIK, "wb") as f:
        pkl.dump(res, f)
    print('save data succesfully....')


    #concat p_data_train and real_data
    all_data_train = pd.concat([p_data_train, real_data])
    #concat p_label_train and real_label
    all_label_train = pd.concat([p_label_train, real_label])

    datas_train = np.array(all_data_train)
    datas_test = np.array(p_data_test)
    labels_train = np.array(all_label_train)
    labels_test = np.array(p_label_test)

    features = np.vstack((datas_train, datas_test))


    import SpiderUtils.RPTree as RPTree
    NumOfTrees = 10
    adj = sp.coo_matrix(np.zeros((features.shape[0], features.shape[0]), dtype=np.float32))
    for r in range(NumOfTrees):
        tree = RPTree.BinaryTree(features)
        features_index = np.arange(features.shape[0])
        print(f"constructing tree {r}")
        tree_root = tree.construct_tree(tree, features_index)
        print(f"tree {r} is constructed")
        # get the indices of points in leaves
        leaves_array = tree_root.get_leaf_nodes()

        # connect points in the same leaf node
        edgeList = []
        for i in range(len(leaves_array)):
            x = leaves_array[i]
            n = x.size
            perm = np.empty((n, n, 2), dtype=x.dtype)
            perm[..., 0] = x[:, None]
            perm[..., 1] = x
            perm1 = np.reshape(perm, (-1, 2))
            if i == 0:
                edgeList = perm1
            else:
                edgeList = np.vstack((edgeList, perm1))

        # assign one as edge weight
        edgeList = edgeList[edgeList[:, 0] != edgeList[:, 1]]
        edgeList = np.hstack((edgeList, np.ones((edgeList.shape[0], 1), dtype=int)))

        # convert edges list to adjacency matrix
        shape = tuple(edgeList.max(axis=0)[:2] + 1)
        adjMatRPTree = sp.coo_matrix((edgeList[:, 2], (edgeList[:, 0], edgeList[:, 1])), shape=shape,
                                        dtype=edgeList.dtype)

        # an adjacency matrix holding weights accumulated from all rpTrees
        adj = adj + (adjMatRPTree / NumOfTrees)

    adj=adj.todense()
    np.save(os.path.join(outputdir, 'adjcent.npy'),adj)

