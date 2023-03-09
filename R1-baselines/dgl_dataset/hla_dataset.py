"""Low Homophily Graphs
"""
import os
import numpy as np

import dgl
import torch

# PRINT ROOT DIR
CURRENT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.abspath(CURRENT_DIR), 'data_geom')
print(f"[{__file__}] DATA_DIR: {DATA_DIR}")

FILE_LIST = ['cora', 'citeseer', 'pubmed', 'texas', 'chameleon',
             'squirrel', 'wisconsin', 'cornell']


def build_graph(dataset):
    dataset = dataset.lower()
    assert dataset in FILE_LIST, f"only support dataset in {FILE_LIST}"

    edge_path = os.path.join(DATA_DIR, f"{dataset}/{dataset}.edge")
    feature_path = os.path.join(DATA_DIR, f"{dataset}/{dataset}.feature")
    label_path = os.path.join(DATA_DIR, f"{dataset}/{dataset}.label")

    edges = np.loadtxt(edge_path, dtype=int)
    feat = np.loadtxt(feature_path, dtype=np.float32)
    labels = np.loadtxt(label_path, dtype=int)
    n_classes = labels.max() + 1

    # debug mode
    # print(edges.shape, edges.min(), edges.max())
    # print(feat.shape)
    # print(labels.shape, labels.min(), labels.max())

    # 构建dgl graph, 并转换为无向图
    graph = dgl.graph((torch.LongTensor(edges[:, 0]),
                       torch.LongTensor(edges[:, 1])))
    g = dgl.to_bidirected(graph)

    feat = torch.from_numpy(feat)
    labels = torch.from_numpy(labels)

    return graph, feat, labels, n_classes


def split_dataset(dataset, split_id=None):
    if split_id is None:
        split_id = ''
    split_path = os.path.join(DATA_DIR, f"{dataset}")
    idx_train = np.loadtxt(os.path.join(
        split_path, f"{split_id}train.txt"), dtype=int)
    idx_valid = np.loadtxt(os.path.join(
        split_path, f"{split_id}val.txt"), dtype=int)
    idx_test = np.loadtxt(os.path.join(
        split_path, f"{split_id}test.txt"), dtype=int)

    idx_train = torch.from_numpy(idx_train)
    idx_valid = torch.from_numpy(idx_valid)
    idx_test = torch.from_numpy(idx_test)

    return idx_train, idx_valid, idx_test


if __name__ == "__main__":
    graph, feat, labels, n_classes = build_graph('texas')
    print(n_classes)
    idx_train, idx_valid, idx_test = split_dataset('texas', 0)
    print(idx_train, idx_valid, idx_test)
    
    # for dataset in FILE_LIST:
    #     print(dataset)
    #     build_graph(dataset)
    #     idx_train, idx_valid, idx_test = split_dataset(dataset, 0)
    #     print(idx_train.shape, idx_valid.shape, idx_test.shape)
    #     print(idx_train, idx_valid, idx_test)
    #     print("\n")
