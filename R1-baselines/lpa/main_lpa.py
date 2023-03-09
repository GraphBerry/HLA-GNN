import os
import sys
import argparse
import torch
import dgl
import numpy as np

from model_lpa import LabelPropagation
BASE_PATH = os.path.dirname(os.getcwd())
print("BASE_PATH: ", BASE_PATH)
sys.path.insert(0, BASE_PATH)

from dgl_dataset.hla_dataset import build_graph, split_dataset
from dgl_dataset.utils import EarlyStopper, calc_mean_sd


def prepare_data(args, split_id=0):
    # load and preprocess dataset
    g, feats, labels, n_classes = build_graph(args.dataset)
    n_nodes= g.number_of_nodes()

    train_idx, valid_idx, test_idx = split_dataset(args.dataset, split_id)

    train_mask = torch.zeros(n_nodes, dtype=bool)
    val_mask = torch.zeros(n_nodes, dtype=bool)
    test_mask = torch.zeros(n_nodes, dtype=bool)
    train_mask[train_idx] = True
    val_mask[valid_idx] = True
    test_mask[test_idx] = True

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    print(g)

    return g, feats, labels, train_mask, val_mask, test_mask, n_classes


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def main(args, split_id=0):
    # check cuda
    device = f'cuda:{args.gpu}' if torch.cuda.is_available(
    ) and args.gpu >= 0 else 'cpu'

    # load data
    g, _, labels, train_mask, val_mask, test_mask, n_classes = prepare_data(args, split_id)

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze().to(device)
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze().to(device)

    g = g.to(device)
    labels = labels.to(device)

    # label propagation
    lp = LabelPropagation(args.num_layers, args.alpha)
    logits = lp(g, labels, mask=train_idx)

    val_acc = accuracy(logits[val_mask], labels[val_mask])
    test_acc = accuracy(logits[test_mask], labels[test_mask])

    print(f'VALID ACC={val_acc:.4f}, TEST ACC={test_acc:.4f}')
    return test_acc


if __name__ == '__main__':
    """
    Label Propagation Hyperparameters
    """
    parser = argparse.ArgumentParser(description='LP')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)

    args = parser.parse_args()
    print(args)

    results = []

    for i in range(10):
        res = main(args, split_id=i)
        results.append(res * 100)
    
    print(results)
    calc_mean_sd(results)
