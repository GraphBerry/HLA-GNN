import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from model_gcn import GCN


BASE_PATH = os.path.dirname(os.getcwd())
print("BASE_PATH: ", BASE_PATH)
sys.path.insert(0, BASE_PATH)

from dgl_dataset.hla_dataset import build_graph, split_dataset
from dgl_dataset.utils import EarlyStopper, calc_mean_sd


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args, split_id):
    # load and preprocess dataset
    g, features, labels, n_classes = build_graph(args.dataset)
    in_feats = features.shape[1]
    n_edges = g.number_of_edges()
    n_nodes= g.number_of_nodes()

    train_idx, valid_idx, test_idx = split_dataset(args.dataset, split_id)

    train_mask = torch.zeros(n_nodes, dtype=bool)
    val_mask = torch.zeros(n_nodes, dtype=bool)
    test_mask = torch.zeros(n_nodes, dtype=bool)
    train_mask[train_idx] = True
    val_mask[valid_idx] = True
    test_mask[test_idx] = True

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)
        train_mask = train_mask.to(args.gpu)
        val_mask = val_mask.to(args.gpu)
        test_mask = test_mask.to(args.gpu)
        features = features.to(args.gpu)
        labels = labels.to(args.gpu)


    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g, in_feats,args.n_hidden, n_classes,
                args.n_layers, F.relu, args.dropout)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    stopper = EarlyStopper(patience=30, dt_name=args.dataset)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        val_acc = evaluate(model, features, labels, val_mask)
        if stopper.step(val_acc, epoch, model):
            break
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             val_acc, n_edges / np.mean(dur) / 1000))

    print()
    model.eval()
    stopper.load_checkpoint(model)
    test_acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.4}".format(test_acc))
    
    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    # parser.add_argument("--split_id", type=int, default=0,
    #                     help="dataset split id [0-9]")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=256,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    results = []
    for split_id in range(10):
        res = main(args, split_id)
        results.append(res * 100)
    
    print(results)
    calc_mean_sd(results)
