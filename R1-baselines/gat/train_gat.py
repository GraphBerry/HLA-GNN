import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from model_gat import GAT


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
    heads = ([args.num_heads] * (args.num_layers-1)) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                in_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    print(model)

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
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    args = parser.parse_args()
    print(args)

    results = []
    for split_id in range(10):
        res = main(args, split_id)
        results.append(res * 100)
    
    print(results)
    calc_mean_sd(results)
