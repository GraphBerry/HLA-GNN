import os
import sys

import torch
from torch import nn
import torch.nn.functional as fn
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
from tqdm import tqdm
import time

# CURR_PATH = os.path.dirname(__file__)
BASE_PATH = os.path.dirname(os.getcwd())
print("BASE_PATH: ", BASE_PATH)
sys.path.insert(0, BASE_PATH)

from dgl_dataset.utils import EarlyStopper, calc_mean_sd
from dgl_dataset.hla_dataset import build_graph, split_dataset

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def evaluate(model, data_loader, device='cpu'):
    model.to(device)
    model.eval()
    prob_list = []
    labels_list = []
    with torch.no_grad():
        for step, (feats, labels) in enumerate(tqdm(data_loader)):
            feats = feats.to(device)
            labels = labels.to(device)
            logits = model(feats)
            probs = torch.nn.Sigmoid()(logits)
            prob_list.extend(probs.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())

    labels = torch.LongTensor(labels_list)
    probs = torch.tensor(prob_list)

    return accuracy(probs, labels)


class MLP(nn.Module):
    def __init__(self, n_feats, n_hidden, n_classes=2):
        super(MLP, self).__init__()
        # yelp
        self.layers = nn.ModuleList([nn.Linear(n_feats, n_hidden),
                                     nn.Linear(n_hidden, n_classes)])
        # amazon
        # self.layers = nn.ModuleList([nn.Linear(n_feats, n_classes)])

    def forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = fn.relu(h)
        return h

    def init_weight(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform(param)


class CustomDataset(Dataset):
    def __init__(self, feats, labels):
        super(CustomDataset, self).__init__()
        self.data = feats
        self.label = labels

    def __getitem__(self, index):
        feats = self.data[index]
        labels = self.label[index]
        return feats, labels

    def __len__(self):
        return len(self.data)


def train(train_set, val_set, test_set, dataset_name, n_feats, n_classes,
          epochs=30, batch_size=256, device='cpu'):

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             drop_last=False, num_workers=0)

    # 建立模型
    model = MLP(n_feats=n_feats, n_hidden=256, n_classes=n_classes)
    model.to(device)

    # 选择优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-3)

    loss_fcn = torch.nn.CrossEntropyLoss()

    stopper = EarlyStopper(patience=30, dt_name=dataset_name)

    cnt_time = 0
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}, AVG time(s)={cnt_time / epoch:>.3f}")
        start = time.time()
        model.train()
        full_loss = 0
        for batch_idx, (batch_feats, batch_labels) in enumerate(tqdm(train_loader)):
            batch_feats = batch_feats.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_feats)
            loss = loss_fcn(logits, batch_labels)
            full_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end = time.time()
        cnt_time += end - start

        if epoch % 1 == 0:
            print("loss={:.4f}".format(epoch, full_loss / len(train_loader)))
            val_acc = evaluate(model, val_loader, device=device)
            print("Evaluation results: acc_valid={val_results}")
            if stopper.step(val_acc, epoch, model):
                break

    model.eval()
    stopper.load_checkpoint(model)
    print(f"\nBest epoch: {stopper.best_ep}, Final test results:\n")

    acc_test = evaluate(model, test_loader, device=device)

    return acc_test


if __name__ == "__main__":
    dataset_name = 'pubmed'
    _, feats, labels, n_classes = build_graph(dataset_name)
    n_feats = feats.shape[1]
    print(feats.dtype, labels.dtype)

    results = []

    for i in range(10):
        train_nid, val_nid, test_nid = split_dataset(dataset_name, split_id=i)
        # 处理数据集
        dataset = CustomDataset(feats, labels)
        train_set = Subset(dataset, train_nid)
        val_set = Subset(dataset, val_nid)
        test_set = Subset(dataset, test_nid)
        res = train(train_set, val_set, test_set, dataset_name=dataset_name, n_feats=n_feats, 
                    n_classes=n_classes, epochs=200, device='cuda:0')
        results.append(res.item() * 100)
    
    print(results)
    calc_mean_sd(results)
