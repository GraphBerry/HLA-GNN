import torch
import numpy as np
import hashlib
import datetime
from dgl.data import FraudDataset
from sklearn import metrics, preprocessing
from collections import namedtuple

import torch
import os
import datetime
import numpy as np


class EarlyStopperWithLoss:
    def __init__(self, patience=30, dt_name='default'):
        dt = datetime.datetime.now()
        self._filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self._save_dir = os.path.join('checkpoints', dt_name)

        os.makedirs(self._save_dir, exist_ok=True)
        self.save_path = os.path.join(self._save_dir, self._filename)
        print(f"[{self.__class__.__name__}]: Saving model to {self.save_path}")

        self.patience = patience
        self.counter = 0
        self.best_epoch = -1
        self.best_loss = -1
        self.best_result = -1
        self.early_stop = False

    def step(self, epoch, loss, result, model):
        if self.best_loss is None:
            self.best_epoch = epoch
            self.best_result = result
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (result < self.best_result):
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (result >= self.best_result):
                self.save_checkpoint(model)
            self.best_epoch = epoch
            self.best_loss = np.min((loss, self.best_loss))
            self.best_result = np.max((result, self.best_result))
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.save_path)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.save_path))


class EarlyStopper:
    def __init__(self, patience=30, dt_name='default'):
        dt = datetime.datetime.now()
        self._filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self._save_dir = os.path.join('checkpoints', dt_name)

        os.makedirs(self._save_dir, exist_ok=True)
        self.save_path = os.path.join(self._save_dir, self._filename)
        print(f"[{self.__class__.__name__}]: Saving model to {self.save_path}")

        self.patience = patience
        self.counter = 0
        self.best_ep = -1
        self.best_score = -1
        self.early_stop = False

    def step(self, acc, epoch, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.best_ep = epoch
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_ep = epoch
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.save_path)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.save_path))


def normalize(feats, train_nid, dtype=np.float32):
    """
        Row-normalize sparse matrix.
        Code from https://github.com/PonderLY/PC-GNN/blob/24372e340810b9ed68cd6b71789f7e265eb6ab7c/src/utils.py#L59
    """
    train_feats = feats[train_nid]
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    return feats.astype(dtype)


def calc_loss(tr_logits, tr_labels, weighted=True):
    weight = None
    if weighted:
        _, cnt = torch.unique(tr_labels, return_counts=True)
        weight = 1 / cnt
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    return loss_fn(tr_logits, tr_labels)


def calc_acc(y_true, y_pred):
    """
    Compute the accuracy of prediction given the labels.
    """
    # return (y_pred == y_true).sum() * 1.0 / len(y_pred)
    return metrics.accuracy_score(y_true, y_pred)


def calc_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5


def prob2pred(y_prob, thres=0.5):
    """
    Convert probability to predicted results according to given threshold
    :param y_prob: numpy array of probability in [0, 1]
    :param thres: binary classification threshold, default 0.5
    :returns: the predicted result with the same shape as y_prob
    """
    y_pred = np.zeros_like(y_prob, dtype=np.int32)
    y_pred[y_prob >= thres] = 1
    y_pred[y_prob < thres] = 0
    return y_pred


def to_numpy(x):
    # Convert tensor on the gpu to tensor of the cpu.
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    return x.cpu().numpy() if x.is_cuda else x.numpy()


def convert_probs(labels, probs, threshold_moving=True, thres=0.5):
    labels = to_numpy(labels)
    probs = torch.nn.Sigmoid()(probs)
    probs = to_numpy(probs)
    probs_1 = probs[:, 1]
    if threshold_moving:
        preds = prob2pred(probs_1, thres=thres)
    else:
        preds = probs.argmax(axis=1)

    return labels, probs_1, preds


def calc_mean_sd(results):
    results = np.around(results, decimals=5)
    MEAN = np.mean(results, axis=0)
    # PSD = np.std(results, axis=0)
    SSD = np.std(results, axis=0, ddof=1)

    # print(MEAN, PSD, SSD)
    metric_name = ['accuracy']
    for i, name in enumerate(metric_name):
        print("{}= {:3.2f}±{:3.2f}".format(name, MEAN, SSD))
