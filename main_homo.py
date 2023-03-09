from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from models_homo import MLP, HLAGNN
from sklearn.metrics import f1_score
import os
import argparse
from config import Config
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(args, config):
    adj, sadj = load_graph_homo(config)
    features, labels, idx_train, idx_val, idx_test = load_data(config)

    adj = torch.tensor(adj.todense(), dtype=torch.float32)

    print(f"feature_info={type(features), features.shape}")
    print(f"label_info={type(labels), labels.shape}")
    print(f"train_size={idx_train.shape}, val_size={idx_val.shape}, "
          f"test_size={idx_test.shape}")

    model_MLP = MLP(n_feat=config.fdim,
                    n_hid=config.nhid2,
                    nclass=config.class_num,
                    dropout=config.dropout)

    optimizer_mlp = optim.Adam(model_MLP.parameters(),
                               lr=config.lr,
                               weight_decay=0.02)
    mlp_acc_val_best = 0
    mlp_best_test = 0

    # 增加计时
    start = time.time()

    # 预训练soft labels
    for i in range(10):
        model_MLP.train()
        optimizer_mlp.zero_grad()

        output = model_MLP(features)
        loss = F.nll_loss(output[idx_train], labels[idx_train])
        acc = accuracy(output[idx_train], labels[idx_train])

        loss.backward()
        optimizer_mlp.step()

        model_MLP.eval()
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        if acc_val > mlp_acc_val_best:
            mlp_acc_val_best = acc_val
            mlp_best_test = acc_test

        print(f"epoch: {i+1:4d}, loss: {loss.item(): .4f}, acc: train={acc.item(): .4f} "
              f"valid={acc_val.item(): .4f}  test={acc_test.item(): .4f}")
    
    print(f"best acc={mlp_best_test:.4f}")

    si_adj = adj.clone()
    bi_adj = adj.mm(adj)
    # 邻接矩阵改良，k-hop neighbors
    bi_adj = si_adj + bi_adj
    bi_adj[bi_adj > 0] = 1

    # 测试1-hop neighbors
    # bi_adj = si_adj

    feat_dim = config.fdim
    idx_train_unmasked = idx_train
    feat = features

    if args.use_labels:
        print("use labels")
        # 若使用标记信息，需要增加标记onehot向量的长度
        feat_dim = config.fdim + config.class_num

    model_HLAGNN = HLAGNN(nfeat=feat_dim,
                          adj=adj,
                          nhid1=config.nhid1,
                          nhid2=config.nhid2,
                          nclass=config.class_num,
                          n_nodes=config.n_nodes,
                          dropout=config.dropout)

    optimizer_HLAGNN = optim.Adam(model_HLAGNN.parameters(),
                                  lr=config.lr,
                                  weight_decay=config.weight_decay)

    best_acc_val_HLAGNN = 0
    best_f1 = 0
    best = 0
    best_test = 0

    for i in range(config.epochs):
        model_HLAGNN.train()
        model_MLP.train()

        optimizer_HLAGNN.zero_grad()
        optimizer_mlp.zero_grad()

        # 每一轮添加新的标记信息
        if args.use_labels:
            mask = torch.rand(idx_train.shape) < args.mask_rate
            idx_train_masked = idx_train[mask]
            idx_train_unmasked = idx_train[~mask]
            feat = add_labels(features, labels, config.class_num, idx_train_masked)

        # 输出soft label和预测结果
        output = model_MLP(features)
        out, adj_mask, emb = model_HLAGNN(feat, si_adj, bi_adj, output)

        # 计算loss
        loss_mlp = F.nll_loss(output[idx_train_unmasked], labels[idx_train_unmasked])
        loss = loss_mlp + F.nll_loss(out[idx_train_unmasked], labels[idx_train_unmasked])

        acc = accuracy(out[idx_train_unmasked], labels[idx_train_unmasked])
        loss.backward()
        optimizer_HLAGNN.step()
        optimizer_mlp.step()

        # 评估模型
        model_HLAGNN.eval()
        model_MLP.eval()

        # 此处重新添加label feature
        if args.use_labels:
            feat = add_labels(features, labels, config.class_num, idx_train)
        
        out, adj_mask, emb = model_HLAGNN(feat, si_adj, bi_adj, output)
        acc_val = accuracy(out[idx_val], labels[idx_val])
        acc_test = accuracy(out[idx_test], labels[idx_test])

        if acc_val > best_acc_val_HLAGNN:
            best_acc_val_HLAGNN = acc_val
            best_test = acc_test

        if best < acc_test:
            best = acc_test

        print(f"epoch: {i+1:4d}, loss: {loss.item()-loss_mlp.item(): .4f}, acc: train={acc.item(): .4f} "
              f"valid={acc_val.item(): .4f}  test={acc_test.item(): .4f}")

    end = time.time()

    print(f"Elapsed Time={end - start}(s)")
    print("Best accuracy:", best_test.item())


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset",
                       default="cora", type=str, required=False)
    parse.add_argument("-m", "--mask_rate",
                       help="masked labeled data for training", default=0.5, type=float, required=False)
    parse.add_argument("--use_labels",
                       help="use labels for propagation, default: false", action="store_true", required=False)

    args = parse.parse_args()
    config_file = "./config/" + str(args.dataset) + ".ini"
    config = Config(config_file)

    # 乱七八糟，到处都有flag配置，后面需要统一
    cuda = True
    use_seed = True  # True

    if use_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if cuda:
            torch.cuda.manual_seed(config.seed)

    # 打印配置文件
    print(vars(config))

    train(args, config)
