import math

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GraphConvolution_homo(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, adj, out_features, bias=True):
        super(GraphConvolution_homo, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_bi = Parameter(
            torch.FloatTensor(in_features, out_features))
        self.w = Parameter(torch.FloatTensor(1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # 这是论文中公式11的T
        self.LPA_weight = Parameter(adj.clone())
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        stdv_bi = 1. / math.sqrt(self.weight_bi.size(1))
        self.weight_bi.data.uniform_(-stdv_bi, stdv_bi)
        self.w.data.uniform_(0.5, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, bi_adj, output):
        """
            input: node feature matrix
            adj: original adjacency matrix
            bi_adj: A_k
            output: soft labels
        """
        # output是MLP的输出，下面两行对应公式（4）
        output = output.exp()
        
        # 求出, 即相似度
        sim_matrix = torch.matmul(output, output.t())

        # 公式（10）A_k*H
        bi_adj = torch.mul(bi_adj, sim_matrix)

        # 公式（10）\hat{D}^{-1}
        with torch.no_grad():
            bi_row_sum = torch.sum(bi_adj, dim=1, keepdim=True)
            # np.power(rowsum, -1).flatten()
            bi_r_inv = torch.pow(bi_row_sum, -1).flatten()
            bi_r_inv[torch.isinf(bi_r_inv)] = 0.
            bi_r_mat_inv = torch.diag(bi_r_inv)
        
        # 公式（10），计算\hat{D}^{-1}A_k*H
        bi_adj = torch.matmul(bi_r_mat_inv, bi_adj)

        # 公式（10）Z^(l-1)W_e^{l}
        support = torch.mm(input, self.weight)
        
        # 公式（10）Z^(l-1)W_n^{l}
        support_bi = torch.mm(input, self.weight_bi)
        
        # 目标节点表示（下为特征矩阵）
        identity = torch.eye(adj.shape[0])
        output = torch.spmm(identity, support)
        
        # 邻居节点表示（下为特征矩阵）
        output_bi = torch.spmm(bi_adj, support_bi)
        
        # 公式（10）求得Z^(l)
        output = output + torch.mul(self.w, output_bi)

        if self.bias is not None:
            return output + self.bias, sim_matrix
        else:
            return output, sim_matrix

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
