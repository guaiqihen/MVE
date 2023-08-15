import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from scipy.special import comb
from utils import dot_sim, use_cuda, sparse_index_tensor_to_sparse_mx
from pytorch_metric_learning.losses import NTXentLoss

device = use_cuda()



class CreateFeature(nn.Module):
    def __init__(self, bow_dim, node_bert_dim, node_kg_dim, hidden, n_dim=256):
        super(CreateFeature, self).__init__()
        self.bow_dim = bow_dim
        self.node_bert_dim = node_bert_dim
        self.node_kg_dim = node_kg_dim
        self.hidden = hidden
        self.loss_func = NTXentLoss()
        self.n_dim = n_dim

        self.bowLinear = nn.Linear(self.bow_dim, hidden * 2)
        self.nodeBertLinear = nn.Linear(self.node_bert_dim, hidden)
        self.nodeKgLinear = nn.Linear(self.node_kg_dim, hidden)

        self.encoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, self.n_dim)
        )

        self.projection = nn.Sequential(
            nn.Linear(self.n_dim, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden * 2)
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

    def forward(self, bow, node, kg, lda):
        batch_size = bow.size(0)

        x_bow = self.bowLinear(bow)
        x_node = self.nodeBertLinear(node)
        x_kg = self.nodeKgLinear(kg)

        x_bert = torch.concat((x_node, x_kg), dim=-1)

        x_bert_encoded = self.encoder(x_bert)
        x_bow_encoded = self.encoder(x_bow)

        x_bert_proj = self.projection(x_bert_encoded)
        x_bow_proj = self.projection(x_bow_encoded)

        # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/179
        embeddings = torch.concat((x_bert_proj, x_bow_proj), dim=0)
        y = torch.arange(batch_size)
        y = torch.concat((y, y))
        if self.training:
            loss = self.loss_func(embeddings, y)
        else:
            loss = None
        x_final = torch.concat((x_bert_encoded, lda), dim=-1)
        return x_final, loss


class GCN(nn.Module):
    def __init__(self, dim, hidden, dropout=0.1):
        super(GCN, self).__init__()
        self.dim = dim
        self.hidden = hidden
        self.dropout = dropout

        self.graphconv = _GraphConv(self.dim, self.dropout)
        self.linear = _GraphLinear(self.dim, self.hidden, self.dropout)

    def forward(self, nodes, edges, labelmatrix):
        feature_list = self.graphconv(nodes, edges)
        compositional_press = self.linear(feature_list, labelmatrix)
        return compositional_press

class _GraphConv(nn.Module):
    def __init__(self, dim, dropout):
        super(_GraphConv, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.gcn1 = GCNConv(dim, 256)
        self.gcn2 = GCNConv(256, 192)
        self.gcn3 = GCNConv(192, 128)

    def forward(self, nodes, edges):
        x = nodes
        x = self.gcn1(x, edges)
        x = self.gcn2(x, edges)
        x = self.gcn3(x, edges)
        return x


class _GraphLinear(nn.Module):
    def __init__(self, n_in, n_h, dropout):
        super(_GraphLinear, self).__init__()
        self.fc1 = nn.Linear(n_h, n_h, bias=True)
        self.fc_final_pred_csd = nn.Linear(n_h, n_h, bias=True)

        self.dropout = dropout
        self.act = nn.ReLU()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

    def forward(self, feature_list, csdmatrix):
        templist = []
        temp_embedds = self.fc1(feature_list)
        temp_embedds = self.act(temp_embedds)
        temp_embedds = F.dropout(temp_embedds, p=self.dropout, training=self.training)

        total_embedds = temp_embedds
        total_embedds_pred_csd = self.fc_final_pred_csd(total_embedds)
        total_embedds_pred_csd = F.dropout(total_embedds_pred_csd, p=self.dropout, training=self.training)
        preds = dot_sim(total_embedds_pred_csd, csdmatrix.t())
        return preds
