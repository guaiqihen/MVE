import os
import numpy as np
import scipy.sparse as sp
import torch
import random
import torch.nn.functional as F
from collections import defaultdict


def load_data_set(dataset):
    filepath = 'datasets'
    label_file = os.path.join(filepath, '{}/group.txt'.format(dataset))
    edge_file = os.path.join(filepath, '{}/graph.txt'.format(dataset))
    feature_file = os.path.join(filepath, '{}/feature.txt'.format(dataset))
    csd_file = os.path.join(filepath, '{}/csds.txt'.format(dataset))
    lda_file = os.path.join(filepath, '{}/lda.txt'.format(dataset))
    node_file = os.path.join(filepath, '{}/node_embeddings.txt'.format(dataset))
    kg_file = os.path.join(filepath, '{}/node_kg_embeddings.txt'.format(dataset))

    idx, labellist = read_node_label(label_file)
    G = read_graph(nodeids=idx, edge_file=edge_file)
    features = np.genfromtxt(feature_file, dtype=np.float)[:, 1:]
    csd_matrix = get_csd_matrix(csd_file)

    lda_feature = np.genfromtxt(lda_file, dtype=np.float)
    node_feature = np.genfromtxt(node_file, dtype=np.float)
    kg_feature = np.genfromtxt(kg_file, dtype=np.float)

    return idx, labellist, G, torch.FloatTensor(features), csd_matrix, torch.FloatTensor(
        lda_feature), torch.FloatTensor(node_feature), torch.FloatTensor(kg_feature)


def get_csd_matrix(csd_file):
    csdmatrix = np.loadtxt(csd_file)  # [c, csd_dim]
    csdmatrix = torch.FloatTensor(csdmatrix)
    csdmatrix = F.normalize(csdmatrix, p=2, dim=1)
    return csdmatrix


def dot_sim(x, y):
    # Inner product similarity
    ip_sim = torch.mm(x, y)
    return ip_sim


def get_data_split(c_train, c_val, idx, labellist):
    '''Input:
        idx: list[n, 1]
        labellist: list[n, string]
    Return:
            train_list: [num_train_samples, 1]
            val_list: [num_val_samples, 1]
            test_list: [num_test_samples, 1]
            total_class: num_class
    '''
    label_list_dict = defaultdict(list)
    for x, labels in zip(idx, labellist):
        for y in labels:
            label_list_dict[int(y)].append(int(x))

    train_list = []
    val_list = []
    test_list = []
    for i in label_list_dict.keys():
        # print(i, len(label_list_dict[i]))
        if i < c_train:
            train_list = train_list + label_list_dict[i]
        elif c_train <= i < (c_train + c_val):
            val_list = val_list + label_list_dict[i]
        else:
            test_list = test_list + label_list_dict[i]
    # print(len(train_list), len(val_list), len(test_list))
    return train_list, test_list, val_list


def get_acc(pred, label, c_train, c_val, model):
    # assume the c_train, c_val, c_test are ranked according to their c_ids
    mypred = torch.ones(pred.shape) * float('-inf')
    if (model == 'train'):
        mypred[:, :c_train] = pred[:, :c_train]
    elif model == 'val':
        mypred[:, c_train: c_train + c_val] = pred[:, c_train: c_train + c_val]
    elif model == 'test':
        mypred[:, c_train + c_val:] = pred[:, c_train + c_val:]
    return get_acc_basic(mypred, label)


def get_acc_basic(predict, label):
    predict = torch.argmax(predict, axis=1)
    acc = (label.cpu() == predict)
    result = acc.cpu().sum().numpy()
    # print('Train true/false acc:', result/len(acc))
    return result / len(acc)


# -------------------------------------
def read_node_label(filename):
    # print(os.getcwd())
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


def sparse_mx_to_torch_sparse_index_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return indices


def sparse_index_tensor_to_sparse_mx(edges, length):
    edges = edges.cpu()
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(length, length),
                        dtype=np.float32)
    return adj


def symmetrize(adj):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj


def read_graph(nodeids, edge_file):
    ''' Read a symmetric adjacency matrix from a file
        Input: nodeids: [1,2,3,4,...]
        Return: the sparse adjacency matrix
    '''

    idx = np.array(nodeids, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(edge_file, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)
    print('Edges', sp.coo_matrix.count_nonzero(adj))
    adj = symmetrize(adj)
    adj = sparse_mx_to_torch_sparse_index_tensor(adj)
    # return torch.FloatTensor(adj.todense())
    return adj


def row_normalize(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def use_cuda():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    return device
