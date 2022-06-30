from matplotlib.pyplot import delaxes
import torch
import random
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import time


def  set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.linear = nn.Linear(n_hid, n_out)

    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)

    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)

#refer to https://github.com/iMoonLab/THU-HyperG/blob/master/hyperg/hyperg.py
class HyperG:
    def __init__(self, H, X=None, w=None):
        """ Initial the incident matrix, node feature matrix and hyperedge weight vector of hypergraph
        :param H: scipy coo_matrix of shape (n_nodes, n_edges)
        :param X: numpy array of shape (n_nodes, n_features)
        :param w: numpy array of shape (n_edges,)
        """
        assert sparse.issparse(H)
        assert H.ndim == 2

        self._H = H
        self._n_nodes = self._H.shape[0]
        self._n_edges = self._H.shape[1]

        if X is not None:
            assert isinstance(X, np.ndarray) and X.ndim == 2
            self._X = X
        else:
            self._X = None

        if w is not None:
            self.w = w.reshape(-1)
            assert self.w.shape[0] == self._n_edges
        else:
            self.w = np.ones(self._n_edges)

        self._DE = None
        self._DV = None
        self._INVDE = None
        self._DV2 = None
        self._THETA = None
        self._L = None

    def num_edges(self):
        return self._n_edges

    def num_nodes(self):
        return self._n_nodes

    def incident_matrix(self):
        return self._H

    def hyperedge_weights(self):
        return self.w

    def node_features(self):
        return self._X

    def node_degrees(self):
        if self._DV is None:
            H = self._H.tocsr()
            dv = H.dot(self.w.reshape(-1, 1)).reshape(-1)
            self._DV = sparse.diags(dv, shape=(self._n_nodes, self._n_nodes))
        return self._DV

    def edge_degrees(self):
        if self._DE is None:
            H = self._H.tocsr()
            de = H.sum(axis=0).A.reshape(-1)
            self._DE = sparse.diags(de, shape=(self._n_edges, self._n_edges))
        return self._DE

    def inv_edge_degrees(self):
        if self._INVDE is None:
            self.edge_degrees()
            inv_de = np.power(self._DE.data.reshape(-1), -1.)
            self._INVDE = sparse.diags(inv_de, shape=(self._n_edges, self._n_edges))
        return self._INVDE

    def inv_square_node_degrees(self):
        if self._DV2 is None:
            self.node_degrees()
            dv2 = np.power(self._DV.data.reshape(-1)+1e-6, -0.5)
            self._DV2 = sparse.diags(dv2, shape=(self._n_nodes, self._n_nodes))
        return self._DV2

    def theta_matrix(self):
        if self._THETA is None:
            self.inv_square_node_degrees()
            self.inv_edge_degrees()

            W = sparse.diags(self.w)
            self._THETA = self._DV2.dot(self._H).dot(W).dot(self._INVDE).dot(self._H.T).dot(self._DV2)

        return self._THETA

    def laplacian(self):
        if self._L is None:
            self.theta_matrix()
            self._L = sparse.eye(self._n_nodes) - self._THETA
        return self._L

    def update_hyedge_weights(self, w):
        assert isinstance(w, (np.ndarray, list)), \
            "The hyperedge array should be a numpy.ndarray or list"

        self.w = np.array(w).reshape(-1)
        assert w.shape[0] == self._n_edges

        self._DV = None
        self._DV2 = None
        self._THETA = None
        self._L = None

    def update_incident_matrix(self, H):
        assert sparse.issparse(H)
        assert H.ndim == 2
        assert H.shape[0] == self._n_nodes
        assert H.shape[1] == self._n_edges

        # TODO: reset hyperedge weights?

        self._H = H
        self._DE = None
        self._DV = None
        self._INVDE = None
        self._DV2 = None
        self._THETA = None
        self._L = None
def print_log(message):
    """
    :param message: str,
    :return:
    """
    print("[{}] {}".format(time.strftime("%Y-%m-%d %X", time.localtime()), message))

import numpy as np
import scipy.sparse as sparse


def gen_attribute_hg(n_nodes, attr_dict, X=None):
    """
    :param attr_dict: dict, eg. {'attri_1': [node_idx_1, node_idx_1, ...], 'attri_2':[...]} (zero-based indexing)
    :param n_nodes: int,
    :param X: numpy array, shape = (n_samples, n_features) (optional)
    :return: instance of HyperG
    """

    if X is not None:
        assert n_nodes == X.shape[0]

    n_edges = len(attr_dict)
    node_idx = []
    edge_idx = []

    for idx, attr in enumerate(attr_dict):
        nodes = sorted(attr_dict[attr])
        node_idx.extend(nodes)
        edge_idx.extend([idx] * len(nodes))

    node_idx = np.asarray(node_idx)
    edge_idx = np.asarray(edge_idx)
    values = np.ones(node_idx.shape[0])

    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))
    return HyperG(H, X=X)

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy的sparse matrix转换成torch的sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

    
#refer to https://github.com/alge24/DyGNN/blob/b161555a5df69bd3fa9cc3ae5d4f5cd65ebe3a0f/decayer.py
class Decayer(nn.Module):
    def __init__(self, w1=0.01,w2=0.1, decay_method='rev'):
    # def __init__(self, w1=100,w2=200, decay_method='rev'):
        super(Decayer,self).__init__()
        self.decay_method = decay_method
        self.w1 = w1
        self.w2=w2

    def exponetial_decay(self, w, delta_t):
        return torch.exp(-w*delta_t)
    def log_decay(self, w, delta_t):
        return 1/torch.log(2.7183 + w*delta_t)
    def rev_decay(self, w, delta_t):
        return 1/(1 + w*delta_t)

    def forward(self,delta_t):
        seq=torch.zeros_like(delta_t)

        idx1=(delta_t<=24)
        idx2=(delta_t>24)

        # # print(delta_t)
        if self.decay_method == 'exp':
            seq[idx1]=self.exponetial_decay(self.w1,delta_t[idx1])
            seq[idx2]=self.exponetial_decay(self.w2,delta_t[idx2])
        elif self.decay_method == 'log':
            seq[idx1]=self.log_decay(self.w1,delta_t[idx1])
            seq[idx2]=self.log_decay(self.w2,delta_t[idx2])
        elif self.decay_method == 'rev':
            seq[idx1]=self.rev_decay(self.w1,delta_t[idx1])
            seq[idx2]=self.rev_decay(self.w2,delta_t[idx2])

        else:
            seq[idx1]=self.exponetial_decay(delta_t[idx1])
            seq[idx2]=self.exponetial_decay(delta_t[idx2])
        # print(seq,"----")

        return seq
        

def initializae_company_info(risk_data,company_attr,company_num,cause_type_num,court_type,category,idx=None):
    if idx:
        idx_dict={index:ser for ser,index in enumerate(idx)}
    company_risk=np.zeros((company_num,cause_type_num+court_type+category+1))
    for index in risk_data:
        risk_info=risk_data[index]
        cause_info=[0 for i in range(cause_type_num)]
        court_info=[0 for i in range(court_type)]
        res_info=[0 for i in range(category)]
        time_info=[]
        for i in range(len(risk_info)):
            justify=risk_info[i]
            cause=justify[0]
            court=justify[1]
            res=justify[2]
            time=justify[3]
            cause_info[cause]+=1
            court_info[court]+=1
            res_info[res]+=1
            time_info+=[time]
        time_ave=[np.average(time_info)]
        if idx:
            company_risk[idx_dict[index]]=np.concatenate((cause_info,court_info,res_info,time_ave),axis=0)
        else:
            company_risk[index]=np.concatenate((cause_info,court_info,res_info,time_ave),axis=0)
    company_attr=np.array(company_attr)
    company_info=np.concatenate((company_attr,company_risk),axis=1)
    return company_info


