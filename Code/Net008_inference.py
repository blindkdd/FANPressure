#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import networkx as nx
import matplotlib.pyplot as plt
import torch
import dgl
import numpy as np
import torch.nn.functional as F
import torch as th
import torch.nn as nn
import envPressureSubsEdges
from scipy.special import softmax

from tensorboardX import SummaryWriter


macrostep = 10
DEVICE = "cuda:0"

with open("stoppedEdges.pkl", 'rb') as f:
    stoppedEdges = pickle.load(f)

with open("amatrix_edges.pkl", 'rb') as f:
    A = pickle.load(f)

indices = {c:i for i,c in enumerate(list(A.columns))}
invertedIndices = {i:c for i,c in enumerate(list(A.columns))}
g = dgl.DGLGraph(np.eye(A.values.shape[0]) + A.values)
#g = dgl.DGLGraph(A.values)
N = g.number_of_nodes()
embedding_n = 32

g.ndata['entered'] = torch.zeros((g.number_of_nodes(),1)).cuda().to(DEVICE)

class PredictParkingModule(nn.Module):
    def __init__(self, in_feats, embedding_n):
        super(PredictParkingModule, self).__init__()
        self.embed = nn.Embedding(in_feats, embedding_n)
        self.L2 = nn.Linear(embedding_n+1, 1)
        self.A3 = F.relu

    def forward(self, node):
        #stack cars with embedding
        embedding = self.embed(node.data['features'])
        l2input = th.cat((embedding,node.data['cars']),dim=1)

        parked = self.L2(l2input)
        cars = self.A3(parked) + node.data['cars']
        cars = self.A3(cars)

        return {'cars' : cars, 'embedding': embedding}

class PredictEdgeProbModule(nn.Module):
    def __init__(self, embedding_n):
        super(PredictEdgeProbModule, self).__init__()
        self.L2 = nn.Linear(embedding_n*2, embedding_n)
        self.L3 = nn.Linear(embedding_n+3, 8)
        self.A3 = F.relu
        self.L4 = nn.Linear(8, 1)
        self.A4 = F.relu

    def forward(self, edge):
        h = torch.cat([edge.src['embedding'], edge.dst['embedding']], dim=1)
        h = self.L2(h)
        h = torch.cat([h, edge.src['cars'], edge.dst['cars'], edge.src['entered']], dim=1)
        h = self.L3(h)
        h = self.A3(h)
        h = self.L4(h)


        #enabled if it's self-loop or if it's not blocked
        selfloop = (edge.src['features'] == edge.dst['features']).type(torch.FloatTensor).cuda().to(DEVICE).unsqueeze(1)

        enabled = (selfloop != edge.src['free']).type(torch.FloatTensor).cuda().to(DEVICE)

        logit = self.A4(h)

        logit = logit * enabled

        return {'logit' : logit}


def flow_message_func(edges):
    #cars not from a selfloop
    selfloop = (edges.src['features'] == edges.dst['features']).type(torch.FloatTensor).cuda().to(DEVICE).unsqueeze(1)
    outer = (edges.src['cars']*edges.data['prob']) *(1-selfloop)
    all_moving = edges.src['cars']*edges.data['prob']


    return {'c' : all_moving, 'e' : edges.src['embedding'], 'o': outer}


def flow_reduce_func(nodes):
    incoming = torch.sum(nodes.mailbox['c'], dim=1)
    outer = torch.sum(nodes.mailbox['o'], dim=1)
    cars =  incoming

    return {'cars' : cars, 'entered': outer}


def softmax_feat(edges): return {'prob': th.softmax(edges.data['logit'], dim=1)}


class GCN(nn.Module):
    def __init__(self, nodes_feats_size, embedding_n):
        super(GCN, self).__init__()
        self.predict_parking = PredictParkingModule(nodes_feats_size, embedding_n)
        self.predict_edge_prob = PredictEdgeProbModule(embedding_n)

    def forward(self, g, nfeatures):
        g.ndata['features'] = nfeatures
        g.apply_nodes(v=g.nodes(),func=self.predict_parking)
        g.apply_edges(func=self.predict_edge_prob)
        g.group_apply_edges(func=softmax_feat, group_by='src')
        g.pull(v=g.nodes(),message_func=flow_message_func, reduce_func=flow_reduce_func)


        return g.ndata['cars'],g.ndata['embedding'], g.ndata.pop('entered')


class Net(nn.Module):
    def __init__(self, nfeatures_size, embedding_n):
        super(Net, self).__init__()
        self.gcn = GCN(nfeatures_size, embedding_n)

    def forward(self, g, nfeatures, cars, freeEdges, entered):
        g.ndata['cars'] = cars
        g.ndata['free'] = freeEdges
        g.ndata['entered'] = entered
        x = self.gcn(g, nfeatures)
        return x
