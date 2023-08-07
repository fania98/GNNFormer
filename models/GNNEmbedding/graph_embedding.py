import dgl.nn as gnn
import torch.nn as nn
import torch.nn.functional as F

class GCNEmbedding(nn.Module):
    def __init__(self,in_feats, h_feats):
        super(GCNEmbedding, self).__init__()
        self.conv1 = gnn.GraphConv(in_feats, h_feats)
        self.conv2 = gnn.GraphConv(h_feats, h_feats)
        self.avg = gnn.AvgPooling()

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = self.avg(h)
        return h
