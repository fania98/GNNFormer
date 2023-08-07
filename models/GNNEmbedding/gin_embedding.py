"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type, use_classification=False, tags=None):
        """model parameters setting
        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.num_channels = hidden_dim
        self.output_dim = output_dim
        self.use_classification = use_classification
        self.node_pool = torch.nn.Conv1d(in_channels=num_layers-1, out_channels=1, kernel_size=(1,))
        # self.node_output = torch.nn.Linear(hidden_dim, hidden_dim)

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()


        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        # self.node_prediction = torch.nn.ModuleList()

        if self.use_classification:
            self.tags = tags
            self.linears_prediction = torch.nn.ModuleList()
            self.feature_heads = torch.nn.ModuleList()

            for layer in range(num_layers):
                if layer == 0:
                    self.linears_prediction.append(
                        nn.Linear(input_dim, hidden_dim))
                    # self.node_prediction.append(nn.Linear(input_dim, output_dim))
                else:
                    self.linears_prediction.append(
                        nn.Linear(hidden_dim, hidden_dim))
                    # self.node_prediction.append(nn.Linear(hidden_dim, output_dim))


            for tag in self.tags:
                # head = torch.nn.ModuleList()
                # head.append(nn.Linear(hidden_dim, output_dim))
                # head.append(nn.Linear(output_dim, tag).cuda())
                self.feature_heads.append(nn.Linear(hidden_dim, output_dim))

            self.classfy_head = nn.Linear(output_dim, 3)

            self.drop = nn.Dropout(final_dropout)
            self.classify_batch_norm = nn.BatchNorm1d(len(self.tags))

            if graph_pooling_type == 'sum':
                self.pool = SumPooling()
            elif graph_pooling_type == 'mean':
                self.pool = AvgPooling()
            elif graph_pooling_type == 'max':
                self.pool = MaxPooling()
            else:
                raise NotImplementedError

    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        # hidden_rep = [h]
        hidden_rep = torch.empty(0).cuda()
        # node_predict_merge = torch.zeros(g.num_nodes(),self.output_dim).to(g.device)
        for i in range(self.num_layers-1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            # node_predicts = self.node_prediction[i](h)
            # node_predict_merge += node_predicts
            # hidden_rep.append(h)
            hidden_rep = torch.concat([hidden_rep, torch.unsqueeze(h,-2)], dim=-2) # [node_num, numlayer-1, feat]
        # pool_h = sum(hidden_rep)/len(hidden_rep)
        # h_out = self.node_output(pool_h)

        h_out = self.node_pool(hidden_rep)
        h_out = torch.squeeze(h_out)

        # node_predict_merge = 0
        scores_over_layer = 0
        semantic_features = None

        if self.use_classification:
            # perform pooling over all nodes in each graph in every layer
            # scores_over_layer = torch.zeros((16, len(self.tags), 3)).cuda()
            semantic_features = torch.zeros((16, len(self.tags), self.output_dim)).cuda()
            hidden_rep = hidden_rep.permute([1, 0, 2])
            for i, h in enumerate(hidden_rep):
                pooled_h = self.pool(g, h)  # h层pool得到的特征
                # score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
                layer_feature = self.linears_prediction[i](pooled_h)

                for j, tag_head in enumerate(self.feature_heads):
                    semantic_feature = self.drop(tag_head(layer_feature))  # [batch_size, 256]
                    semantic_features[:, j, :] += semantic_feature
                    # scores_over_layer[:, j, :] += tag_head[1](semantic_feature)

            semantic_features = self.classify_batch_norm(semantic_features)

            scores_over_layer = self.classfy_head(semantic_features)


            # scores_over_layer = F.softmax(scores_over_layer, dim=-1)

            # scores_over_layer = torch.reshape(scores_over_layer, [-1, 3])
            # semantic_features = F.softmax(semantic_features, dim=-1)
            # semantic_features =



        # node_predict_merge = F.softmax(node_predict_merge, dim=1)
        # return hidden_rep[-1], score_over_layer, node_predict_merge
        return h_out, scores_over_layer, semantic_features