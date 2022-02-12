# Neural Transformation Learning for Anomaly Detection (NeuTraLAD) - a self-supervised method for anomaly detection
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from torch_scatter import scatter
import torch.nn.init as init

class GraphNorm(nn.Module):
    def __init__(self, dim, affine=True):
        super(GraphNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim),requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(dim),requires_grad=False)
        self.scale = nn.Parameter(torch.ones(dim),requires_grad=affine)
    def forward(self,node_emb,graph):
        try:
            num_nodes_list = torch.tensor(graph.__num_nodes_list__).long().to(node_emb.device)
        except:
            num_nodes_list = graph.ptr[1:]-graph.ptr[:-1]
        num_nodes_list = num_nodes_list.long().to(node_emb.device)
        node_mean = scatter(node_emb, graph.batch, dim=0, dim_size=graph.__num_graphs__, reduce='mean')
        node_mean = node_mean.repeat_interleave(num_nodes_list, 0)

        sub = node_emb - node_mean*self.scale
        node_std = scatter(sub.pow(2), graph.batch, dim=0, dim_size=graph.__num_graphs__, reduce='mean')
        node_std = torch.sqrt(node_std + 1e-8)
        node_std = node_std.repeat_interleave(num_nodes_list, 0)
        norm_node = self.weight * sub / node_std + self.bias
        return norm_node
    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)
        init.ones_(self.scale)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, bias=True):
        super(MLP, self).__init__()
        self.lin1 = Linear(in_dim, hidden, bias=bias)
        self.lin2 = Linear(hidden, out_dim, bias=bias)
        # self.lin3 = Linear(hidden, out_dim, bias=bias)

    def forward(self, z):
        z = self.lin2(F.relu(self.lin1(z)))
        # return self.lin3(F.relu(z))
        return z
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

class GIN(torch.nn.Module):

    def __init__(self, dim_features, dim_targets, config):
        super(GIN, self).__init__()

        hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.nns = []
        self.convs = []
        self.norms = []
        self.projs = []
        self.use_norm = config['norm_layer']
        bias = config['bias']

        if config['aggregation'] == 'add':
            self.pooling = global_add_pool
        elif config['aggregation'] == 'mean':
            self.pooling = global_mean_pool

        for layer in range(self.num_layers):

            if layer == 0:
                input_emb_dim = dim_features
            else:
                input_emb_dim = hidden_dim
            self.nns.append(Sequential(Linear(input_emb_dim, hidden_dim, bias=bias), ReLU(),
                                       Linear(hidden_dim, hidden_dim, bias=bias)))
            self.convs.append(GINConv(self.nns[-1], train_eps=bias))  # Eq. 4.2
            if self.use_norm == 'bn':
                self.norms.append(BatchNorm1d(hidden_dim, affine=True))
            elif self.use_norm == 'gn':
                self.norms.append(GraphNorm(hidden_dim, True))

            self.projs.append(MLP(hidden_dim, hidden_dim, dim_targets, bias))
        self.nns = nn.ModuleList(self.nns)
        self.convs = nn.ModuleList(self.convs)
        self.norms = nn.ModuleList(self.norms)
        self.projs = nn.ModuleList(self.projs)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        z_cat = []
        # x = self.input_layer(x)
        for layer in range(self.num_layers):

            x = self.convs[layer](x, edge_index)
            if self.use_norm == 'bn':
                x = self.norms[layer](x)
            elif self.use_norm == 'gn':
                x = self.norms[layer](x, graph)
            x = F.relu(x)
            z = self.projs[layer](x)
            z = self.pooling(z, batch)
            z_cat.append(z)
        z_cat = torch.cat(z_cat, -1)
        return z_cat

    def reset_parameters(self):
        for norm in self.norms:
            norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for proj in self.projs:
            proj.reset_parameters()