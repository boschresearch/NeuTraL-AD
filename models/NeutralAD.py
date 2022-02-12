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

class TabNeutralAD(nn.Module):
    def __init__(self, model, x_dim,config):
        super(TabNeutralAD, self).__init__()

        self.enc,self.trans = model._make_nets(x_dim,config)
        self.num_trans = config['num_trans']
        self.trans_type = config['trans_type']
        self.device = config['device']
        try:
            self.z_dim = config['latent_dim']
        except:
            if 32<=x_dim <= 300:
                self.z_dim = 32
            elif x_dim<32:
                self.z_dim = 2 * x_dim
            else:
                self.z_dim = 64

    def forward(self,x):
        x = x.type(torch.FloatTensor).to(self.device)
        x_T = torch.empty(x.shape[0],self.num_trans,x.shape[-1]).to(x)
        for i in range(self.num_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                x_T[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_T[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_T[:, i] = mask + x
        x_cat = torch.cat([x.unsqueeze(1),x_T],1)
        zs = self.enc(x_cat.reshape(-1,x.shape[-1]))
        zs = zs.reshape(x.shape[0],self.num_trans+1,self.z_dim)

        return zs

class SeqNeutralAD(nn.Module):
    def __init__(self, model, x_dim,config):
        super(SeqNeutralAD, self).__init__()

        self.enc,self.trans = model._make_nets(x_dim,config)
        self.num_trans = config['num_trans']
        self.trans_type = config['trans_type']
        self.device = config['device']
        self.z_dim = config['latent_dim']

    def forward(self,x):
        x = x.type(torch.FloatTensor).to(self.device)

        x_T = torch.empty(x.shape[0],self.num_trans,x.shape[1],x.shape[2]).to(x)
        for i in range(self.num_trans):
            mask = self.trans[i](x)

            if self.trans_type == 'forward':
                x_T[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_T[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_T[:, i] = mask + x
        x_cat = torch.cat([x.unsqueeze(1),x_T],1)
        zs = self.enc[0](x_cat.reshape(-1,x.shape[1],x.shape[2]))
        zs = zs.reshape(x.shape[0],self.num_trans+1,self.z_dim)

        return zs

class FeatNeutralAD(nn.Module):
    def __init__(self, model, x_dim,config):
        super(FeatNeutralAD, self).__init__()

        self.enc,self.trans = model._make_nets(x_dim,config)
        self.num_trans = config['num_trans']
        self.trans_type = config['trans_type']
        self.device = config['device']
        self.z_dim = config['enc_zdim']

    def forward(self,x):
        x = x.type(torch.FloatTensor).to(self.device)

        x_T = torch.empty(x.shape[0],self.num_trans,x.shape[-1]).to(x)
        for i in range(self.num_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                x_T[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_T[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_T[:, i] = mask + x
        x_cat = torch.cat([x.unsqueeze(1),x_T],1)
        zs = self.enc(x_cat.reshape(-1,x.shape[-1]))
        zs = zs.reshape(x.shape[0],self.num_trans+1,self.z_dim)

        return zs

class VisNeutralAD(nn.Module):
    def __init__(self, model, x_dim,config):
        super(VisNeutralAD, self).__init__()
        self.enc,self.trans = model._make_nets(x_dim,config)
        self.num_trans = config['num_trans']
        self.trans_type = config['trans_type']
        self.device = config['device']
        self.z_dim =  config['enc_zdim']

    def forward(self, x):
        x = x.type(torch.FloatTensor).to(self.device)

        x_T = torch.empty(x.shape[0],self.num_trans,x.shape[1],x.shape[2],x.shape[3]).to(x)
        for i in range(self.num_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                mask = torch.tanh(mask)
                x_T[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_T[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_T[:, i] = mask + x
        x_cat = torch.cat([x.unsqueeze(1),x_T],1)
        zs = self.enc(x_cat.reshape(-1,x.shape[1],x.shape[2],x.shape[3]))
        zs = zs.reshape(x.shape[0],self.num_trans+1,self.z_dim)

        return zs

class TextNeutralAD(nn.Module):
    def __init__(self, model,dataset,config):
        super(TextNeutralAD, self).__init__()
        self.pt_model,self.atn,self.trans,self.enc = model._make_nets(dataset, config)

        self.z_dim = config['latent_dim']
        self.num_trans = config['num_trans']
        self.trans_type = config['trans_type']
        self.device = config['device']
        self.dataset = dataset

    def transform(self,x_emb,i):
        mask = self.trans[i](x_emb)
        if self.trans_type == 'forward':
            x_t = mask
        elif self.trans_type == 'mul':
            mask = torch.sigmoid(mask)
            x_t = mask * x_emb
        elif self.trans_type == 'residual':
            x_t = mask + x_emb
        return x_t
    def forward(self,x):
        x = x.to(self.device)
        x_emb = self.pt_model(x)
        x_emb = x_emb.type(torch.FloatTensor).to(self.device)

        zs = torch.empty(x_emb.shape[1], self.num_trans + 1, self.z_dim).to(x_emb)
        A = self.atn(x_emb)
        z = A @ x_emb.transpose(0, 1)
        zs[:, 0] = z[:,0]
        for i in range(self.num_trans):
            x_t = self.transform(x_emb,i)
            z = A @ x_t.transpose(0, 1)
            zs[:,i+1]= z[:,0]

        z_projs = self.enc(zs)
        return z_projs

from .GraphNets import GIN
import torch.nn.init as init

class GraphNeutralAD(nn.Module):
    def __init__(self, dim_features,config):
        super(GraphNeutralAD, self).__init__()

        num_trans = config['num_trans']
        dim_targets = config['hidden_dim']
        num_layers = config['num_layers']
        self.gins = []
        for _ in range(num_trans):

            self.gins.append(GIN(dim_features,dim_targets,config))
        self.gins = nn.ModuleList(self.gins)

        self.bias = nn.Parameter(torch.empty(1, 1,dim_targets*num_layers), requires_grad=True)
        self.device = config['device']
        self.reset_parameters()
    def forward(self,data):
        data = data.to(self.device)
        z_cat = []
        for i,model in enumerate(self.gins):
            z = model(data)
            if i ==0:
                z = z+self.bias[:,i]
            z_cat.append(z.unsqueeze(1))

        return torch.cat(z_cat,1)

    def reset_parameters(self):
        init.normal_(self.bias)
        for nn in self.gins:
            nn.reset_parameters()