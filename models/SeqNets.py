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
import numpy as np

class res_trans1d_block(torch.nn.Module):

    def __init__(self, channel,bias=False):
        super(res_trans1d_block, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(channel, channel, 3, 1, 1, bias=bias)
        self.in1 = nn.InstanceNorm1d(channel, affine=bias)
        self.conv2 = nn.Conv1d(channel, channel, 3, 1, 1, bias=bias)
        self.in2 = nn.InstanceNorm1d(channel, affine=bias)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        #        out = self.pool(out)
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1,bias = False):
        super(ConvLayer, self).__init__()
        #        padding = kernel_size // 2
        padding = dilation * (kernel_size // 2)
        self.reflection_pad = nn.ReflectionPad1d(padding)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=bias)  # , padding)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv1d(out)
        return out


class SeqTransformNet(nn.Module):
    def __init__(self, x_dim,hdim,bias,num_layers):
        super(SeqTransformNet, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.conv1 = ConvLayer(x_dim, hdim, 3, 1,bias=bias)
        #        self.conv1 = nn.Conv1d(args.x_dim,2*args.x_dim,3,1,0,dilation=2**i)
        self.in1 = nn.InstanceNorm1d(hdim, affine=False)
        res_blocks = []
        for _ in range(num_layers-2):
            res_blocks.append(res_trans1d_block(hdim,bias))
        self.res = nn.Sequential(*res_blocks)
        #        self.conv2 = nn.ConvTranspose1d(args.x_dim,2*args.x_dim,3,1,0,dilation=2**i)
        self.conv2 = ConvLayer(hdim, x_dim, 3, 1,bias=bias)

    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))
        for block in self.res:
            out = block(out)
        out = self.conv2(out)

        return out


class res_block(nn.Module):

    def __init__(self, in_dim, out_dim, conv_param=None, downsample=None, batchnorm=False,bias=False):
        super(res_block, self).__init__()

        self.conv1 = nn.Conv1d(in_dim, in_dim, 1, 1, 0,bias=bias)
        if conv_param is not None:
            self.conv2 = nn.Conv1d(in_dim, in_dim, conv_param[0], conv_param[1], conv_param[2],bias=bias)
        else:
            self.conv2 = nn.Conv1d(in_dim, in_dim, 3, 1, 1,bias=bias)

        self.conv3 = nn.Conv1d(in_dim, out_dim, 1, 1, 0,bias=bias)
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(in_dim)
            self.bn2 = nn.BatchNorm1d(in_dim)
            self.bn3 = nn.BatchNorm1d(out_dim)
            if downsample:
                self.bn4 = nn.BatchNorm1d(out_dim)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.batchnorm = batchnorm

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batchnorm:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            if self.batchnorm:
                residual = self.bn4(residual)

        out += residual
        out = self.relu(out)

        return out


class SeqEncoder(nn.Module):
    def __init__(self, x_dim,x_len,h_dim,z_dim,bias,num_layers,batch_norm):

        super(SeqEncoder, self).__init__()

        self.bias = bias
        self.batchnorm = batch_norm
        enc = [self._make_layer(x_dim,h_dim,(3,1,1))]
        in_dim = h_dim
        window_size = x_len
        for i in range(num_layers - 2):
            out_dim = h_dim*2**i
            enc.append(self._make_layer(in_dim,out_dim,(3,2,1)))
            in_dim =out_dim
            window_size = np.floor((window_size+2-3)/2)+1


        self.enc = nn.Sequential(*enc)
        self.final_layer = nn.Conv1d(in_dim,z_dim,int(window_size),1,0)


    def _make_layer(self, in_dim, out_dim, conv_param=None):
        downsample = None
        if conv_param is not None:
            downsample = nn.Conv1d(in_dim, out_dim, conv_param[0], conv_param[1], conv_param[2],bias=self.bias)
        elif in_dim != out_dim:
            downsample = nn.Conv1d(in_dim, out_dim, 1, 1, 0,bias=self.bias)

        layer = res_block(in_dim, out_dim, conv_param, downsample=downsample, batchnorm=self.batchnorm,bias = self.bias)

        return layer

    def forward(self, x):

        z = self.enc(x)
        z = self.final_layer(z)

        return z.squeeze(-1)


class SeqNets():

    def _make_nets(self,x_dim,config):
        enc_nlayers = config['enc_nlayers']
        enc_hdim = config['enc_hdim']
        z_dim = config['latent_dim']
        x_len = config['x_length']
        trans_nlayers = config['trans_nlayers']
        num_trans = config['num_trans']
        batch_norm = config['batch_norm']

        enc = nn.ModuleList(
            [SeqEncoder(x_dim,x_len, enc_hdim, z_dim, config['enc_bias'],enc_nlayers,batch_norm) for _ in range(num_trans+1)])
        trans = nn.ModuleList(
            [SeqTransformNet(x_dim, x_dim,config['trans_bias'], trans_nlayers) for _ in range(num_trans)])

        return enc,trans
