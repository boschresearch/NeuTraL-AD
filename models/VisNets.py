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

import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,bias):
        super(ConvLayer, self).__init__()
        pad_size = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,bias=bias)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None,bias=True):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='nearest')
        pad_size = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,bias=bias)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        out = self.pad(x)
        out = self.conv(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels,bias):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1,bias=bias)
        self.bn1 = nn.BatchNorm2d(channels, affine=bias)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1,bias=bias)
        self.bn2 = nn.BatchNorm2d(channels, affine=bias)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out

class ImageTransformNet(nn.Module):
    def __init__(self, channel,num_layer):
        super(ImageTransformNet, self).__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        # encoding layers
        self.conv1 = ConvLayer(channel, 8, kernel_size=3, stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(8, affine=False)

        self.conv2 = ConvLayer(8, 16, kernel_size=3, stride=1,bias=False)
        self.bn2 = nn.BatchNorm2d(16, affine=False)

        # residual layers
        res_blocks = []
        for _ in range(num_layer):
            res_blocks.append(ResidualBlock(16,False))
        self.res = nn.Sequential(*res_blocks)
        #
        self.deconv2 = UpsampleConvLayer(16, 8, kernel_size=3, stride=1, upsample=2,bias=False)
        self.bn3 = nn.BatchNorm2d(8, affine=False)
        #
        self.deconv1 = UpsampleConvLayer(8, channel, kernel_size=3, stride=1,bias=False)


    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.maxpool(self.relu(self.bn2(self.conv2(y))))
        y = self.res(y)
        y = self.relu(self.bn3(self.deconv2(y)))
        y = self.deconv1(y)

        return y


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, batchnorm=False, bias=False):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=False)

        ifequal = (in_channels == out_channels)
        self.shortcut = (not ifequal) and nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                                                padding=0, bias=bias) or None
        self.batchnorm = batchnorm

    def forward(self, x):

        if self.shortcut is not None:
            identity = self.shortcut(x)
        else:
            identity = x
        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.batchnorm:
            out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_channels, out_channels, block, stride, batchnorm,bias):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_channels, out_channels, nb_layers, stride, batchnorm,bias)

    def _make_layer(self, block,in_channels, out_channels, nb_layers, stride,batchnorm,bias):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_channels or out_channels, out_channels, i == 0 and stride or 1, batchnorm,bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class ResNet(nn.Module):
    def __init__(self, channel,depth=3, hdim = 16,zdim=64,batchnorm=False,bias=False):
        super(ResNet, self).__init__()

        self.zdim = zdim
        self.nChannels = hdim*depth+hdim
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(channel, hdim, kernel_size=3, stride=1,
                               padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(hdim,affine=bias)
        res_blocks = []
        for i in range(depth):
            res_blocks.append(NetworkBlock(1,hdim*(i+1),hdim*(i+2),BasicBlock,2,batchnorm,bias))

        self.res = nn.Sequential(*res_blocks)
        self.fc = nn.Linear(self.nChannels*7*7,zdim,bias=bias)

        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = batchnorm

    def forward(self, x):
        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn(out)
        out = self.relu(out)
        out = self.res(out)
        out = self.fc(out.view(out.shape[0],-1))

        return out


class VisNets():

    def _make_nets(self,x_dim,config):
        enc_nlayers = config['enc_nlayers']
        enc_hdim = config['enc_hdim']
        enc_zdim = config['enc_zdim']
        trans_nlayers = config['trans_nlayers']
        num_trans = config['num_trans']
        batch_norm = config['batch_norm']

        enc = ResNet(x_dim,enc_nlayers, enc_hdim,enc_zdim,batch_norm, config['enc_bias'])
        trans = nn.ModuleList(
            [ImageTransformNet(x_dim, trans_nlayers) for _ in range(num_trans)])

        return enc,trans

