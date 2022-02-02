import torch.nn as nn

# Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,bias):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,bias=bias)  # , padding)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None,bias=True):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,bias=bias)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


# Residual Block
#   adapted from pytorch tutorial
#   https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-
#   intermediate/deep_residual_network/main.py
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

    # Image Transform Network

class Transformation(nn.Module):
    def __init__(self, channel,num_layer,bias):
        super(Transformation, self).__init__()

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool2d(2, 2)
        # encoding layers
        self.conv1 = ConvLayer(channel, 8, kernel_size=3, stride=1,bias=bias)
        self.bn1_e = nn.BatchNorm2d(8, affine=bias)

        self.conv2 = ConvLayer(8, 16, kernel_size=3, stride=1,bias=bias)
        self.bn2_e = nn.BatchNorm2d(16, affine=bias)

        # residual layers
        res_blocks = []
        for _ in range(num_layer):
            res_blocks.append( ResidualBlock(16,bias))
        self.res = nn.Sequential(*res_blocks)
        #
        self.deconv2 = UpsampleConvLayer(16, 8, kernel_size=3, stride=1, upsample=2,bias=bias)
        self.bn1_d = nn.BatchNorm2d(8, affine=bias)
        #
        self.deconv1 = UpsampleConvLayer(8, channel, kernel_size=3, stride=1,bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encode
        y = self.relu(self.bn1_e(self.conv1(x)))
        y = self.maxpool(self.relu(self.bn2_e(self.conv2(y))))
        # residual layers
        y = self.res(y)
        # decode
        y = self.relu(self.bn1_d(self.deconv2(y)))
        y = self.deconv1(y)

        return y


# The code is adapted from https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, batchnorm=False, bias=False):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes, affine=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes, affine=False)

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=bias) or None
        self.batchnorm = batchnorm

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.batchnorm:
            out = self.bn2(out)

        if self.convShortcut is not None:
            identity = self.convShortcut(x)

        out += identity
        out = self.relu(out)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, batchnorm,bias):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, batchnorm,bias)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,batchnorm,bias):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, batchnorm,bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class ResNet(nn.Module):
    def __init__(self, channel,depth=3, hdim = 16,zdim=64,batchnorm=False,bias=False):
        super(ResNet, self).__init__()

        block = BasicBlock
        self.zdim = zdim
        self.nChannels = hdim*depth+hdim
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(channel, hdim, kernel_size=3, stride=1,
                               padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(hdim,affine=bias)

        self.fc0 = nn.Linear(self.nChannels*7*7,zdim,bias=bias)

        res_blocks = []
        for i in range(depth):
            res_blocks.append(NetworkBlock(1,hdim*(i+1),hdim*(i+2),block,2,batchnorm,bias))

        self.res = nn.Sequential(*res_blocks)

        self.relu = nn.ReLU(inplace=True)

        self.batch_norm = batchnorm

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        out = self.res(out)
        out = self.fc0(out.view(out.shape[0],-1))

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
            [Transformation(x_dim, trans_nlayers,config['trans_bias']) for _ in range(num_trans)])

        return enc,trans

