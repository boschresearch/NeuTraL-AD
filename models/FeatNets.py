import torch.nn as nn

class FeatTransformNet(nn.Module):
    def __init__(self, x_dim,h_dim,bias,num_layers):
        super(FeatTransformNet, self).__init__()
        net = []
        input_dim = x_dim
        for _ in range(num_layers-1):
            net.append(nn.Linear(input_dim,h_dim,bias=bias))
            net.append(nn.BatchNorm1d(h_dim,affine=bias))
            net.append(nn.ReLU())
            input_dim= h_dim
        net.append(nn.Linear(input_dim,x_dim,bias=bias))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)

        return out


class FeatEncoder(nn.Module):
    def __init__(self, x_dim,z_dim,bias,num_layers,batch_norm):

        super(FeatEncoder, self).__init__()

        enc = []
        input_dim = x_dim
        for _ in range(num_layers - 1):
            enc.append(nn.Linear(input_dim, int(input_dim/2),bias=bias))
            if batch_norm:
                enc.append(nn.BatchNorm1d(int(input_dim/2),affine=bias))
            enc.append(nn.ReLU())
            input_dim = int(input_dim/2)

        self.fc = nn.Linear(input_dim, z_dim,bias=bias)
        self.enc = nn.Sequential(*enc)



    def forward(self, x):

        z = self.enc(x)
        z = self.fc(z)

        return z



class FeatNets():

    def _make_nets(self,x_dim,config):
        enc_nlayers = config['enc_nlayers']
        z_dim = config['enc_zdim']

        trans_nlayers = config['trans_nlayers']
        trans_hdim = config['trans_hdim']
        num_trans = config['num_trans']
        batch_norm = config['batch_norm']

        enc = FeatEncoder(x_dim,  z_dim, config['enc_bias'],enc_nlayers,batch_norm)
        trans = nn.ModuleList(
            [FeatTransformNet(x_dim, trans_hdim, config['trans_bias'], trans_nlayers) for _ in range(num_trans)])

        return enc,trans

