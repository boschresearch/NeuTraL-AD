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
from torchnlp.word_to_vector import GloVe


class MyEmbedding(nn.Embedding):
    """Embedding base class."""

    def __init__(self, vocab_size, embedding_size, update_embedding=False):
        super().__init__(vocab_size, embedding_size)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weight.requires_grad = update_embedding


    def forward(self, x):

        embedded = super().forward(x)

        return embedded


def pretrained_model(dataset, embedding_size=None, pretrained_model=None):
    """Builds the neural network."""

    vocab_size = dataset.encoder.vocab_size

    # Set embedding

    # Load pre-trained model if specified

    if pretrained_model in ['GloVe_6B']:
        assert embedding_size in (50, 100, 200, 300)
        word_vectors = GloVe(name='6B', dim=embedding_size, cache='DATA/reuters/word_vectors_cache')
        embedding = MyEmbedding(vocab_size, embedding_size)
        # Init embedding with pre-trained word vectors
        for i, token in enumerate(dataset.encoder.vocab):
            embedding.weight.data[i] = word_vectors[token]
    else:
        raise Exception('unknown pre-trained model')
    return embedding


class TextTransformNet(nn.Module):
    def __init__(self, x_dim, h_dim, num_layers):
        super(TextTransformNet, self).__init__()

        net = [nn.Linear(x_dim, h_dim, bias=False), nn.BatchNorm1d(h_dim,affine=False),nn.ReLU()]

        for _ in range(num_layers - 2):
            net.append(nn.Linear(h_dim, h_dim, bias=False))
            net.append(nn.BatchNorm1d(h_dim,affine=False))
            net.append(nn.ReLU())
        net.append(nn.Linear(h_dim, x_dim, bias=False))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        len_shape = len(x.shape)
        if len_shape==3:
            x_len,num_batch,x_dim = x.shape
            x = x.reshape(-1,x_dim)
        out = self.net(x)
        if len_shape==3:
            out = out.reshape(x_len,num_batch,x_dim)

        return out


class TextAttention(nn.Module):
    def __init__(self, x_dim, h_dim):

        super(TextAttention, self).__init__()

        self.W1 = nn.Linear(x_dim, h_dim, bias=False)
        self.W2 = nn.Linear(h_dim, 1, bias=False)


    def forward(self, hidden):

        hidden = hidden.transpose(0, 1)

        x = torch.tanh(self.W1(hidden))
        x = torch.sigmoid(self.W2(x))
        A = x.transpose(1, 2)

        return A



class TextEnc(nn.Module):
    def __init__(self, z_dim, bias):
        super(TextEnc, self).__init__()

        self.proj = nn.Sequential(nn.Linear(z_dim, z_dim, bias=bias), nn.ReLU(),
                                  nn.Linear(z_dim, 100, bias=bias), nn.ReLU(),
                                  nn.Linear(100,100, bias=bias))

    def forward(self, z):
        z_proj = self.proj(z)
        return z_proj


class TextNets():

    def _make_nets(self,dataset,config):
        enc_hdim = config['enc_hdim']
        z_dim = config['latent_dim']

        trans_nlayers = config['trans_nlayers']
        trans_hdim = config['trans_hdim']
        num_trans = config['num_trans']

        pt_model = pretrained_model(dataset, z_dim, config['pretrained_model'])
        attention = TextAttention(z_dim, enc_hdim)
        trans = nn.ModuleList(
            [TextTransformNet(z_dim, trans_hdim, trans_nlayers) for _ in range(num_trans)])
        enc = TextEnc(z_dim, config['enc_bias'])

        return pt_model,attention, trans, enc
