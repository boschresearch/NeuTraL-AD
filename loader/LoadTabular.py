# Neural Transformation Learning for Anomaly Detection (NeuTraLAD) - a self-supervised method for anomaly detection
# Copyright (c) 2022 Robert Bosch GmbH
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
#
# This source code is derived from GOAD
#   (https://github.com/lironber/GOAD)

import pandas as pd
from scipy import io
import numpy as np

def Thyroid_train_valid_data():
    data = io.loadmat("DATA/thyroid.mat")
    samples = data['X']  # 3772
    labels = ((data['y']).astype(np.int32)).reshape(-1)

    norm_samples = samples[labels == 0]  # 3679 norm
    anom_samples = samples[labels == 1]  # 93 anom

    n_train = len(norm_samples) // 2
    train = norm_samples[:n_train]  # 1839 train
    train_label = np.zeros(train.shape[0])
    val_real = norm_samples[n_train:]
    val_fake = anom_samples
    val = np.concatenate([val_real,val_fake],0)

    val_label = np.zeros(val.shape[0])
    val_label[val_real.shape[0]:]=1

    return train,train_label,val,val_label

def Arrhythmia_train_valid_data():
    data = io.loadmat("DATA/arrhythmia.mat")
    samples = data['X']  # 518
    labels = ((data['y']).astype(np.int32)).reshape(-1)

    norm_samples = samples[labels == 0]  # 452 norm
    anom_samples = samples[labels == 1]  # 66 anom

    n_train = len(norm_samples) // 2
    train = norm_samples[:n_train]  # 226 train
    train_label = np.zeros(train.shape[0])
    val_real = norm_samples[n_train:]
    val_fake = anom_samples
    val = np.concatenate([val_real,val_fake],0)

    val_label = np.zeros(val.shape[0])
    val_label[val_real.shape[0]:]=1

    return train,train_label,val,val_label



def KDD99_train_valid_data():
    samples, labels, cont_indices = KDD99_preprocessing()
    anom_samples = samples[labels == 1]  # norm: 97278

    norm_samples = samples[labels == 0]  # attack: 396743

    n_norm = norm_samples.shape[0]
    ranidx = np.random.permutation(n_norm)
    n_train = n_norm // 2

    x_train = norm_samples[ranidx[:n_train]]
    norm_test = norm_samples[ranidx[n_train:]]

    val_real = norm_test
    val_fake = anom_samples
    train,val_real,val_fake= norm_kdd_data(x_train, val_real, val_fake, cont_indices)
    train_label = np.zeros(train.shape[0])
    val = np.concatenate([val_real,val_fake],0)
    val_label = np.zeros(val.shape[0])
    val_label[val_real.shape[0]:]=1

    return train,train_label,val,val_label


def KDD99Rev_train_valid_data():
    samples, labels, cont_indices = KDD99_preprocessing()

    norm_samples = samples[labels == 1]  # norm: 97278

    # Randomly draw samples labeled as 'attack'
    # so that the ratio btw norm:attack will be 4:1
    # len(anom) = 24,319
    anom_samples = samples[labels == 0]  # attack: 396743

    rp = np.random.permutation(len(anom_samples))
    rp_cut = rp[:24319]
    anom_samples = anom_samples[rp_cut]  # attack:24319

    n_norm = norm_samples.shape[0]
    ranidx = np.random.permutation(n_norm)
    n_train = n_norm // 2

    x_train = norm_samples[ranidx[:n_train]]
    norm_test = norm_samples[ranidx[n_train:]]

    val_real = norm_test
    val_fake = anom_samples
    train,val_real,val_fake= norm_kdd_data(x_train, val_real, val_fake, cont_indices)
    train_label = np.zeros(train.shape[0])
    val = np.concatenate([val_real,val_fake],0)
    val_label = np.zeros(val.shape[0])
    val_label[val_real.shape[0]:]=1

    return train,train_label,val,val_label

def KDD99_preprocessing():
    urls = [
    "DATA/kddcup.data_10_percent.gz",
    "DATA/kddcup.names"
    ]
    df_colnames = pd.read_csv(urls[1], skiprows=1, sep=':', names=['f_names', 'f_types'])
    df_colnames.loc[df_colnames.shape[0]] = ['status', ' symbolic.']
    df = pd.read_csv(urls[0], header=None, names=df_colnames['f_names'].values)
    df_symbolic = df_colnames[df_colnames['f_types'].str.contains('symbolic.')]
    df_continuous = df_colnames[df_colnames['f_types'].str.contains('continuous.')]
    samples = pd.get_dummies(df.iloc[:, :-1], columns=df_symbolic['f_names'][:-1])

    smp_keys = samples.keys()
    cont_indices = []
    for cont in df_continuous['f_names']:
        cont_indices.append(smp_keys.get_loc(cont))

    labels = np.where(df['status'] == 'normal.', 1, 0)
    return np.array(samples), np.array(labels), cont_indices

def norm_kdd_data( train_real, val_real, val_fake, cont_indices):
    symb_indices = np.delete(np.arange(train_real.shape[1]), cont_indices)
    mus = train_real[:, cont_indices].mean(0)
    sds = train_real[:, cont_indices].std(0)
    sds[sds == 0] = 1

    def get_norm(xs, mu, sd):
        bin_cols = xs[:, symb_indices]
        cont_cols = xs[:, cont_indices]
        cont_cols = np.array([(x - mu) / sd for x in cont_cols])
        return np.concatenate([bin_cols, cont_cols], 1)

    train_real = get_norm(train_real, mus, sds)
    val_real = get_norm(val_real, mus, sds)
    val_fake = get_norm(val_fake, mus, sds)
    return train_real, val_real, val_fake
