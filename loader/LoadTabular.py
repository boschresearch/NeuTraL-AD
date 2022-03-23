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


import pandas as pd
from scipy import io
import numpy as np

def train_test_split(inliers,outliers):
    num_split = len(inliers) // 2
    train_data = inliers[:num_split]
    train_label = np.zeros(num_split)
    test_data = np.concatenate([inliers[num_split:],outliers],0)

    test_label = np.zeros(test_data.shape[0])
    test_label[num_split:]=1
    return train_data, train_label, test_data, test_label

def Thyroid_train_test_split(path):
    data = io.loadmat(path+"thyroid.mat")
    samples = data['X']  # 3772
    labels = ((data['y']).astype(np.int)).reshape(-1)

    inliers = samples[labels == 0]  # 3679 norm
    outliers = samples[labels == 1]  # 93 anom

    train_data, train_label, test_data, test_label=train_test_split(inliers,outliers)
    return train_data, train_label, test_data, test_label

def Arrhythmia_train_test_split(path):
    data = io.loadmat(path+"arrhythmia.mat")
    samples = data['X']  # 518
    labels = ((data['y']).astype(np.int)).reshape(-1)

    inliers = samples[labels == 0]  # 452 norm
    outliers = samples[labels == 1]  # 66 anom

    train_data, train_label, test_data, test_label=train_test_split(inliers,outliers)
    return train_data, train_label, test_data, test_label



def KDD_train_test_split(path):
    samples, labels, continual_idx = KDD_preprocessing(path)
    inliers = samples[labels == 0]  # attack: 396743
    outliers = samples[labels == 1]  # norm: 97278
    idx_perm = np.random.permutation(inliers.shape[0])
    inliers = inliers[idx_perm]

    train_data, train_label, test_data, test_label = train_test_split(inliers, outliers)
    train_data, test_data= norm_kdd_data(train_data, test_data, continual_idx)

    return train_data, train_label, test_data, test_label


def KDDRev_train_test_split(path):
    samples, labels, continual_idx = KDD_preprocessing(path)

    inliers = samples[labels == 1]  # norm: 97278
    outliers = samples[labels == 0]  # attack: 396743

    random_cut = np.random.permutation(len(outliers))[:24319]
    outliers = outliers[random_cut]  # attack:24319

    idx_perm = np.random.permutation(inliers.shape[0])
    inliers = inliers[idx_perm]

    train_data, train_label, test_data, test_label = train_test_split(inliers, outliers)
    train_data, test_data= norm_kdd_data(train_data, test_data, continual_idx)

    return train_data, train_label, test_data, test_label


def KDD_preprocessing(path):
    file_names = [path+"kddcup.data_10_percent.gz",path+"kddcup.names"]

    column_name = pd.read_csv(file_names[1], skiprows=1, sep=':', names=['f_names', 'f_types'])
    column_name.loc[column_name.shape[0]] = ['status', ' symbolic.']
    data = pd.read_csv(file_names[0], header=None, names=column_name['f_names'].values)
    data_symbolic = column_name[column_name['f_types'].str.contains('symbolic.')]
    data_continuous = column_name[column_name['f_types'].str.contains('continuous.')]
    samples = pd.get_dummies(data.iloc[:, :-1], columns=data_symbolic['f_names'][:-1])

    sample_keys = samples.keys()
    continuous_idx = []
    for cont_idx in data_continuous['f_names']:
        continuous_idx.append(sample_keys.get_loc(cont_idx))

    labels = np.where(data['status'] == 'normal.', 1, 0)
    return np.array(samples), np.array(labels), continuous_idx


def norm_kdd_data(train_data, test_data, continuous_idx):
    symbolic_idx = np.delete(np.arange(train_data.shape[1]), continuous_idx)
    mu = np.mean(train_data[:, continuous_idx],0,keepdims=True)
    std = np.std(train_data[:, continuous_idx],0,keepdims=True)
    std[std == 0] = 1

    train_continual = (train_data[:, continuous_idx]-mu)/std
    train_normalized = np.concatenate([train_data[:, symbolic_idx], train_continual], 1)
    test_continual = (test_data[:, continuous_idx]-mu)/std
    test_normalized = np.concatenate([test_data[:, symbolic_idx], test_continual], 1)

    return train_normalized, test_normalized
