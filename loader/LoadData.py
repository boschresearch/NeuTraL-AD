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

from .utils import *
from .LoadTabular import *
import torch


def CIFAR10_feat(path,normal_class):
    trainset = torch.load(path+'trainset_2048.pt')
    train_data,train_targets = trainset
    testset = torch.load(path+'testset_2048.pt')
    test_data,test_targets = testset
    test_labels = np.ones_like(test_targets)
    test_labels[test_targets==normal_class]=0
    
    train_clean = train_data[train_targets==normal_class]
    train_labels = np.zeros(train_clean.shape[0])

    return train_clean,train_labels,test_data,test_labels


def split_in_out(train_label, test_label, cls, cls_type):
    if type(cls) is not list:
        cls = [cls]
    labels = np.unique(train_label)

    if cls_type == 'normal':
        train_idx = np.zeros(train_label.shape[0])
        testin_idx = np.zeros(test_label.shape[0])

        for i in cls:
            train_idx = train_idx + (train_label == labels[i]).astype(np.int)
            testin_idx = testin_idx + (test_label == labels[i]).astype(np.int)

    elif cls_type=='abnormal':
        train_idx = np.ones(train_label.shape[0])
        testin_idx = np.ones(test_label.shape[0])

        for i in cls:
            train_idx = train_idx * (train_label != labels[i]).astype(np.int)
            testin_idx = testin_idx * (test_label != labels[i]).astype(np.int)

    train_idx[train_idx > 0] = 1
    testin_idx[testin_idx > 0] = 1

    y_train = np.ones(train_label.shape[0])
    y_train[train_idx==1] = 0
    y_test = np.ones(test_label.shape[0])
    y_test[testin_idx==1] = 0

    return y_train, y_test

def load_data(data_name,cls,cls_type):
    path = 'DATA/'
    if data_name == 'thyroid':
        train, train_label, test, test_label = Thyroid_train_test_split(path)
    elif data_name == 'arrhythmia':
        train, train_label, test, test_label = Arrhythmia_train_test_split(path)
    elif data_name == 'kdd':
        train, train_label, test, test_label = KDD_train_test_split(path)
    elif data_name == 'kddrev':
        train, train_label, test, test_label = KDDRev_train_test_split(path)
    elif data_name == 'cifar10_feat':
        train, train_label, test, test_label = CIFAR10_feat(path,cls)
    else:
        data_path = path + data_name + '/'
        train = np.load(data_path + 'train_array.npy')
        train_label = np.load(data_path + 'train_label.npy')
        test = np.load(data_path + 'test_array.npy')
        test_label = np.load(data_path + 'test_label.npy')
        train_label,test_label = split_in_out(train_label,test_label,cls,cls_type)
        train = train[train_label==0]
        train_label = train_label[train_label==0]
        train = np.transpose(train,(0,2,1))
        test = np.transpose(test,(0,2,1))

    trainset = CustomDataset(train,train_label)
    testset = CustomDataset(test,test_label)
    return trainset,testset,testset

