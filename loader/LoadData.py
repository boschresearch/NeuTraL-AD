from scipy import io
from .utils import *
import torch
import pandas as pd
import scipy.io
from scipy.io import arff
import sys
import zipfile
import os
import torchvision.datasets as dset
import numpy as np

def CIFAR10_feat(normal_class,root='/fs/scratch/rng_cr_bcai_dl/qic2rng/datasets/cifar10_features/'):
    trainset = torch.load(root+'trainset_2048.pt')
    train_data,train_targets = trainset
    testset = torch.load(root+'testset_2048.pt')
    test_data,test_targets = testset
    test_labels = np.ones_like(test_targets)
    test_labels[test_targets==normal_class]=0
    
    train_clean = train_data[train_targets==normal_class]
    train_labels = np.zeros(train_data.shape[0])

    return train_clean,train_labels,test_data,test_labels


    
def CIFAR10_Dataset(normal_class,root='/fs/scratch/rng_cr_bcai_dl/qic2rng/datasets/'):
    if not os.path.exists(root):
        os.mkdir(root)

    trainset = dset.CIFAR10(root, train=True, download=True)
    train_data = np.array(trainset.data)
    train_targets = np.array(trainset.targets)

    testset = dset.CIFAR10(root, train=False, download=True)
    test_data = np.array(testset.data)
    test_targets = np.array(testset.targets)
    test_labels = np.ones_like(test_targets)
    test_labels[test_targets==normal_class]=0
    train_clean = train_data[np.where(train_targets == normal_class)]
    x_train = norm_data(np.asarray(train_clean, dtype='float32'))
    x_test = norm_data(np.asarray(test_data, dtype='float32'))

    x_train = x_train.transpose(0, 3, 1, 2)
    x_test = x_test.transpose(0, 3, 1, 2)
    train_labels = np.zeros(x_train.shape[0])
    return x_train,train_labels, x_test, test_labels


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

    if data_name == 'thyroid':
        train, train_label, test, test_label = Thyroid_train_valid_data()
    elif data_name == 'arrhythmia':
        train, train_label, test, test_label = Arrhythmia_train_valid_data()
    elif data_name == 'kdd':
        train, train_label, test, test_label = KDD99_train_valid_data()
    elif data_name == 'kddrev':
        train, train_label, test, test_label = KDD99Rev_train_valid_data()
    elif data_name == 'cifar10_feat':
        train, train_label, test, test_label = CIFAR10_feat(cls)
    else:
        data_path = 'DATA/' + data_name + '/'
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
