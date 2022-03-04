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
# This source code is derived from A Fair Comparison of Graph Neural Networks for Graph Classification (ICLR 2020)
#   (https://github.com/diningphil/gnn-comparison)
# Copyright (C)  2020  University of Pisa
# licensed under GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

# The modifications include adjusting the functions under the class 'DatasetManager'.
# The date of modifications: April, 2021

from pathlib import Path
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch_geometric.transforms as T
import torch
from torch_geometric.utils import degree
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

class DatasetManager:
    def __init__(self,  num_folds=10, seed=0, holdout_test_size=0.1,
                  DATA_DIR = 'DATA'):

        self.root_dir = Path(DATA_DIR)
        self.holdout_test_size = holdout_test_size

        self.num_folds = num_folds
        assert (num_folds is not None and num_folds > 0) or num_folds is None

        self.seed = seed

        self.dataset = TUDataset(self.root_dir, name=self.data_name)
        self._process()
        splits_filename = self.root_dir/ f"{self.data_name}"/ 'processed'/f"{self.data_name}_splits.json"
        if not splits_filename.exists():
            self.splits = []
            self._make_splits()
        else:
            self.splits = json.load(open(splits_filename, "r"))

    def _make_splits(self):
        """
        DISCLAIMER: train_test_split returns a SUBSET of the input indexes,
            whereas StratifiedKFold.split returns the indexes of the k subsets, starting from 0 to ...!
        """

        targets = self.dataset.data.y
        all_idxs = np.arange(len(self.dataset))

        if self.num_folds is None:  # holdout assessment strategy
            assert self.holdout_test_size is not None

            if self.holdout_test_size == 0:
                train_o_split, test_split = all_idxs, []
            else:
                train_o_split, test_split = train_test_split(all_idxs,
                                               stratify=targets,
                                               test_size=self.holdout_test_size,random_state=self.seed)

            split = {"test": all_idxs[test_split], 'model_selection': []}

            train_o_targets = targets[train_o_split]


            if self.holdout_test_size == 0:
                train_i_split, val_i_split = train_o_split, []
            else:
                train_i_split, val_i_split = train_test_split(train_o_split,
                                                              stratify=train_o_targets,
                                                              test_size=self.holdout_test_size,random_state=self.seed)
            split['model_selection'].append(
                {"train": train_i_split, "validation": val_i_split})



            self.splits.append(split)

        else:  # cross validation assessment strategy

            outer_kfold = StratifiedKFold(
                n_splits=self.num_folds, shuffle=True,random_state=self.seed)

            for train_ok_split, test_ok_split in outer_kfold.split(X=all_idxs, y=targets):
                split = {"test": all_idxs[test_ok_split], 'model_selection': []}

                train_ok_targets = targets[train_ok_split]

                assert self.holdout_test_size is not None
                train_i_split, val_i_split = train_test_split(train_ok_split,
                                                              stratify=train_ok_targets,
                                                              test_size=self.holdout_test_size,random_state=self.seed)
                split['model_selection'].append(
                    {"train": train_i_split, "validation": val_i_split})


                self.splits.append(split)

        filename = self.root_dir/ f"{self.data_name}"/ 'processed'/f"{self.data_name}_splits.json"
        with open(filename, "w") as f:
            json.dump(self.splits[:], f, cls=NumpyEncoder)

    def _process(self):
        raise NotImplementedError

    def get_test_fold(self, outer_idx):

        idxs = self.splits[outer_idx]["test"]
        if not isinstance(idxs,list):
            idxs = idxs.tolist()
        test_data = self.dataset[idxs]

        return test_data

    def get_model_selection_fold(self, outer_idx, normal_cls):

        idxs = self.splits[outer_idx]["model_selection"][0]
        idx_train = idxs["train"]
        idx_validation = idxs["validation"]

        idx_in = np.where(self.dataset.data.y == normal_cls)[0]
        idx_train_in = np.intersect1d(idx_in, idx_train)

        try:
            train_data = self.dataset[idx_train_in]
        except:
            train_data = self.dataset[idx_train_in.tolist()]
        try:
            val_data = self.dataset[idx_validation]
        except:
            val_data = self.dataset[idx_validation.tolist()]

        return train_data, val_data

class TUDatasetManager(DatasetManager):

    def _process(self):
        if self.dataset.data.x is None:
            max_degree = 0
            degs = []
            for data in self.dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                self.dataset.transform = T.OneHotDegree(max_degree)
                self.dim_features = max_degree+1
            else:

                self.dataset.transform = T.Constant(value=1, cat=False)
                self.dim_features = 1


