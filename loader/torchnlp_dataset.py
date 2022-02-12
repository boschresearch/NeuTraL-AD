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
# This source code is derived from Context Vector Data Description (CVDD): An unsupervised anomaly detection method for text
#   (https://github.com/lukasruff/CVDD-PyTorch)
# Copyright (c) 2019 lukasruff
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from torchnlp.samplers import BucketBatchSampler
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors

import torch
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class BaseADDataset(ABC):
    """Anomaly detection dataset base class."""

    def __init__(self, root: str):
        super().__init__()
        self.root = root  # root path to data

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.outlier_classes = None  # tuple with original class labels that define the outlier class

        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.test_set = None  # must be of type torch.utils.data.Dataset

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__


class TorchnlpDataset(BaseADDataset):
    """TorchnlpDataset class for datasets already implemented in torchnlp.datasets."""

    def __init__(self, root: str):
        super().__init__(root)
        self.encoder = None  # encoder of class Encoder() from torchnlp

    def loaders(self, batch_size: int, shuffle_train=False, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):

        # Use BucketSampler for sampling
        train_sampler = BucketBatchSampler(self.train_set, batch_size=batch_size, drop_last=False,
                                           sort_key=lambda r: len(r['text']))
        test_sampler = BucketBatchSampler(self.test_set, batch_size=batch_size, drop_last=False,
                                          sort_key=lambda r: len(r['text']))

        train_loader = DataLoader(dataset=self.train_set, batch_sampler=train_sampler, collate_fn=collate_fn,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_sampler=test_sampler, collate_fn=collate_fn,
                                 num_workers=num_workers)
        return train_loader, test_loader


def collate_fn(batch):
    """ list of tensors to a batch tensors """
    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    transpose = (lambda b: b.t_().squeeze(0).contiguous())

    # indices = [row['index'] for row in batch]
    text_batch, _ = stack_and_pad_tensors([row['text'] for row in batch])
    label_batch = torch.stack([row['label'] for row in batch])

    return  transpose(text_batch), label_batch.float()
