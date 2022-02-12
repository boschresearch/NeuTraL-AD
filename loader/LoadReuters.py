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

from .torchnlp_dataset import TorchnlpDataset
from torchnlp.datasets.dataset import Dataset
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN
from torch.utils.data import Subset
from nltk.corpus import reuters
from nltk import word_tokenize

import torch
import nltk
from nltk.corpus import stopwords
import string
import re

class Reuters_Dataset(TorchnlpDataset):

    def __init__(self, normal_class, root, append_sos=False,
                 append_eos=False, clean_txt=True):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        classes = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'ship']


        self.normal_classes = [classes[normal_class]]
        del classes[normal_class]
        self.outlier_classes = classes

        try:
            text_corpus = torch.load(root+'reuters_'+str(normal_class)+'_text.pt')
            self.train_set = torch.load(root+'reuters_'+str(normal_class)+'_train.pt')
            self.test_set = torch.load(root+'reuters_'+str(normal_class)+'_test.pt')
            self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=append_eos)
        except:
            # Load the reuters dataset
            self.train_set, self.test_set = reuters_dataset(directory=root, train=True, test=True, clean_txt=clean_txt)


            train_idx_normal = []  # for subsetting train_set to normal class
            for i, row in enumerate(self.train_set):
                if any(label in self.normal_classes for label in row['label']) and (len(row['label']) == 1):
                    train_idx_normal.append(i)
                    row['label'] = torch.tensor(0)
                else:
                    row['label'] = torch.tensor(1)
                row['text'] = row['text'].lower()

            test_idx = []  # for subsetting test_set to selected normal and anomalous classes
            for i, row in enumerate(self.test_set):
                if any(label in self.normal_classes for label in row['label']) and (len(row['label']) == 1):
                    test_idx.append(i)
                    row['label'] = torch.tensor(0)
                elif any(label in self.outlier_classes for label in row['label']) and (len(row['label']) == 1):
                    test_idx.append(i)
                    row['label'] = torch.tensor(1)
                else:
                    row['label'] = torch.tensor(1)
                row['text'] = row['text'].lower()

            # Subset train_set to normal class
            self.train_set = Subset(self.train_set, train_idx_normal)
            # Subset test_set to selected normal and anomalous classes
            self.test_set = Subset(self.test_set, test_idx)

            # Make corpus and set encoder
            text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.test_set)]
            torch.save(text_corpus, root+'reuters_' + str(normal_class) + '_text.pt')

            self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=append_eos)

            # Encode
            for row in datasets_iterator(self.train_set, self.test_set):
                if append_sos:
                    sos_id = self.encoder.stoi[DEFAULT_SOS_TOKEN]
                    row['text'] = torch.cat((torch.tensor(sos_id).unsqueeze(0), self.encoder.encode(row['text'])))
                else:
                    row['text'] = self.encoder.encode(row['text'])
            torch.save(self.train_set,root+'reuters_'+str(normal_class)+'_train.pt')
            torch.save(self.test_set,root+'reuters_'+str(normal_class)+'_test.pt')


def reuters_dataset(directory, train=True, test=False, clean_txt=False):
    """
    Load the Reuters-21578 dataset.

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train and test) depending on if their respective boolean argument
        is ``True``.
    """

    nltk.download('reuters', download_dir=directory)
    if directory not in nltk.data.path:
        nltk.data.path.append(directory)

    doc_ids = reuters.fileids()

    ret = []
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]

    for split_set in splits:

        split_set_doc_ids = list(filter(lambda doc: doc.startswith(split_set), doc_ids))
        examples = []

        for id in split_set_doc_ids:
            if clean_txt:
                text = clean_text(reuters.raw(id))
            else:
                text = ' '.join(word_tokenize(reuters.raw(id)))
            labels = reuters.categories(id)

            examples.append({
                'text': text,
                'label': labels,
            })

        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)

def clean_text(text: str, rm_numbers=True, rm_punct=True, rm_stop_words=True, rm_short_words=True):
    """ Function to perform common NLP pre-processing tasks. """

    # make lowercase
    text = text.lower()

    # remove punctuation
    if rm_punct:
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # remove numbers
    if rm_numbers:
        text = re.sub(r'\d+', '', text)

    # remove whitespaces
    text = text.strip()

    # remove stopwords
    if rm_stop_words:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        text_list = [w for w in word_tokens if not w in stop_words]
        text = ' '.join(text_list)

    # remove short words
    if rm_short_words:
        text_list = [w for w in text.split() if len(w) >= 3]
        text = ' '.join(text_list)

    return text
