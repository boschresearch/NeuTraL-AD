import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, samples, labels):
        self.labels = labels
        self.samples = samples
        self.dim_features = samples.shape[1]
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        sample = self.samples[idx]
        data = {"sample": sample, "label": label}
        return data

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

def norm_data(data, mu=1):
    return 2 * (data / 255.) - mu