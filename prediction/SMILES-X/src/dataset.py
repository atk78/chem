import multiprocessing

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# CORES = multiprocessing.cpu_count() // 2
CORES = 2

class Data(Dataset):
    def __init__(self, X, prop):
        self.X = X
        self.prop = prop

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.prop[index]


def make_dataloader(X, prop, batch_size=1, shuffle=False):
    dataset = Data(X=X, prop=prop)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=CORES,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader


class DataModule(pl.LightningDataModule):
    def __init__(self, x_train, x_valid, y_train, y_valid, batch_size=1):
        super().__init__()
        self.x_train, self.x_valid = x_train, x_valid
        self.y_train, self.y_valid = y_train, y_valid
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = Data(X=self.x_train, prop=self.y_train)
        self.valid_dataset = Data(X=self.x_valid, prop=self.y_valid)
        # self.test_dataset = Data(X=self.token.enum_tokens_test, prop=self.token.enum_prop_test)

    def train_dataloader(self):
        if len(self.train_dataset) != 0:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=CORES,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True
            )
        else:
            raise Exception("length of train dataset is zero.")

    def val_dataloader(self):
        sample_size = len(self.valid_dataset)
        if len(self.valid_dataset) != 0:
            return DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size if sample_size > self.batch_size else sample_size,
                shuffle=False,
                num_workers=CORES,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True
            )
        else:
            raise Exception("length of valid dataset is zero.")
