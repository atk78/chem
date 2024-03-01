from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class Data(Dataset):
    def __init__(self, X, prop):
        self.X = X
        self.prop = prop

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.prop[index]


def make_dataloader(X, prop, batch_size=1, train=False):
    if train:
        shuffle=True
    else:
        shuffle=False
    dataset = Data(X=X, prop=prop)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self, train_data, valid_data, test_data):
        self.train_dataset = Data(X=train_data[0], prop=train_data[1])
        self.valid_dataset = Data(X=valid_data[0], prop=valid_data[1])
        self.test_dataset = Data(X=test_data[0], prop=test_data[1])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        dataloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader

