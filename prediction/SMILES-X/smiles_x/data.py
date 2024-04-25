import logging

from numpy import ndarray
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from lightning import LightningDataModule


CORES = 2


def random_split(
    smiles_input: ndarray,
    y_input: ndarray,
    train_ratio=0.8,
    validation_ratio=0.1,
    test_ratio=0.1,
    random_state=42,
    scaling=True
):
    scaler = None
    y_input = y_input.reshape(-1, 1)
    if train_ratio + validation_ratio + test_ratio != 1:
        raise RuntimeError("Make sure the sum of the ratios is 1.")
    logging.info(f"Train/valid/test splits:{train_ratio:.2f}/{validation_ratio:.2f}/{test_ratio:.2f}")
    x_train, x_test, y_train, y_test = train_test_split(
        smiles_input,
        y_input,
        test_size=1 - train_ratio,
        shuffle=True,
        random_state=random_state,
    )
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_test,
        y_test,
        test_size=test_ratio / (test_ratio + validation_ratio),
        shuffle=True,
        random_state=random_state,
    )
    if scaling:
        scaler = RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(5.0, 95.0),
            copy=True
        )
        y_train = scaler.fit_transform(y_train)
        y_valid = scaler.transform(y_valid)
        y_test = scaler.transform(y_test)
    return x_train, x_valid, x_test, y_train, y_valid, y_test, scaler


class Data(Dataset):
    def __init__(self, X: torch.Tensor, prop: torch.Tensor):
        self.X = X
        self.prop = prop

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.prop[index]


class DataModule(LightningDataModule):
    def __init__(
        self,
        x_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_train: torch.Tensor,
        y_valid: torch.Tensor,
        batch_size: int = 1
    ):
        super().__init__()
        self.x_train, self.x_valid = x_train, x_valid
        self.y_train, self.y_valid = y_train, y_valid
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = Data(X=self.x_train, prop=self.y_train)
        self.valid_dataset = Data(X=self.x_valid, prop=self.y_valid)

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
        batch_size = (
            self.batch_size if sample_size > self.batch_size else sample_size
        )
        if len(self.valid_dataset) != 0:
            return DataLoader(
                self.valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=CORES,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True
            )
        else:
            raise Exception("length of valid dataset is zero.")
