import logging

import numpy as np
from torch_geometric.loader import DataLoader
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

CORES = 2


def random_split(
    smiles_input: np.ndarray,
    y_input: np.ndarray,
    train_ratio=0.8,
    validation_ratio=0.1,
    test_ratio=0.1,
    random_state=42,
    scaling=True
):
    scaler = None
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


class DataModule(LightningDataModule):
    def __init__(self, graph_train, graph_valid, batch_size=1):
        super().__init__()
        self.graph_train, self.graph_valid = graph_train, graph_valid
        self.batch_size = batch_size

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        if len(self.graph_train) != 0:
            return DataLoader(
                self.graph_train,
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
        sample_size = len(self.graph_valid)
        if len(self.graph_valid) != 0:
            return DataLoader(
                self.graph_valid,
                batch_size=(
                    self.batch_size
                    if sample_size > self.batch_size else sample_size
                ),
                shuffle=False,
                num_workers=CORES,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True
            )
        else:
            raise Exception("length of valid dataset is zero.")
