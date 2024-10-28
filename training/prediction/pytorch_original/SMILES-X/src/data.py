import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch import Tensor
from torch.utils.data import Dataset

from src import augm, token


CORES = 2


class Data(Dataset):
    def __init__(self, X: Tensor, prop: Tensor):
        self.X = X
        self.prop = prop

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.prop[index]


class SmilesXData:
    def __init__(
        self,
        smiles: np.ndarray,
        y: np.ndarray,
        augmentation: bool,
        validation_method="holdout",
        kfold_n_splits=None,
        batch_size=1,
        dataset_ratio=[0.8, 0.1, 0.1],
        random_state=42,
        scaling=True,
    ):
        if sum(dataset_ratio) != 1.0:
            raise RuntimeError("Make sure the sum of the ratios is 1.")
        self.smiles = smiles
        self.y = y
        self.augmentation = augmentation
        self.validation_method = validation_method
        self.batch_size = batch_size
        self.dataset_ratio = dataset_ratio
        self.random_state = random_state
        if scaling:
            self.scaler = RobustScaler(
                with_centering=True,
                with_scaling=True,
                quantile_range=(5.0, 95.0),
                copy=True,
            )
        else:
            self.scaler = None
        self.original_datasets = dict()
        self.enum_cards = dict()
        self.tokenized_datasets = dict()
        self.tensor_datasets = dict()
        self.kfold_n_splits = kfold_n_splits
        self.data_splitting()
        self.tokenized_smiles()

    def data_splitting(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.smiles,
            self.y,
            test_size=1 - self.dataset_ratio[0],
            shuffle=True,
            random_state=self.random_state,
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_test,
            y_test,
            test_size=self.dataset_ratio[2] / (self.dataset_ratio[2] + self.dataset_ratio[1]),
            shuffle=True,
            random_state=self.random_state,
        )
        if self.validation_method == "holdout":
            if self.scaler is not None:
                y_train = self.scaler.fit_transform(y_train)
                y_valid = self.scaler.transform(y_valid)
                y_test = self.scaler.transform(y_test)
            X_train, enum_card_train, y_train = augm.augment_data(
                X_train, y_train, self.augmentation
            )
            X_valid, enum_card_valid, y_valid = augm.augment_data(
                X_valid, y_valid, self.augmentation
            )
            X_test, enum_card_test, y_test = augm.augment_data(
                X_test, y_test, self.augmentation
            )
            self.original_datasets["train"] = [X_train, y_train]
            self.original_datasets["valid"] = [X_valid, y_valid]
            self.original_datasets["test"] = [X_test, y_test]
            self.enum_cards["train"] = enum_card_train
            self.enum_cards["valid"] = enum_card_valid
            self.enum_cards["test"] = enum_card_test
        else:
            X_train = np.concatenate([X_train, X_valid], axis=0)
            y_train = np.concatenate([y_train, y_valid], axis=0)
            if self.scaler is not None:
                y_train = self.scaler.fit_transform(y_train)
                y_test = self.scaler.transform(y_test)
            X_train, enum_card_train, y_train = augm.augment_data(
                X_train, y_train, self.augmentation
            )
            X_test, enum_card_test, y_test = augm.augment_data(
                X_test, y_test, self.augmentation
            )
            self.original_datasets["train"] = [X_train, y_train]
            self.original_datasets["test"] = [X_test, y_test]
            self.enum_cards["train"] = enum_card_train
            self.enum_cards["test"] = enum_card_test

    def tokenized_smiles(self):
        for phase, [smiles, y] in self.original_datasets.items():
            self.tokenized_datasets[phase] = [token.get_tokens(smiles), y]

    def tensorize(self, max_length: int, tokens: list[str]):
        for phase, [tokenized_smiles, y] in self.tokenized_datasets.items():
            token_tensor, y_tensor = token.convert_to_int_tensor(
                tokenized_smiles, y, max_length, tokens
            )
            # ! enum_card
            self.tensor_datasets[phase] = Data(X=token_tensor, prop=y_tensor)
