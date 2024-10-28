import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.mol2graph import Mol2Graph


CORES = 2


class GraphData:
    def __init__(
        self,
        smiles: np.ndarray,
        y: np.ndarray,
        mol2graph: Mol2Graph,
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
        self.mol2graph = mol2graph
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
        self.graph_datasets = dict()
        self.kfold_n_splits = kfold_n_splits

    # def get_graphs_datasets(self):
        n_node_features, n_edge_features = self.mol2graph.get_graph_features()
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
            splitted_datasets = {
                "train": [X_train, y_train],
                "valid": [X_valid, y_valid],
                "test": [X_test, y_test],
            }
        else:
            X_train = np.concatenate([X_train, X_valid], axis=0)
            y_train = np.concatenate([y_train, y_valid], axis=0)
            if self.scaler is not None:
                y_train = self.scaler.fit_transform(y_train)
                y_test = self.scaler.transform(y_test)
            splitted_datasets = {
                "train": [X_train, y_train],
                "test": [X_test, y_test],
            }
        for phase, [X, y] in splitted_datasets.items():
            self.graph_datasets[phase] = self.mol2graph.get_graph_vectors(
                X, y, n_node_features, n_edge_features
            )
