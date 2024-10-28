from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data.dataset import Subset
from torchmetrics import (
    MetricCollection, R2Score, MeanAbsoluteError, MeanSquaredError
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.data import SmilesXData
from src.model import SmilesX


class EarlyStopping:
    def __init__(self, patience=100, best_model_path="."):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_val_loss = np.Inf
        self.best_model_path = best_model_path

    def __call__(self, val_loss: float, model: SmilesX):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_loss, model)

        elif score > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(val_loss, model)
            self.counter = 0

    def checkpoint(self, val_loss: float, model: SmilesX):
        torch.save(model.state_dict(), str(self.best_model_path))
        self.best_val_loss = val_loss


class Trainer:
    def __init__(
        self,
        smilesX_data: SmilesXData,
        output_dir: Path,
        criterion,
        optimizer,
        scheduler=None,
        n_epochs: int = 100,
        early_stopping: EarlyStopping = None,
        device="cpu",
    ):
        self.smilesX_data = smilesX_data
        self.output_dir = output_dir
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.device = device
        self.history = {
            "train_RMSE": [], "train_MAE": [], "train_R2": [],
            "valid_RMSE": [], "valid_MAE": [], "valid_R2": []
        }

    def epoch_train(
        self,
        model: SmilesX,
        dataloader: DataLoader,
        metrics: MetricCollection
    ):
        epoch_loss = 0.0
        epoch_y_true = torch.Tensor()
        epoch_y_pred = torch.Tensor()

        model.train()
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = model(X)
            loss = torch.sqrt(self.criterion(outputs, y))
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            if self.scheduler is not None:
                self.scheduler.step()
            epoch_y_true = torch.cat(
                (epoch_y_true.cpu(), y.cpu())
            )
            epoch_y_pred = torch.cat(
                (epoch_y_pred.cpu(), outputs.cpu())
            )
        epoch_loss = epoch_loss / len(dataloader.dataset)
        all_metrics = metrics(epoch_y_true, epoch_y_pred)
        return epoch_loss, all_metrics

    def epoch_valid(
        self,
        model: SmilesX,
        dataloader: DataLoader,
        metrics: MetricCollection
    ):
        epoch_loss = 0.0
        epoch_y_true = torch.Tensor()
        epoch_y_pred = torch.Tensor()

        model.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X)
                loss = torch.sqrt(self.criterion(outputs, y))
                epoch_loss += loss.item()
                epoch_y_true = torch.cat(
                    (epoch_y_true.cpu(), y.cpu())
                )
                epoch_y_pred = torch.cat(
                    (epoch_y_pred.cpu(), outputs.cpu())
                )
        epoch_loss = epoch_loss / len(dataloader.dataset)
        all_metrics = metrics(epoch_y_true, epoch_y_pred)
        return epoch_loss, all_metrics

    def refresh_metrics(self):
        self.history = {
            "train_RMSE": [], "train_MAE": [], "train_R2": [],
            "valid_RMSE": [], "valid_MAE": [], "valid_R2": []
        }

    def save_metrics(self, filename: str):
        metrics_dirpath = self.output_dir.joinpath(filename)
        metrics_df = pd.DataFrame.from_dict(self.history)
        metrics_df.index = range(1, len(metrics_df) + 1)
        metrics_df.reset_index().rename(columns={"index": "epoch"})
        metrics_df.to_csv(metrics_dirpath)

    def initialize_early_stopping(self):
        if self.early_stopping is not None:
            self.early_stopping.counter = 0
            self.early_stopping.best_score = None
            self.early_stopping.early_stop = False
            self.early_stopping.best_val_loss = np.Inf


class CrossValidationTrainer(Trainer):
    def __init__(
        self,
        smilesX_data: SmilesXData,
        output_dir: Path,
        criterion,
        optimizer,
        scheduler=None,
        n_epochs: int = 100,
        early_stopping: EarlyStopping = None,
        device="cpu",
    ):
        Trainer.__init__(
            self,
            smilesX_data,
            output_dir,
            criterion,
            optimizer,
            scheduler,
            n_epochs,
            early_stopping,
            device,
        )
        self.kfold_n_splits = smilesX_data.kfold_n_splits
        self.kf = KFold(
            n_splits=self.kfold_n_splits,
            shuffle=True,
            random_state=smilesX_data.random_state
        )
        self.loss = 0.0

    def fit(self, model: SmilesX):
        model = model.to(self.device)
        metrics = MetricCollection(
            metrics={
                "RMSE": MeanSquaredError(squared=False, num_outputs=1),
                "MAE": MeanAbsoluteError(),
                "R2": R2Score(
                    model.output_layer.out_features,
                    multioutput="uniform_average"
                )
            }
        )

        for fold, (train_idx, valid_idx) in enumerate(self.kf.split(self.smilesX_data.tensor_datasets["train"])):
            self.initialize_early_stopping()
            kfold_loss = 0.0
            train_dataset = Subset(self.smilesX_data.tensor_datasets["train"], train_idx)
            valid_dataset = Subset(self.smilesX_data.tensor_datasets["train"], valid_idx)
            train_dataloader = DataLoader(
                train_dataset,
                self.smilesX_data.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
            )
            valid_dataloader = DataLoader(
                valid_dataset,
                self.smilesX_data.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
            )
            with tqdm(range(1, self.n_epochs + 1)) as pbar:
                for _epoch in pbar:
                    pbar.set_description_str(f"Fold: {fold}")
                    epoch_train_loss, train_metrics = self.epoch_train(model, train_dataloader, metrics)
                    self.history["train_RMSE"].append(train_metrics["RMSE"].item())
                    self.history["train_MAE"].append(train_metrics["MAE"].item())
                    self.history["train_R2"].append(train_metrics["R2"].item())

                    epoch_valid_loss, valid_metrics = self.epoch_valid(model, valid_dataloader, metrics)
                    self.history["valid_RMSE"].append(valid_metrics["RMSE"].item())
                    self.history["valid_MAE"].append(valid_metrics["MAE"].item())
                    self.history["valid_R2"].append(valid_metrics["R2"].item())
                    pbar.set_postfix_str(f"loss={epoch_train_loss:.4f} valid_loss={epoch_valid_loss:.4f} valid_r2={valid_metrics['R2'].item():.4f}")

                    if self.early_stopping is not None:
                        self.early_stopping(epoch_valid_loss, model)
                        if self.early_stopping.early_stop:
                            kfold_loss = self.early_stopping.best_score
                            break
                    else:
                        kfold_loss = epoch_valid_loss
                self.loss += kfold_loss
            filename = f"fold{fold}_metrics.csv"
            self.save_metrics(filename)
            self.refresh_metrics()
            self.loss /= self.kfold_n_splits
        return model


class HoldOutTrainer(Trainer):
    def __init__(
        self,
        smilesX_data: SmilesXData,
        output_dir: Path,
        criterion,
        optimizer,
        scheduler=None,
        n_epochs: int = 100,
        early_stopping: EarlyStopping = None,
        device="cpu",
    ):
        Trainer.__init__(
            self,
            smilesX_data,
            output_dir,
            criterion,
            optimizer,
            scheduler,
            n_epochs,
            early_stopping,
            device,
        )

        self.train_dataloader = DataLoader(
            self.smilesX_data.tensor_datasets["train"],
            self.smilesX_data.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
        self.valid_dataloader = DataLoader(
            self.smilesX_data.tensor_datasets["valid"],
            self.smilesX_data.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
        )
        self.loss = 0.0

    def fit(self, model: SmilesX):
        model = model.to(self.device)
        metrics = MetricCollection(
            metrics={
                "RMSE": MeanSquaredError(squared=False, num_outputs=1),
                "MAE": MeanAbsoluteError(),
                "R2": R2Score(
                    model.output_layer.out_features,
                    multioutput="uniform_average"
                )
            }
        )
        with tqdm(range(1, self.n_epochs + 1)) as pbar:
            for _epoch in pbar:
                train_loss, train_metrics = self.epoch_train(
                    model, self.train_dataloader, metrics
                )
                self.history["train_RMSE"].append(train_metrics["RMSE"].item())
                self.history["train_MAE"].append(train_metrics["MAE"].item())
                self.history["train_R2"].append(train_metrics["R2"].item())
                pbar.set_postfix_str(f"loss={train_loss:.4f}")

                valid_loss, valid_metrics = self.epoch_valid(
                    model, self.valid_dataloader, metrics
                )
                self.history["valid_RMSE"].append(valid_metrics["RMSE"].item())
                self.history["valid_MAE"].append(valid_metrics["MAE"].item())
                self.history["valid_R2"].append(valid_metrics["R2"].item())
                pbar.set_postfix_str(f"loss={train_loss:.4f} valid_loss={valid_loss:.4f} valid_r2={valid_metrics['R2'].item():.4f}")
                if self.early_stopping is not None:
                    self.early_stopping(valid_loss, model)
                    if self.early_stopping.early_stop:
                        valid_loss = self.early_stopping.best_score
                        break
        filename = "hold-out_metrics.csv"
        self.save_metrics(filename)
        self.loss = valid_loss
        return model
