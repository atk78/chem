import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics import (
    MetricCollection, R2Score, MeanAbsoluteError, MeanSquaredError
)

from src.models.smiles_x import SmilesX


class LightningModel(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        lstm_units=16,
        dense_units=16,
        embedding_dim=32,
        learning_rate=1e-3,
        return_proba=False,
        loss_func="MAE",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=True)
        self.lr = learning_rate
        if loss_func == "MAE":
            self.loss_func = F.l1_loss
            self.train_metrics = MetricCollection(
                {
                    "train_r2": R2Score(),
                    "train_loss": MeanAbsoluteError()
                }
            )
            self.valid_metrics = MetricCollection(
                {
                    "valid_r2": R2Score(),
                    "valid_loss": MeanAbsoluteError()
                }
            )
        else:
            self.loss_func = F.mse_loss
            self.train_metrics = MetricCollection(
                {
                    "train_r2": R2Score(),
                    "train_loss": MeanSquaredError(squared=True)
                }
            )
            self.valid_metrics = MetricCollection(
                {
                    "valid_r2": R2Score(),
                    "valid_loss": MeanSquaredError(squared=True)
                }
            )
        self.model = SmilesX(
            vocab_size, lstm_units, dense_units, embedding_dim, return_proba
        )

    def loss(self, y, y_pred):
        return self.loss_func(y, y_pred)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=self.trainer.max_epochs,
            T_mult=int(self.trainer.max_epochs * 0.1)
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, X):
        # モデルの順伝播
        return self.model(X)

    def training_step(self, batch, batch_idx):
        # モデルの学習
        X, y = batch
        output = self.forward(X)
        loss = self.loss(output, y)
        self.train_metrics(output, y)
        self.log_dict(
            dictionary=self.train_metrics,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # モデルの学習
        X, y = batch
        output = self.forward(X)
        loss = self.valid_metrics(output, y)
        self.log_dict(
            dictionary=self.valid_metrics,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        return loss
