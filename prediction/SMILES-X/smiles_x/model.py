import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import (
    MetricCollection, R2Score, MeanAbsoluteError, MeanSquaredError
)


class TimeDistributedDense(nn.Module):
    def __init__(self, lstm_untis, dense_units, batch_first: bool = False):
        super().__init__()
        self.dense = nn.Linear(lstm_untis, dense_units)
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-1)
        )  # (samples * timesteps, input_size)
        y = self.dense(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.size(0), -1, y.size(-1)
            )  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class Attention(nn.Module):
    def __init__(self, dense_units, return_proba=False):
        super().__init__()
        self.return_proba = return_proba
        self.inner_dense = nn.Linear(
            in_features=dense_units, out_features=1
        )  # nn.bmmでも可
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask=None):
        et = self.inner_dense(x)
        et = self.tanh(et)
        et = torch.squeeze(input=et)
        # et = et.view(et.shape[0], -1)
        at = self.softmax(et)
        if mask is not None:
            at *= mask.type(torch.float32)
        atx = torch.unsqueeze(input=at, dim=-1)
        # atx = at.view(at.shape[0], at.shape[1], 1)
        ot = x * atx
        if self.return_proba:
            return atx
        else:
            return torch.sum(ot, dim=1)


class SmilesX(nn.Module):
    def __init__(
        self,
        vocab_size,
        lstm_units=16,
        dense_units=16,
        embedding_dim=32,
        return_proba=False,
    ):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm_layer = nn.LSTM(
            embedding_dim, lstm_units, bidirectional=True, batch_first=True
        )
        self.timedistributed_dense_layer = TimeDistributedDense(
            2 * lstm_units, dense_units, batch_first=True
        )
        self.attention_layer = Attention(dense_units, return_proba)
        self.output_layer = nn.Linear(dense_units, out_features=1)

    def forward(self, X):
        X = self.embedding_layer(X)
        X, _ = self.bilstm_layer(X)
        X = self.timedistributed_dense_layer(X)
        X = self.attention_layer(X)
        X = self.output_layer(X)
        return X


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
