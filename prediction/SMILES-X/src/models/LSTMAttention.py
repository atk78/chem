import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import MetricCollection, R2Score, MeanAbsoluteError, MeanSquaredError
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .. import utils
from ..dataset import Data


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
        at = self.softmax(et)
        if mask is not None:
            at *= mask.type(torch.float32)
        atx = torch.unsqueeze(input=at, dim=-1)
        ot = x * atx
        if self.return_proba:
            return atx
        else:
            return torch.sum(ot, dim=1)


class LSTMAttention(nn.Module):
    def __init__(
        self,
        token_size,
        lstm_units=16,
        dense_units=16,
        embedding_dim=32,
        return_proba=False,
    ):
        super().__init__()
        self.embedding_layer = nn.Embedding(token_size, embedding_dim)
        self.bilstm_layer = nn.LSTM(
            embedding_dim, lstm_units, bidirectional=True, batch_first=True
        )
        self.timedistributed_dense_layer = TimeDistributedDense(
            2 * lstm_units, dense_units, batch_first=True
        )
        self.attention_layer = Attention(dense_units, return_proba)
        self.output_layer = nn.Linear(dense_units, out_features=1)

    def forward(self, X):
        # モデルの順伝播
        X = self.embedding_layer(X)
        X, _ = self.bilstm_layer(X)
        X = self.timedistributed_dense_layer(X)
        X = self.attention_layer(X)
        X = self.output_layer(X)
        return X


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        token_size,
        learning_rate=1e-3,
        lstm_units=16,
        dense_units=16,
        embedding_dim=32,
        return_proba=False,
        log_flag=False,
        loss_func="MAE",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(*args)
        self.lr = learning_rate
        self.log_flag = log_flag
        if loss_func == "MAE":
            self.loss_func = F.l1_loss
            self.valid_metrics = MetricCollection(
                {"valid_r2": R2Score(), "valid_loss": MeanAbsoluteError()}
            )
        else:
            self.loss_func = F.mse_loss
            self.valid_metrics = MetricCollection(
                {"valid_r2": R2Score(), "valid_loss": MeanSquaredError(squared=True)}
            )

        self.model = LSTMAttention(
            token_size, lstm_units, dense_units, embedding_dim, return_proba
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
        return loss

    def validation_step(self, batch, batch_idx):
        # モデルの学習
        X, y = batch
        output = self.forward(X)
        loss = self.loss(output, y)
        self.valid_metrics(output, y)
        if self.log_flag:
            self.log_dict(
                dictionary=self.valid_metrics,
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=True,
            )
        else:
            self.log(
                name="valid_loss",
                value=loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return loss

    def evaluation_model(self, enum_smiles, enum_prop, card):
        data = Data(enum_smiles, enum_prop)
        if len(enum_prop) < 10000:
            batch_size = len(enum_prop)
        else:
            batch_size = 10000
        dataloader = DataLoader(
            data, batch_size=batch_size, shuffle=False, drop_last=False
        )
        y_pred_list = []
        y_list = []
        for dataset in dataloader:
            x, y = dataset[0], list(dataset[1].detach().numpy().copy().flatten())
            with torch.no_grad():
                y_pred = self.forward(x)
            y_pred = list(y_pred.detach().numpy().copy().flatten())
            y_pred_list.extend(y_pred)
            y_list.extend(y)
        card = np.array(card)
        y_pred, _ = utils.mean_median_result(card, y_pred_list)
        y, _ = utils.mean_median_result(card, y_list)
        mae = mean_absolute_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        r2 = r2_score(y, y_pred)
        return y, y_pred, mae, rmse, r2
