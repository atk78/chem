import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from . import utils


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
        self.inner_dense = nn.Linear(in_features=dense_units, out_features=1)
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
        return_proba=False
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
        log_flag=False
    ):
        super().__init__()
        self.lr = learning_rate
        self.log_flag = log_flag
        self.model = LSTMAttention(
            token_size,
            lstm_units,
            dense_units,
            embedding_dim,
            return_proba
        )

        self.train_metrics_epoch = {
            "R2Score": [], "MeanAbsoluteError": [], "MeanSquaredError": []
        }
        self.valid_metrics_epoch = {
            "R2Score": [], "MeanAbsoluteError": [], "MeanSquaredError": []
        }
        self.train_metrics = MetricCollection([
            R2Score(), MeanAbsoluteError(), MeanSquaredError(squared=True)
        ])
        self.valid_metrics = MetricCollection([
            R2Score(), MeanAbsoluteError(), MeanSquaredError(squared=True)
        ])

    def loss_function(self, y, y_pred):
        return F.mse_loss(y, y_pred)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, X):
        # モデルの順伝播
        return self.model(X)

    def training_step(self, batch, batch_idx):
        # モデルの学習
        X, y = batch
        output = self.forward(X)
        loss = self.loss_function(output, y)
        # if self.log_flag:
        #     self.train_metrics(output, y)
        #     for key, metrics in self.train_metrics.items():
        #         self.train_metrics_epoch[key].append(float(metrics(output, y)))
        return loss

    def validation_step(self, batch, batch_idx):
        # モデルの学習
        X, y = batch
        output = self.forward(X)
        loss = self.loss_function(output, y)
        self.valid_metrics(output, y)
        if self.log_flag:
            self.log_dict(
                self.valid_metrics, on_epoch=True, on_step=False, logger=True, prog_bar=True
            )
        else:
            self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def evaluation_model(self, x, y, card, scaler=None):
        with torch.no_grad():
            y_pred = self.forward(x).detach().numpy()
        card = np.array(card)
        y_pred, _ = utils.mean_median_result(card, y_pred)
        y_pred = y_pred.reshape(-1, 1)
        if scaler:
            y = scaler.inverse_transform(y)
            y_pred = scaler.inverse_transform(y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred) ** 0.5
        r2 = r2_score(y, y_pred)
        return y, y_pred, mae, rmse, r2
