import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool

from .. import utils


class MolecularGCN(nn.Module):
    def __init__(
        self, n_features, n_conv_hidden=3, n_mlp_hidden=3, dim=64, drop_rate=0.1
    ):
        """
        分子構造に対して畳み込みグラフニューラルネットワークを実施する。pytorch_geometricを用いている。
        DeepChemを参考にして作成した。
        URL：https://iwatobipen.wordpress.com/2019/04/05/make-graph-convolution-model-with-geometric-deep-learning-extension-library-for-pytorch-rdkit-chemoinformatics-pytorch/

        Parameters
        ----------
        n_features : int
            特徴量の数
        n_conv_hidden : int, optional
            畳み込み層の数, by default 3
        n_mlp_hidden : int, optional
            隠れ層の数, by default 3
        dim : int, optional
            隠れ層の途中の次元数, by default 64
        drop_rate : float, optional
            ドロップアウトさせるデータの割合, by default 0.1
        """
        super().__init__()
        self.n_features = n_features  # 特徴量の数
        self.n_conv_hidden = n_conv_hidden  # 畳み込み層の数
        self.n_mlp_hidden = n_mlp_hidden  # 隠れ層の数
        self.dim = dim  # 隠れ層の次元の数
        self.drop_rate = drop_rate  # ドロップアウトするデータの割合
        self.graphconv1 = GCNConv(
            in_channels=self.n_features, out_channels=self.dim
        )  # 最初のグラフ畳み込み層
        self.bn1 = nn.BatchNorm1d(num_features=self.dim)  # 最初のBatchNormalization
        self.graphconv_hidden = nn.ModuleList(
            [
                GCNConv(in_channels=self.dim, out_channels=self.dim, cached=False)
                for _ in range(self.n_conv_hidden)
            ]
        )
        self.bn_conv_hidden = nn.ModuleList(
            [nn.BatchNorm1d(num_features=self.dim) for _ in range(self.n_conv_hidden)]
        )
        self.mlp_hidden = nn.ModuleList(
            [
                nn.Linear(in_features=self.dim, out_features=self.dim)
                for _ in range(self.n_mlp_hidden)
            ]
        )
        self.bn_mlp = nn.ModuleList(
            [nn.BatchNorm1d(num_features=self.dim) for _ in range(self.n_mlp_hidden)]
        )
        self.mlp_out = nn.Linear(self.dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(input=self.graphconv1(x, edge_index))
        x = self.bn1(x)
        for graphconv, bn_conv in zip(self.graphconv_hidden, self.bn_conv_hidden):
            x = graphconv(x, edge_index)
            x = bn_conv(x)
        x = global_add_pool(x=x, batch=data.batch)
        for fc_mlp, bn_mlp in zip(self.mlp_hidden, self.bn_mlp):
            x = F.relu(fc_mlp(x))
            x = bn_mlp(x)
            if self.drop_rate > 0:
                x = F.dropout(input=x, p=self.drop_rate, training=self.training)
        x = self.mlp_out(x)
        return x


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        n_features,
        n_conv_hidden=3,
        n_mlp_hidden=3,
        dim=64,
        drop_rate=0.1,
        learning_rate=1e-3,
        log_flag=False,
        loss_func="MAE",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lr = learning_rate
        self.log_flag = log_flag
        self.loss_func = loss_func
        self.model = MolecularGCN(
            n_features,
            n_conv_hidden=n_conv_hidden,
            n_mlp_hidden=n_mlp_hidden,
            dim=dim,
            drop_rate=drop_rate
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
        if self.loss_func == "MAE":
            return F.l1_loss(y, y_pred)
        else:
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
        return loss

    def validation_step(self, batch, batch_idx):
        # モデルの学習
        X, y = batch
        output = self.forward(X)
        loss = self.loss_function(output, y)
        self.valid_metrics(output, y)
        if self.log_flag:
            self.log_dict(
                dictionary=self.valid_metrics,
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=True
            )
        else:
            self.log(
                name="valid_loss",
                value=loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True
            )
        return loss

    def evaluation_model(self, x, y, card):
        with torch.no_grad():
            y_pred = self.forward(x).detach().numpy()
        card = np.array(card)
        y_pred, _ = utils.mean_median_result(card, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred) ** 0.5
        r2 = r2_score(y, y_pred)
        return y, y_pred, mae, rmse, r2
