import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import MetricCollection, R2Score, MeanAbsoluteError, MeanSquaredError
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MolecularGCN(nn.Module):
    def __init__(
        self,
        n_features,
        n_conv_hidden_layer=3,
        n_dense_hidden_layer=3,
        graph_dim=64,
        dense_dim=64,
        drop_rate=0.1
    ):
        """
        分子構造に対して畳み込みグラフニューラルネットワークを実施する。pytorch_geometricを用いている。
        DeepChemを参考にして作成した。
        URL：https://iwatobipen.wordpress.com/2019/04/05/make-graph-convolution-model-with-geometric-deep-learning-extension-library-for-pytorch-rdkit-chemoinformatics-pytorch/

        Parameters
        ----------
        n_features : int
            特徴量の数
        n_conv_hidden_layer : int, optional
            畳み込み層の数, by default 3
        n_dense_hidden_layer : int, optional
            隠れ層の数, by default 3
        dim : int, optional
            隠れ層の途中の次元数, by default 64
        drop_rate : float, optional
            ドロップアウトさせるデータの割合, by default 0.1
        """
        super().__init__()
        self.drop_rate = drop_rate  # ドロップアウトするデータの割合
        self.graphconv1 = GCNConv(
            in_channels=n_features, out_channels=graph_dim
        )  # 最初のグラフ畳み込み層
        self.bn1 = nn.BatchNorm1d(num_features=graph_dim)  # 最初のBatchNormalization
        self.graphconv_hidden = nn.ModuleList(
            [
                GCNConv(in_channels=graph_dim, out_channels=graph_dim, cached=False)
                for _ in range(n_conv_hidden_layer)
            ]
        )
        self.bn_conv_hidden_layer = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=graph_dim)
                for _ in range(n_conv_hidden_layer)
            ]
        )
        self.mlp = nn.Linear(
            in_features=graph_dim, out_features=dense_dim
        )
        self.bn2 = nn.BatchNorm1d(num_features=dense_dim)
        self.dense_hidden = nn.ModuleList(
            [
                nn.Linear(in_features=dense_dim, out_features=dense_dim)
                for _ in range(n_dense_hidden_layer)
            ]
        )
        self.bn_mlp = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=dense_dim)
                for _ in range(n_dense_hidden_layer)
            ]
        )
        self.dense_out = nn.Linear(dense_dim, 1)

    def forward(self, data, training=True):
        x, edge_index = data.x, data.edge_index
        x = F.relu(input=self.graphconv1(x, edge_index))
        x = self.bn1(x)
        for graphconv, bn_conv in zip(
            self.graphconv_hidden, self.bn_conv_hidden_layer
        ):
            x = graphconv(x, edge_index)
            x = bn_conv(x)
        x = global_add_pool(x=x, batch=data.batch)
        x = F.relu(self.mlp(x))
        x = self.bn2(x)
        if self.drop_rate > 0:
            x = F.dropout(input=x, p=self.drop_rate, training=training)
        for fc_mlp, bn_mlp in zip(self.dense_hidden, self.bn_mlp):
            x = F.relu(fc_mlp(x))
            x = bn_mlp(x)
            if self.drop_rate > 0:
                x = F.dropout(input=x, p=self.drop_rate, training=training)
        x = self.dense_out(x)
        return x


class LightningModel(L.LightningModule):
    def __init__(
        self,
        n_features,
        n_conv_hidden_layer=3,
        n_dense_hidden_layer=3,
        graph_dim=64,
        dense_dim=64,
        drop_rate=0.1,
        learning_rate=1e-3,
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
                {"train_r2": R2Score(), "train_loss": MeanAbsoluteError()}
            )
            self.valid_metrics = MetricCollection(
                {"valid_r2": R2Score(), "valid_loss": MeanAbsoluteError()}
            )
        else:
            self.loss_func = F.mse_loss
            self.train_metrics = MetricCollection(
                {"train_r2": R2Score(), "train_loss": MeanSquaredError(squared=True)}
            )
            self.valid_metrics = MetricCollection(
                {"valid_r2": R2Score(), "valid_loss": MeanSquaredError(squared=True)}
            )
        self.model = MolecularGCN(
            n_features, n_conv_hidden_layer, n_dense_hidden_layer, graph_dim, dense_dim, drop_rate
        )

    def loss(self, y, y_pred):
        return self.loss_func(y, y_pred)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        t = int(self.trainer.max_epochs * 0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=self.trainer.max_epochs,
            T_mult=t if t > 0 else 1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, X, training=True):
        # モデルの順伝播
        return self.model.forward(X, training)

    def training_step(self, batch, batch_idx):
        # モデルの学習
        X, y = batch, batch.y
        output = self.forward(X, training=True)
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
        X, y = batch, batch.y
        output = self.forward(X, training=False)
        loss = self.valid_metrics(output, y)
        self.log_dict(
            dictionary=self.valid_metrics,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        return loss
