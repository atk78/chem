import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import global_add_pool


class MolecularGCN(nn.Module):
    def __init__(
        self,
        n_features: int,
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
        self.graph_conv_hidden = nn.Sequential()
        self.bn_conv_hidden = nn.Sequential()
        in_dim = n_features
        for _ in range(n_conv_hidden_layer):
            self.graph_conv_hidden.append(
                GCNConv(in_channels=in_dim, out_channels=graph_dim),
            )
            self.bn_conv_hidden.append(
                nn.BatchNorm1d(num_features=graph_dim)
            )
            in_dim = graph_dim
        self.dense_hidden = nn.Sequential()
        self.bn_hidden = nn.Sequential()
        in_dim = graph_dim
        for _ in range(n_dense_hidden_layer):
            self.dense_hidden.append(
                nn.Linear(in_features=in_dim, out_features=dense_dim)
            ),
            self.bn_hidden.append(nn.BatchNorm1d(num_features=dense_dim))
            in_dim = dense_dim
        self.dense_out = nn.Linear(dense_dim, 1)


    def forward(self, data, training=True):
        x, edge_index = data.x, data.edge_index
        for graph_conv, bn_conv in zip(
            self.graph_conv_hidden, self.bn_conv_hidden
        ):
            x = F.relu(graph_conv(x, edge_index))
            x = bn_conv(x)
        x = global_add_pool(x=x, batch=data.batch)
        for dense_hidden, bn_hidden in zip(self.dense_hidden, self.bn_hidden):
            x = F.relu(dense_hidden(x))
            x = bn_hidden(x)
            if self.drop_rate > 0:
                x = F.dropout(input=x, p=self.drop_rate, training=training)
        x = self.dense_out(x)
        return x


class MolecularGAT(nn.Module):
    def __init__(
        self,
        n_features: int,
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
        self.graph_conv_hidden = nn.Sequential()
        self.bn_conv_hidden = nn.Sequential()
        in_dim = n_features
        for _ in range(n_conv_hidden_layer):
            self.graph_conv_hidden.append(
                GATv2Conv(in_channels=in_dim, out_channels=graph_dim),
            )
            self.bn_conv_hidden.append(
                nn.BatchNorm1d(num_features=graph_dim)
            )
            in_dim = graph_dim
        self.dense_hidden = nn.Sequential()
        self.bn_hidden = nn.Sequential()
        in_dim = graph_dim
        for _ in range(n_dense_hidden_layer):
            self.dense_hidden.append(
                nn.Linear(in_features=in_dim, out_features=dense_dim)
            ),
            self.bn_hidden.append(nn.BatchNorm1d(num_features=dense_dim))
            in_dim = dense_dim
        self.dense_out = nn.Linear(dense_dim, 1)

    def forward(self, data, training=True):
        x, edge_index = data.x, data.edge_index
        for graph_conv, bn_conv in zip(
            self.graph_conv_hidden, self.bn_conv_hidden
        ):
            x = F.relu(graph_conv(x, edge_index))
            x = bn_conv(x)
        x = global_add_pool(x=x, batch=data.batch)
        for dense_hidden, bn_hidden in zip(self.dense_hidden, self.bn_hidden):
            x = F.relu(dense_hidden(x))
            x = bn_hidden(x)
            if self.drop_rate > 0:
                x = F.dropout(input=x, p=self.drop_rate, training=training)
        x = self.dense_out(x)
        return x
