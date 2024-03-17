import torch
from torch import nn, tensor
from torch.distributions import Categorical
import pytorch_lightning as pl


class LitSmilesVAE(pl.LightningModule):
    def __init__(
        self,
        vocab,
        latent_dim,
        emb_dim=128,
        max_len=100,
        encoder_params={"hidden_size": 128, "num_layers": 1, "dropout": 0.0},
        decoder_params={"hidden_size": 128, "num_layers": 1, "dropout": 0.0},
        encoder2out_params={"out_dim_list": [128, 128]},
        beta_schedule=None,
        learning_rate=1e-3
    ):
        """分子生成オートエンコーダは観測変数xを分子に相当するものにして分子生成モデルを作ることとなる。
        モデルの設計の自由度として
        1. 観測変数(x)：分子の表現→今回はSMILES
        2. エンコーダ(事後分布p(z|x)を近似したq(z|x))→平均μと分散σ^2のニューラルネットワークを指定する。今回は近似事後分布にLSTMを用いた。
        3. デコーダ(p(x|z))→潜在ベクトルzを受け取り、SMILES系列に対応する確率分布を定義。今回は確率分布にLSTM
        があげられる。

        Parameters
        ----------
        vocab : list[str]
            語彙が保存されたリスト
        latent_dim : int
            潜在ベクトルのサイズ
        emb_dim : int, optional
            埋め込み層(one-hot-vectorをいい感じにディープラーニングに使えるようにする)のサイズ, by default 128
        max_len : int, optional
            系列長 この値に合わせるように埋め込みベクトルを作成する, by default 100
        encoder_params : dict, optional
            エンコーダ(LSTM)の隠れ層の層数やユニット数、ドロップアウト率のパラメータ, by default {'hidden_size': 128, 'num_layers': 1, 'dropout': 0.}
        decoder_params : dict, optional
            デコーダ(LSTM)の隠れ層の層数やユニット数、ドロップアウト率のパラメータ, by default {'hidden_size': 128, 'num_layers': 1, 'dropout': 0.}
        encoder2out_params : dict, optional
            _description_, by default {'out_dim_list': [128, 128]}
        """
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.lr = learning_rate
        self.loss_func = nn.CrossEntropyLoss(reduction="none")
        self.beta_schedule = beta_schedule

        self.vocab = vocab
        vocab_size = len(self.vocab.char_list)
        self.max_len = max_len
        self.latent_dim = latent_dim
        self.beta = 1.0
        # 埋め込みベクトル（SMILES->整数系列に変換した入力）を作成するインスタンス
        # vocab_size×emb_dimの出力サイズ。paddingはself.bad_idxで行う
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=vocab.pad_idx)
        # batch_fisrt=True -> (batch_size)x(系列長)x(語彙サイズ)
        self.encoder = nn.LSTM(emb_dim, batch_first=True, **encoder_params)
        # エンコーダ(LSTM)の出力の末尾(last_out)の次元(in_dim)をout_dim_list[-1]の次元に変換する
        # エンコーダのLSTMの出力を変換する多層ニューラルネットワーク
        self.encoder2out = nn.Sequential()
        in_dim = (
            encoder_params["hidden_size"] * 2
            if encoder_params.get("bidirectional", False)
            else encoder_params["hidden_size"]
        )
        for each_out_dim in encoder2out_params["out_dim_list"]:
            self.encoder2out.append(nn.Linear(in_dim, each_out_dim))
            self.encoder2out.append(nn.Sigmoid())
            in_dim = each_out_dim
        # encoder2outの出力を潜在空間上の正規分布の平均に変換する線形モデル
        self.encoder_out2mu = nn.Linear(in_dim, latent_dim)
        # encoder2outの出力を潜在空間上の正規分布の分散共分散行列の対角成分に変換する線形モデル
        self.encoder_out2logvar = nn.Linear(in_dim, latent_dim)
        # 潜在ベクトルをデコーダであるLSTMの細胞状態に変換するモデル
        self.latent2dech = nn.Linear(
            in_features=latent_dim,
            out_features=decoder_params["hidden_size"] * decoder_params["num_layers"]
        )
        # 潜在ベクトルをデコーダであるLSTMの隠れ状態に変換するモデル
        self.latent2decc = nn.Linear(
            in_features=latent_dim,
            out_features=decoder_params["hidden_size"] * decoder_params["num_layers"]
        )
        self.latent2emb = nn.Linear(in_features=latent_dim, out_features=emb_dim)
        self.decoder = nn.LSTM(
            input_size=emb_dim, batch_first=True, bidirectional=False, **decoder_params
        )
        self.decoder2vocab = nn.Linear(decoder_params["hidden_size"], vocab_size)
        self.out_dist_cls = Categorical

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def encode(self, in_seq):
        """
        SMILES系列を整数値テンソルで表したin_seqを受け取り、潜在空間上の正規分布の平均と分散共分散行列の対角成分の対数の値を返す
        1. 整数値テンソルを埋め込みベクトル系列に変換
        2. 埋め込みベクトルをエンコーダ(LSTM)に入力し、隠れ状態の系列h=out_seq(サンプルサイズx系列長x隠れ状態の次元)を受け取る
        3. 隠れ状態の系列hの最後の要素(out_seq[:,-1,:] 入力系列すべてを反映した隠れ状態)を
        順伝播型ニューラルネットワークに入力し、エンコーダの出力zの従う正規分布の平均μと、分散共分散行列の対角成分σ^2を出力する
        4. z ~ N(μ, diag(σ^2))を生成し、エンコーダの出力とする

        Parameters
        ----------
        in_seq : str
            変換される文字列（ここではSMILES。Selfiesも可能）

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            エンコーダによって生成された平均と分散
        """
        in_seq_emb = self.embedding(in_seq)
        out_seq, (h, c) = self.encoder(in_seq_emb)
        last_out = out_seq[:, -1, :]
        out = self.encoder2out(last_out)
        return (self.encoder_out2mu(out), self.encoder_out2logvar(out))

    def reparam(self, mu, logvar, deterministic=False):
        """再パラメータ化を行うメソッド。
        encoderメソッドの出力である正規分布のパラメータ(μ, σ^2)を受け取って、
        その正規分布からサンプリングした値を返す。
        単純にサンプリングすると得られた値はencoderのパラメータについて微分することができないため、
        再パラメータ化にもとづいたサンプリングを行う。（deterministic=True）

        Parameters
        ----------
        mu : torch.Tensor
            エンコーダの出力zに従う正規分布の平均μ
        logvar : torch.Tensor
            エンコーダの出力zに従う正規分布の分散共分散の対角成分σ^2
        deterministic : bool, optional
            再パラメータを実行するかのフラグ, by default False

        Returns
        -------
        torch.Tensor
            再パラメータ化にもとづいてサンプリングを行われた潜在ベクトル
        """
        std = torch.exp(0.5 * logvar)
        # パラメータに依存したい確率変数Z0
        eps = torch.randn_like(std)
        if deterministic:
            return mu
        else:
            return mu + std * eps

    def decode(self, z, out_seq=None, deterministic=False):
        """潜在ベクトルzを受け取り、それに対応するSMILES系列とその対数尤度を返す。
        out_seqがない場合、正解のSMILES系列がない場合のデコードに相当。

        Parameters
        ----------
        z : torch.Tensor
            エンコーダによって出力された潜在ベクトル
        out_seq : _type_, optional
            エンコーダによって得られた隠れ状態の系列（float）、Noneの場合は, by default None
        deterministic : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        batch_size = z.shape[0]
        # デコードに用いるLSTMの隠れ状態hと細胞状態cを計算する
        h_unstructured = self.latent2dech(z)
        c_unstructured = self.latent2decc(z)
        h = torch.stack(
            [
                h_unstructured[:, each_idx : each_idx + self.decoder.hidden_size]
                for each_idx in range(0, h_unstructured.shape[1], self.decoder.hidden_size)
            ]
        )
        c = torch.stack(
            [
                c_unstructured[:, each_idx : each_idx + self.decoder.hidden_size]
                for each_idx in range(0, c_unstructured.shape[1], self.decoder.hidden_size)
            ]
        )
        # ここまで ###############################################
        if out_seq is None:
            with torch.no_grad():
                in_seq = torch.tensor(
                    [[self.vocab.sos_idx]] * batch_size, device=self.device
                )
                out_logit_list = []
                for each_idx in range(self.max_len):
                    in_seq_emb = self.embedding(in_seq)
                    out_seq, (h, c) = self.decoder(in_seq_emb[:, -1:, :], (h, c))
                    out_logit = self.decoder2vocab(out_seq)
                    out_logit_list.append(out_logit)
                    if deterministic:
                        out_idx = torch.argmax(out_logit, dim=2)
                    else:
                        out_prob = nn.functional.softmax(out_logit, dim=2)
                        out_idx = self.out_dist_cls(probs=out_prob).sample()
                    in_seq = torch.cat((in_seq, out_idx), dim=1)
                return torch.cat(out_logit_list, dim=1), in_seq
        else:
            out_seq_emb = self.embedding(out_seq)
            out_seq_emb_out, _ = self.decoder(out_seq_emb, (h, c))
            out_seq_vocab_logit = self.decoder2vocab(out_seq_emb_out)
            return out_seq_vocab_logit[:, :-1], out_seq[:-1]

    def forward(self, in_seq, out_seq=None, deterministic=False):
        mu, logvar = self.encode(in_seq)
        z = self.reparam(mu, logvar, deterministic=deterministic)
        out_seq_logit, _ = self.decode(z, out_seq, deterministic=deterministic)
        return out_seq_logit, mu, logvar

    def loss(self, out_seq_logit, mu, logvar, out_seq):
        """KL情報量を最小化にするための損失関数

        Parameters
        ----------
        out_seq_logit : decoderにより出力された
            _description_
        mu : _type_
            _description_
        logvar : _type_
            _description_
        out_seq : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # out_seq_logit, mu, logvar = self.forward(in_seq, out_seq)
        neg_likelihood = self.loss_func(out_seq_logit.transpose(1, 2), out_seq[:, 1:])
        neg_likelihood = neg_likelihood.sum(axis=1).mean()
        kl_div = -0.5 * (1.0 + logvar - mu**2 - torch.exp(logvar)).sum(axis=1).mean()
        return neg_likelihood + self.beta * kl_div

    def on_train_epoch_start(self) -> None:
        try:
            self.beta = self.beta_schedule[self.trainer.current_epoch]
        except:
            pass

    def training_step(self, batch, batch_idx):
        in_seq, out_seq = batch
        out_seq_logit, mu, logvar = self.forward(in_seq, out_seq)
        train_loss = self.loss(out_seq_logit, mu, logvar, out_seq)
        self.log("train_loss", train_loss, prog_bar=True, on_step=True, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        in_seq, out_seq = batch
        out_seq_logit, mu, logvar = self.forward(in_seq, out_seq)
        valid_loss = self.loss(out_seq_logit, mu, logvar, out_seq)
        success = self.reconstruct(in_seq=in_seq, verbose=False)
        reconstruct_rate = sum(success) / len(success)
        self.log_dict(
            dictionary={
                "valid_loss": valid_loss,
                "success_rate": reconstruct_rate
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True
        )
        # self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
        return valid_loss

    def generate(self, z=None, sample_size=None, deterministic=False):
        if z is None:
            z = torch.randn(sample_size, self.latent_dim)
        with torch.no_grad():
            self.eval()
            _, out_seq = self.decode(z, deterministic=deterministic)
            out = [self.vocab.seq2smiles(each_seq) for each_seq in out_seq]
            self.train()
            return out

    def reconstruct(self, in_seq, deterministic=True, max_reconstruct=None, verbose=True):
        """SMILES系列の集合in_seqを受け取り、SmilesVAEを用いてそれらを再構成できるかのメソッド

        Parameters
        ----------
        in_seq : _type_
            _description_
        deterministic : bool, optional
            generateメソッドと同様, by default True
        max_reconstruct : _type_, optional
            _description_, by default None
        verbose : bool, optional
            Trueの場合、入力したSMILES系列と再構成したSMILES系列を表示する, by default True

        Returns
        -------
        _type_
            _description_
        """
        self.eval()
        if max_reconstruct is not None:
            in_seq = in_seq[:max_reconstruct]
        mu, logvar = self.encode(in_seq)
        z = self.reparam(mu, logvar, deterministic=deterministic)
        _, out_seq = self.decode(z, deterministic=deterministic)

        success_list = []
        for each_idx, each_seq in enumerate(in_seq):
            truth = self.vocab.seq2smiles(each_seq)[::-1]
            pred = self.vocab.seq2smiles(out_seq[each_idx])
            success_list.append(truth==pred)
            if verbose:
                print(f"{truth==pred}\t{truth} --> {pred}")
        self.train()
        return success_list


