import torch
from torch import nn, tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import OneHotCategorical
from tqdm import tqdm


class SmilesLSTM(nn.Module):
    def __init__(self, vocab, hidden_size, n_layers):
        """SMILESをLSTMで学習するためのモデルの初期化

        Parameters
        ----------
        vocab : int
            入力データの次元（=|X|）
        hidden_size : int
            隠れ状態の次元
        n_layers : int
            層数
        """
        super().__init__()
        self.vocab = vocab
        vocab_size = len(self.vocab.char_list)
        # LSTM本体の定義→入力データの次元、隠れ状態の次元、層数を指定
        # batch_first = True→入力データや隠れ状態のTensorの次元の定義を
        # (batch_size) × (系列長) × (語彙サイズ)にする
        self.lstm = nn.LSTM(
            input_size=vocab_size,   # 入力データの次元 語彙サイズ
            hidden_size=hidden_size,  # 隠れ状態の次元
            num_layers=n_layers,     # 層の数
            batch_first=True
        )
        # LSTMの最終層の隠れ状態から関数近似器の出力を計算するための線形層
        # 隠れ状態の次元はhidden_size, 関数近似器の出力の次元はvocab_size
        self.out_linear = nn.Linear(hidden_size, vocab_size)
        # 活性化関数の定義 受け取るTensorのサイズは(batch_size)×(系列長)×(語彙サイズ)
        # 最後の語彙サイズの大きさの次元についてSoftmax関数を適用したい→引数:2
        self.out_activation = nn.Softmax(dim=2)
        # 予測分布の条件付確率分布。ワンホットベクトルに従う確率分布にするためOneHotCategoricalを用いる
        self.out_dist_cls = OneHotCategorical

    def forward(self, in_seq):
        """入力系統：in_seq（整数値の系列）をネットワークに通して得られる出力を計算
        1. 整数値の系列である入力系統in_seqをワンホットベクトルに変換
        2. ワンホットベクトルをLSTMモデルに入力
        3. 各時刻における最終層の隠れ状態系列と最終時刻における各層のhとcを出力
        4. 関数近似器の出力を計算するためにout_linearに通す

        Parameters
        ----------
        X : Tensor[int]
            SMILESから変換された整数値の系列Tensor

        Returns
        -------
        torch.Tensor
            LSTMモデルの最終時刻における各層のhとcが出力
            ネットワークの出力を計算する際にはc(LSTMの細胞状態、つまり記憶部分)は必要なく、h(LSTM細胞の出力)のみ用いる。
            出力サイズは(バッチサイズ)×(系列長)×(語彙サイズ)
        """
        one_hot_seq = nn.functional.one_hot(
            in_seq,
            num_classes=self.lstm.input_size
        ).to(torch.float)
        output, _ = self.lstm(one_hot_seq)
        output = self.out_linear(output)
        return output

    def loss(self, in_seq, out_seq):
        """入力・出力の整列系列 in_seq, out_seqを受け取り、損失関数の値を返す

        Parameters
        ----------
        in_seq : torch.Tensor
            入力系列
        out_seq : torch.Tensor
            出力系列

        Returns
        -------
        float
            損失関数によって計算された損失
        """
        return self.loss_func(
            # forwardメソッドの出力は(バッチサイズ)×(系列長)×(語彙サイズ)
            # nn.CrossEntropyLossは(バッチサイズ)×(語彙サイズ)×(系列長)を想定
            # transpose(1, 2)で入れ替え
            self.forward(in_seq).transpose(1, 2),
            out_seq)

    def generate(self, sample_size=1, max_len=100, smiles=True):
        """訓練したSMILES-LSTMを用いてSMILES系列を生成する
        系列を生成する際には「1文字ずつ生成しては得られた文字を再度入力して次の文字を生成」する
        このときLSTMの隠れ状態や細胞状態を引き継ぐ必要がある。隠れ状態はh、細胞状態はc

        Parameters
        ----------
        sample_size : int, optional
            生成するサンプルの数, by default 1
        max_len : int, optional
            生成する系列の系列長の最大値。
            系列生成には<eos>が出るまで生成を続けるが、学習がうまくいっていない場合は,
            いつまで経っても<eox>が出ずに生成が終了しない時があるため。, by default 100
        smiles : bool, optional
            生成した結果をSMILES系列に変換するか
            ワンホットベクトルの系列のまま出力するかを選択するための引数。, by default True

        Returns
        -------
        _type_
            _description_
        """
        device = next(self.parameters()).device
        with torch.no_grad():
            # 推論モードにすることでドロップアウトなどの機能をオフにする
            self.eval()
            in_seq_one_hot = nn.functional.one_hot(
                tensor([[self.vocab.sos_idx]] * sample_size),
                num_classes=self.lstm.input_size).to(
                    torch.float).to(device)
            # 隠れ状態hの初期値
            h = torch.zeros(
                self.lstm.num_layers,
                sample_size,
                self.lstm.hidden_size).to(device)
            # 細胞状態cの初期値
            c = torch.zeros(
                self.lstm.num_layers,
                sample_size,
                self.lstm.hidden_size).to(device)
            out_seq_one_hot = in_seq_one_hot.clone()
            out = in_seq_one_hot
            for _ in range(max_len):
                out, (h, c) = self.lstm(out, (h, c))
                # out_activation => nn.Softmax(dim=2)
                out = self.out_activation(self.out_linear(out))
                # OneHotCategoricalでワンホットベクトル化
                out = self.out_dist_cls(probs=out).sample()
                # 得られた結果を追記して出力系統を保存する
                out_seq_one_hot = torch.cat(
                    (out_seq_one_hot, out), dim=1)
            self.train()
            if smiles:
                # 出力からSMILESに変換
                return [self.vocab.seq2smiles(each_onehot)
                        for each_onehot
                        in torch.argmax(out_seq_one_hot, dim=2)]
            return out_seq_one_hot
