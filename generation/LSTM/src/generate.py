import torch
import torch.nn.functional as F


def generate(model, vocab, sample_size=1, max_len=100, smiles=True):
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
    device = next(model.parameters()).device
    with torch.no_grad():
        # 推論モードにすることでドロップアウトなどの機能をオフにする
        model.eval()
        in_seq_one_hot = (
            F.one_hot(
                torch.tensor(
                    [[vocab.sos_idx]] * sample_size), num_classes=model.lstm.input_size,
                ).to(torch.float)
            .to(device)
        )
        # 隠れ状態hの初期値
        h = torch.zeros(
            model.lstm.num_layers, sample_size, model.lstm.hidden_size
        ).to(device)
        # 細胞状態cの初期値
        c = torch.zeros(
            model.lstm.num_layers, sample_size, model.lstm.hidden_size
        ).to(device)
        out_seq_one_hot = in_seq_one_hot.clone()
        out = in_seq_one_hot
        for _ in range(max_len):
            out, (h, c) = model.lstm(out, (h, c))
            # out_activation => nn.Softmax(dim=2)
            out = model.out_activation(model.out_linear(out))
            # OneHotCategoricalでワンホットベクトル化
            out = model.out_dist_cls(probs=out).sample()
            # 得られた結果を追記して出力系統を保存する
            out_seq_one_hot = torch.cat((out_seq_one_hot, out), dim=1)
        # model.train()
        if smiles:
            # 出力からSMILESに変換
            return [
                vocab.seq2smiles(each_onehot)
                for each_onehot in torch.argmax(out_seq_one_hot, dim=2)
            ]
        return out_seq_one_hot
