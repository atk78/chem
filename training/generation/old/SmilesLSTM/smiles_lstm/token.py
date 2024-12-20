from tqdm import tqdm

import torch
from torch import nn


pad = " "  # 空文字
sos = "!"  # Start of Sentence : 文章の最初
eos = "?"  # End of Sentence : 文章の最後
pad_idx = 0  # 空文字のindex
sos_idx = 1  # sosのindex
eos_idx = 2  # eosのindex


class Token:
    char_list = [pad, sos, eos]


def update(smiles):
    """SMILES系列を受け取り、char_listを更新したうえでsmiles2seqによって対応する整数系列を返す

    Parameters
    ----------
    smiles : str
        変換するSMILES系列

    Returns
    -------
    torch.Tensor
        SMILESに対応する整数系列
    """
    char_set = set(smiles)
    char_set = char_set - set(Token.char_list)
    Token.char_list += sorted(list(char_set))
    return smiles2seq(smiles, Token.char_list)


def smiles2seq(smiles, vocab):
    """SMILES系列を受け取り、対応する整数系列を返す
    （開始記号、終了記号付き、torch.Tensor型）

    Parameters
    ----------
    smiles : str
        変換したSMILES系列

    Returns
    -------
    torch.Tensor
        SMILESに対応する整数系列
    """
    # mol = Chem.MolFromSmiles(smiles)
    # SMILESを[sos(idx=1), ..., eos(idx=2)]で返す
    return torch.tensor(
        [sos_idx]
        + [vocab.index(each_char) for each_char in smiles]
        + [eos_idx]
    )


def seq2smiles(seq, vocab, wo_special_char=True):
    """整数系列seqを受け取り、対応するSMILES系列を返す。標準では特殊記号は含めない。

    Parameters
    ----------
    seq : torch.Tensor
        SMILESに変換する整数系列
    wo_special_char : bool, optional
        特殊記号を含めるかどうかのフラグ, by default True

    Returns
    -------
    str
        整数系列から変換されたSMILES系列
    """
    if wo_special_char:
        # seqが空文字でない かつ sosでない かつ eosでない
        # →空文字、sos、eosを削除
        return seq2smiles(seq[torch.where(
            (seq != pad_idx) * (seq != sos_idx) * (seq != eos_idx)
        )], vocab, wo_special_char=False)
    # すべて結合してSMILESに変換
    return "".join([
        vocab[each_idx] for each_idx in seq
    ])


def batch_update(smiles_list):
    """SMILES系列のリストを受け取り、それぞれにupdateを適用したうえで、空文字を加えて長さを揃えた整数系列を返す。

    Parameters
    ----------
    smiles_list : list[str]
        SMILESが保存されたlist

    Returns
    -------
    right_padded_batch_seq: str
        paddingにより長さを揃えた整数系列
    """
    seq_list = []
    for each_smiles in tqdm(smiles_list):
        if each_smiles.endswith("\n"):
            each_smiles = each_smiles.strip()
        seq_list.append(update(each_smiles))

    right_padded_batch_seq = nn.utils.rnn.pad_sequence(
        seq_list,
        batch_first=True,
        padding_value=pad_idx
    )
    return right_padded_batch_seq
