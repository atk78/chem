import re
import itertools

from tqdm import tqdm
import numpy as np
import torch


ATOM_LEN1 = ["B", "C", "N", "O", "S", "P", "F", "I"]
ATOM_LEN2 = ["Cl", "Br", "Si"]
AROMA = ["b", "c", "n", "o", "s", "p"]
BRACKET = [r"\[[^\[\]]*\]"]  # ex) [Na+], [nH]
BOND = ["-", "=", "#", "$", "/", ".", "*"]
RING = [str(i) for i in range(1, 10)]
LONG_RING = [r"%\d{2}"]  # ex) %10, %23
SOS = "sos"
EOS = "eos"
UNKNOWN = "unk"
PADDING = "pad"


class Token:
    def __init__(self):
        pattern_len2 = ATOM_LEN2 + BRACKET + LONG_RING
        self.pattern_len2 = re.compile(f'({"|".join(pattern_len2)})')
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2

    def smiles_tokenizer(self, smiles: str):
        smiles = self.pattern_len2.sub(r",\1,", string=smiles)
        smiles = smiles.split(",")
        splitted_smiles = list()
        for chr in smiles:
            if self.pattern_len2.match(chr):
                splitted_smiles.append(chr)
                continue
            else:
                for c in chr:
                    splitted_smiles.append(c)
        return [SOS] + splitted_smiles + [EOS]

    def get_tokens(self, smiles_list: list[str]):
        tokenized_smiles_list = list()
        for i_smiles in tqdm(smiles_list):
            tokenized_smiles_tmp = self.smiles_tokenizer(i_smiles)
            tokenized_smiles_list.append(tokenized_smiles_tmp)
        return tokenized_smiles_list

    @staticmethod
    def extract_vocab(tokenized_smiles_list: list[str]):
        vocab = [SOS, EOS] + list(
            set(itertools.chain.from_iterable(tokenized_smiles_list))
        )
        vocab = sorted(set(vocab), key=vocab.index)
        return vocab

    @staticmethod
    def add_extra_tokens(tokens: list[str]):
        tokens.insert(0, PADDING)
        return tokens

    @staticmethod
    def token2int_dict(tokens: list[str]):
        return dict((c, i) for i, c in enumerate(tokens))

    @staticmethod
    def int2token_dict(tokens: list[str]):
        return dict((i, c) for i, c in enumerate(tokens))

    def smiles2seq(self, tokenized_smiles: torch.Tensor, tokens: list[str]):
        # mol = Chem.MolFromSmiles(smiles)
        # SMILESを[sos(idx=1), ..., eos(idx=2)]で返す
        converter = self.token2int_dict(tokens)
        return torch.tensor([converter[token] for token in tokenized_smiles])

    def seq2smiles(self, seq: torch.Tensor, tokens: list[str]):
        converter = self.int2token_dict(tokens)
        seq = seq[torch.where(
            (seq != self.pad_idx) * (seq != self.eos_idx) * (seq != self.sos_idx)
        )].tolist()
        return "".join([converter[each_int] for each_int in seq])

    def batch_update(
        self,
        tokenized_smiles_list: list[str],
        max_length: int,
        tokens: list[str]
    ):
        converter = self.token2int_dict(tokens)
        int_smiles_array = np.zeros(
            shape=(len(tokenized_smiles_list), max_length), dtype=np.int32
        )
        for idx, ismiles in tqdm(enumerate(tokenized_smiles_list)):
            ismiles_tmp = list()
            pad_len = (max_length - len(ismiles))
            if pad_len >= 0:
                ismiles_tmp = ismiles + [PADDING] * pad_len  # Force output vectors to have same length
            else:
                ismiles_tmp = ismiles[:max_length]  # longer vectors are truncated (to be changed...)
            integer_encoded = [converter[itoken] for itoken in ismiles_tmp]
            int_smiles_array[idx] = integer_encoded
        return torch.from_numpy(np.array(int_smiles_array, dtype=np.int64))
