import os
import re
import yaml
import logging
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
        return sorted(set(vocab), key=vocab.index)

    @staticmethod
    def add_extra_tokens(tokens: list[str], vocab_size: int):
        tokens.insert(0, UNKNOWN)
        tokens.insert(0, PADDING)
        vocab_size = vocab_size + 2
        return tokens, vocab_size

    def get_vocab(self, save_dir: str, tokens: list[str]):
        vocab = self.extract_vocab(tokens)
        vocab_file = os.path.join(save_dir, "Vocabulary.yaml")
        with open(vocab_file, "w") as f:
            yaml.dump(list(vocab), f)
        return vocab

    @staticmethod
    def token2int_dict(tokens: list[str]):
        return dict((c, i) for i, c in enumerate(tokens))

    @staticmethod
    def int2token_dict(tokens: list[str]):
        return dict((i, c) for i, c in enumerate(tokens))

    def smiles2seq(
        self,
        tokenized_smiles_list: list[str],
        max_length: int,
        tokens: list[str]
    ):
        converter = self.token2int_dict(tokens)
        int_array = np.zeros(
            shape=(len(tokenized_smiles_list), max_length), dtype=np.int32
        )
        for idx, tokenized_smiles in enumerate(tokenized_smiles_list):
            smiles_tmp = list()
            pad_len = (max_length - len(tokenized_smiles))
            if pad_len >= 0:
                smiles_tmp = tokenized_smiles + [PADDING] * pad_len  # Force output vectors to have same length
            else:
                smiles_tmp = tokenized_smiles[:max_length]  # longer vectors are truncated (to be changed...)
            integer_encoded = [
                converter[itoken]
                if (itoken in tokens) else converter[UNKNOWN]
                for itoken in smiles_tmp
            ]
            int_array[idx] = integer_encoded
        return int_array

    def convert_to_int_tensor(
        self,
        tokenized_smiles_list: list[str],
        y: list[float],
        max_length: int,
        tokens: list[str]
    ):
        int_array = self.smiles2seq(tokenized_smiles_list, max_length, tokens)
        int_tensor = torch.from_numpy(np.array(int_array).astype(np.int32))
        y_tensor = torch.from_numpy(np.array(y, dtype=np.float32))
        return int_tensor, y_tensor


def check_unique_tokens(
    tokens_train: list[str],
    tokens_valid: list[str],
    tokens_test: list[str]
):
    train_unique_tokens = set(Token.extract_vocab(tokens_train))
    valid_unique_tokens = set(Token.extract_vocab(tokens_valid))
    test_unique_tokens = set(Token.extract_vocab(tokens_test))
    logging.info(f"Number of tokens only present in a training set: {len(train_unique_tokens)}")
    logging.info(f"Number of tokens only present in a validation set: {len(valid_unique_tokens)}")
    logging.info(f"Is the validation set a subset of the training set: {valid_unique_tokens.issubset(train_unique_tokens)}")
    logging.info(f"What are the tokens by which they differ: {valid_unique_tokens.difference(train_unique_tokens)}")
    logging.info(f"Number of tokens only present in a test set: {len(test_unique_tokens)}")
    logging.info(f"Is the test set a subset of the training set: {test_unique_tokens.issubset(train_unique_tokens)}")
    logging.info(f"What are the tokens by which they differ: {test_unique_tokens.difference(train_unique_tokens)}")
    logging.info(f"Is the test set a subset of the validation set: {test_unique_tokens.issubset(valid_unique_tokens)}")
    logging.info(f"What are the tokens by which they differ: {test_unique_tokens.difference(valid_unique_tokens)}")
