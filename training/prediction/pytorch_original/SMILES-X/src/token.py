import yaml
import logging
from pathlib import Path

import numpy as np
import torch


class Tokens:
    aliphatic_organic = ["B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    aromatic_organic = ["b", "c", "n", "o", "s", "p"]
    bracket = ["[", "]"]  # ex) [Na+], [nH]
    bond = ["-", "=", "#", "$", "/", ".", "*"]
    lrb = ["%"]
    terminator = [" "]
    wildcard = ["*"]
    oov = ["oov"]


def get_vocab(save_dir: Path, tokens):
    vocab = extract_vocab(tokens)
    vocab_file = save_dir.joinpath("Vocabulary.yaml")
    with open(vocab_file, "w") as f:
        yaml.dump(list(vocab), f)
    return vocab


def get_tokens(smiles_array: list[str], split_l=1, poly_flag=False):
    tokenized_smiles_list = list()
    for i_smiles in smiles_array:
        tokenized_smiles_tmp = smiles_tokenizer(i_smiles, poly_flag)
        tokenized_smiles_list.append(
            [
                "".join(tokenized_smiles_tmp[i: i + split_l])
                for i in range(0, len(tokenized_smiles_tmp) - split_l + 1, 1)
            ]
        )
    return tokenized_smiles_list


def smiles_tokenizer(smiles: str, poly_flag=False):
    if poly_flag:
        Tokens.aliphatic_organic += ["*"]
    smiles = smiles.replace("'", "")
    smiles = smiles.replace(Tokens.bracket[0], " " + Tokens.bracket[0])
    smiles = smiles.replace(Tokens.bracket[1], Tokens.bracket[1] + " ")
    lrb_print = [
        smiles[i: i+3]
        for i, chr in enumerate(smiles) if chr == Tokens.lrb[0]
    ]
    if len(lrb_print) != 0:
        for chr in lrb_print:
            smiles = smiles.replace(chr, " " + chr + " ")
    smiles = smiles.split(" ")
    splited_smiles = list()
    for ifrag in smiles:
        ifrag_tag = False
        for inac in Tokens.bracket + Tokens.lrb:
            if inac in ifrag:
                ifrag_tag = True
                break
        if ifrag_tag is False:
            for iaa in Tokens.aliphatic_organic[7:9]:
                ifrag = ifrag.replace(iaa, " " + iaa + " ")
            ifrag_tmp = ifrag.split(" ")
            for iifrag_tmp in ifrag_tmp:
                if (
                    iifrag_tmp != Tokens.aliphatic_organic[7]
                    and iifrag_tmp != Tokens.aliphatic_organic[8]
                ):
                    splited_smiles.extend(iifrag_tmp)
                else:
                    splited_smiles.extend([iifrag_tmp])
        else:
            splited_smiles.extend([ifrag])
    return (
        Tokens.terminator + splited_smiles + Tokens.terminator
    )


def get_token_to_int(tokens):
    return dict((c, i) for i, c in enumerate(tokens))


def get_int_to_token(tokens: list[str]):
    return dict((i, c) for i, c in enumerate(tokens))


def extract_vocab(lltokens):
    return list(set([i_token for i_smiles in lltokens for i_token in i_smiles]))


def add_extract_tokens(tokens: list[str], vocab_size: int):
    tokens.insert(0, "unk")
    tokens.insert(0, "pad")
    vocab_size = vocab_size + 2
    return tokens, vocab_size


def int_vec_encode(
    tokenized_smiles_list: list[str], max_length: int, tokens: list[str]
):
    token_to_int = get_token_to_int(tokens)
    int_smiles_array = np.zeros(
        shape=(len(tokenized_smiles_list), max_length), dtype=np.int32
    )
    for idx, i_smiles in enumerate(tokenized_smiles_list):
        i_smiles_tmp = list()
        if len(i_smiles) <= max_length:
            i_smiles_tmp = ["pad"] * (max_length - len(i_smiles)) + i_smiles
        else:
            i_smiles_tmp = i_smiles[-max_length:]
        int_encoded = [
            token_to_int[i_token] if (i_token in tokens) else token_to_int["unk"]
            for i_token in i_smiles_tmp
        ]
        int_smiles_array[idx] = int_encoded
    return int_smiles_array


def convert_to_int_tensor(tokenized_smiles_list: list[str], y, max_length: int, tokens):
    int_vec_tokens = int_vec_encode(tokenized_smiles_list, max_length, tokens)
    tokens_tensor = torch.from_numpy(np.array(int_vec_tokens).astype(np.int32))
    y_tensor = torch.from_numpy(np.array(y, dtype=np.float32))
    return tokens_tensor, y_tensor


def check_unique_tokens(
    tokens_train: list[str],
    tokens_valid: list[str],
    tokens_test: list[str]
):
    train_unique_tokens = extract_vocab(tokens_train)
    valid_unique_tokens = extract_vocab(tokens_valid)
    test_unique_tokens = extract_vocab(tokens_test)
    logging.info(f"Number of tokens only present in a training set: {len(train_unique_tokens)}")
    logging.info(f"Number of tokens only present in a validation set: {len(valid_unique_tokens)}")
    logging.info(f"Is the validation set a subset of the training set: {valid_unique_tokens.issubset(train_unique_tokens)}")
    logging.info(f"What are the tokens by which they differ: {valid_unique_tokens.difference(train_unique_tokens)}")
    logging.info(f"Number of tokens only present in a test set: {len(test_unique_tokens)}")
    logging.info(f"Is the test set a subset of the training set: {test_unique_tokens.issubset(train_unique_tokens)}")
    logging.info(f"What are the tokens by which they differ: {test_unique_tokens.difference(train_unique_tokens)}")
    logging.info(f"Is the test set a subset of the validation set: {test_unique_tokens.issubset(valid_unique_tokens)}")
    logging.info(f"What are the tokens by which they differ: {test_unique_tokens.difference(valid_unique_tokens)}")
