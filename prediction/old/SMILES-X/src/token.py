import os
import re
import yaml
import logging
import itertools

from tqdm import tqdm
import numpy as np
import torch


class Token:
    atom_len1 = ["B", "C", "N", "O", "S", "P", "F", "I"]
    atom_len2 = ["Cl", "Br", "Si"]
    atom_aroma = ["b", "c", "n", "o", "s", "p"]
    bracket = [r"\[[^\[\]]*\]"]
    bond = ["-", "=", "#", "$", "/", ".", "*"]
    ring = [str(i) for i in range(1, 10)]
    long_ring = [r"%\d{2}"]
    teminator = [" "]
    unknown = ["unk"]
    padding = ["pad"]
    all_atoms = atom_len1 + atom_len2
    pattern_len2 = atom_len2 + bracket + long_ring
    pattern_len1 = atom_len1 + atom_aroma + bond + ring


def smiles_tokenizer(smiles, pattern_len2):
    smiles = pattern_len2.sub(r",\1,", string=smiles)
    smiles = smiles.split(",")
    splitted_smiles = list()
    for chr in smiles:
        if pattern_len2.match(chr):
            splitted_smiles.append(chr)
            continue
        else:
            for c in chr:
                splitted_smiles.append(c)
    return Token.teminator + splitted_smiles + Token.teminator


def get_tokens(smiles_array):
    tokenized_smiles_list = list()
    pattern_len2 = re.compile(f'({"|".join(Token.pattern_len2)})')
    for i_smiles in tqdm(smiles_array):
        tokenized_smiles_tmp = smiles_tokenizer(i_smiles, pattern_len2)
        tokenized_smiles_list.append(tokenized_smiles_tmp)
    return tokenized_smiles_list


def extract_vocab(tokenized_smiles_list):
    return set(itertools.chain.from_iterable(tokenized_smiles_list))


def add_extra_tokens(tokens, vocab_size):
    tokens.insert(0, "unk")
    tokens.insert(0, "pad")
    vocab_size = vocab_size + 2
    return tokens, vocab_size


def get_vocab(save_dir, tokens):
    vocab = extract_vocab(tokens)
    vocab_file = os.path.join(save_dir, "Vocabulary.yaml")
    with open(vocab_file, "w") as f:
        yaml.dump(list(vocab), f)
    return vocab


def get_tokentoint(tokens):
    return dict((c, i) for i, c in enumerate(tokens))


def get_inttotoken(tokens):
    return dict((i, c) for i, c in enumerate(tokens))


def int_vec_encode(tokenized_smiles_list, max_length, tokens):
    token_to_int = get_tokentoint(tokens)
    int_smiles_array = np.zeros(
        shape=(len(tokenized_smiles_list), max_length), dtype=np.int32
    )
    for idx, ismiles in enumerate(tokenized_smiles_list):
        ismiles_tmp = list()
        if len(ismiles) <= max_length:
            ismiles_tmp = ["pad"] * (max_length - len(ismiles)) + ismiles  # Force output vectors to have same length
        else:
            ismiles_tmp = ismiles[-max_length:]  # longer vectors are truncated (to be changed...)
        integer_encoded = [
            token_to_int[itoken] if (itoken in tokens) else token_to_int["unk"]
            for itoken in ismiles_tmp
        ]
        int_smiles_array[idx] = integer_encoded
    return int_smiles_array


def convert_to_int_tensor(tokenized_smiles_list, y, max_length, tokens):
    int_vec_tokens = int_vec_encode(tokenized_smiles_list, max_length, tokens)
    tokens_tensor = torch.from_numpy(np.array(int_vec_tokens).astype(np.int32))
    y_tensor = torch.from_numpy(np.array(y, dtype=np.float32))
    return tokens_tensor, y_tensor


def check_unique_tokens(tokens_train, tokens_valid, tokens_test):
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
