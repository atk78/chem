import os
import re
import logging

import yaml
import numpy as np
import torch


def get_vocab(save_dir, tokens):
    vocab = extract_vocab(tokens)
    vocab_file = os.path.join(save_dir, "Vocabulary.yaml")
    with open(vocab_file, "w") as f:
        yaml.dump(list(vocab), f)
    return vocab


def get_tokens(selfies_array):
    tokenized_selfies_list = list()
    for i_selfeies in selfies_array.tolist():
        tokenized_selfies_tmp = selfies_tokenizer(i_selfeies)
        tokenized_selfies_list.append(tokenized_selfies_tmp)
    return tokenized_selfies_list


def selfies_tokenizer(selfies):
    splitted_selfies = re.findall(pattern=r"\[[^\[\]]*]", string=selfies)
    return (
        "[ ]" + splitted_selfies + "[ ]"
    )  # add start + ... + end of SMILES


def get_tokentoint(tokens):
    return dict((c, i) for i, c in enumerate(tokens))


def get_inttotoken(tokens):
    return dict((i, c) for i, c in enumerate(tokens))


def extract_vocab(lltokens):
    return set([itoken for iselfies in lltokens for itoken in iselfies])


def add_extra_tokens(tokens, vocab_size):
    tokens.insert(0, "[unk]")
    tokens.insert(0, "[pad]")
    vocab_size = vocab_size + 2
    return tokens, vocab_size


def int_vec_encode(tokenized_selfies_list, max_length, tokens):
    token_to_int = get_tokentoint(tokens)
    int_selfies_array = np.zeros(
        shape=(len(tokenized_selfies_list), max_length), dtype=np.int32
    )
    for idx, iselfies in enumerate(tokenized_selfies_list):
        iselfies_tmp = list()
        if len(iselfies) <= max_length:
            iselfies_tmp = ["[pad]"] * (max_length - len(iselfies)) + iselfies  # Force output vectors to have same length
        else:
            iselfies_tmp = iselfies[-max_length:]  # longer vectors are truncated (to be changed...)
        integer_encoded = [
            token_to_int[itoken] if (itoken in tokens) else token_to_int["unk"]
            for itoken in iselfies_tmp
        ]
        int_selfies_array[idx] = integer_encoded
    return int_selfies_array


def convert_to_int_tensor(tokenized_selfies_list, y, max_length, tokens):
    int_vec_tokens = int_vec_encode(tokenized_selfies_list, max_length, tokens)
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
