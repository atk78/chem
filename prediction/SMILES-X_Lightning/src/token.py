import os
import yaml
import logging

import numpy as np
import torch


class Tokens:
    aliphatic_organic = ["B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    aromatic_organic = ["b", "c", "n", "o", "s", "p"]
    bracket = ["[", "]"]  # includes isotope, symbol, chiral, hcount, charge, class
    bond = ["-", "=", "#", "$", "/", "\\", "."]
    lrb = ["%"]  # long ring bonds '%TWODIGITS'
    terminator = [" "]  # SPACE - start/end of SMILES
    wildcard = ["*"]
    oov = ["oov"]  # out-of-vocabulary tokens


def get_vocab(save_dir, tokens):
    vocab = extract_vocab(tokens)
    vocab_file = os.path.join(save_dir, "Vocabulary.yaml")
    with open(vocab_file, "w") as f:
        yaml.dump(list(vocab), f)
    return vocab


def get_tokens(smiles_array, split_l=1, poly_flag=False):
    tokenized_smiles_list = list()
    for i_smiles in smiles_array.tolist():
        tokenized_smiles_tmp = smiles_tokenizer(i_smiles, poly_flag)
        tokenized_smiles_list.append(
            [
                "".join(tokenized_smiles_tmp[i: i + split_l])
                for i in range(0, len(tokenized_smiles_tmp) - split_l + 1, 1)
            ]
        )
    return tokenized_smiles_list


def smiles_tokenizer(smiles, poly_flag=False):
    if poly_flag:
        Tokens.aliphatic_organic += ["*"]
    smiles = smiles.replace("", "")  # avoid '' if exists in smiles
    # '[...]' as single token
    smiles = smiles.replace(Tokens.bracket[0], " " + Tokens.bracket[0])
    smiles = smiles.replace(Tokens.bracket[1], Tokens.bracket[1] + " ")
    # '%TWODIGITS' as single token
    lrb_print = [
        smiles[i: i + 3]
        for i, ichar in enumerate(smiles) if ichar == Tokens.lrb[0]
    ]
    if len(lrb_print) != 0:
        for ichar in lrb_print:
            smiles = smiles.replace(ichar, " " + ichar + " ")
    # split SMILES for [...] recognition
    smiles = smiles.split(" ")
    # split fragments other than [...]
    splitted_smiles = list()
    for ifrag in smiles:
        ifrag_tag = False
        for inac in Tokens.bracket + Tokens.lrb:
            if inac in ifrag:
                ifrag_tag = True
                break
        if ifrag_tag is False:
            # check for Cl, Br in alphatic branches to not dissociate letters (e.g. Cl -> C, l is prohibited)
            for iaa in Tokens.aliphatic_organic[7:9]:
                ifrag = ifrag.replace(iaa, " " + iaa + " ")
            ifrag_tmp = ifrag.split(" ")
            for iifrag_tmp in ifrag_tmp:
                if (
                    iifrag_tmp != Tokens.aliphatic_organic[7]
                    and iifrag_tmp != Tokens.aliphatic_organic[8]
                ):  # not 'Cl' and not 'Br'
                    splitted_smiles.extend(iifrag_tmp)  # automatic split char by char
                else:
                    splitted_smiles.extend([iifrag_tmp])
        else:
            splitted_smiles.extend([ifrag])  # keep the original token size
    return (
        Tokens.terminator + splitted_smiles + Tokens.terminator
    )  # add start + ... + end of SMILES


def get_tokentoint(tokens):
    return dict((c, i) for i, c in enumerate(tokens))


def get_inttotoken(tokens):
    return dict((i, c) for i, c in enumerate(tokens))


def extract_vocab(lltokens):
    return set([itoken for ismiles in lltokens for itoken in ismiles])


def add_extra_tokens(tokens, vocab_size):
    tokens.insert(0, "unk")
    tokens.insert(0, "pad")
    vocab_size = vocab_size + 2
    return tokens, vocab_size


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
