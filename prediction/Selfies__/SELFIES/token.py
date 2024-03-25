import os
import ast

import numpy as np
import torch

from . import augm


class Token:
    aliphatic_organic = ["B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    aromatic_organic = ["b", "c", "n", "o", "s", "p"]
    bracket = ["[", "]"]  # includes isotope, symbol, chiral, hcount, charge, class
    bond = ["-", "=", "#", "$", "/", "\\", "."]
    lrb = ["%"]  # long ring bonds '%TWODIGITS'
    terminator = [" "]  # SPACE - start/end of SMILES
    wildcard = ["*"]
    oov = ["oov"]  # out-of-vocabulary tokens

    def __init__(
        self,
        smiles_train,
        smiles_valid,
        smiles_test,
        prop_train,
        prop_valid,
        prop_test,
        augmentation=True,
        save_dir=".",
    ):
        self.augmentation = augmentation
        self.save_dir = save_dir
        if augmentation == True:
            print(f"***Data augmentation is {augmentation}***\n")
            self.canonical = False
            self.rotation = True
        else:
            print("***No data augmentation has been required.***\n")
            self.canonical = True
            self.rotation = False

        print("Enumerated Selfies:")
        enum_selfies_train, self.enum_card_train, enum_prop_train = augm.Augmentation(
            smiles_train, prop_train, canon=self.canonical, rotate=self.rotation
        )
        enum_tokens_train = self.get_tokens(enum_selfies_train)
        enum_selfies_valid, self.enum_card_valid, enum_prop_valid = augm.Augmentation(
            smiles_valid, prop_valid, canon=self.canonical, rotate=self.rotation
        )
        enum_tokens_valid = self.get_tokens(enum_selfies_valid)
        enum_selfies_test, self.enum_card_test, enum_prop_test = augm.Augmentation(
            smiles_test, prop_test, canon=self.canonical, rotate=self.rotation
        )
        enum_tokens_test = self.get_tokens(enum_selfies_test)

        all_selfies_tokens = enum_tokens_train + enum_tokens_valid + enum_tokens_test
        tokens = self.get_vocab(all_selfies_tokens)

        vocab_size = len(tokens)
        self.check_unique_tokens(enum_tokens_train, enum_tokens_valid, enum_tokens_test)
        print("Full vocabulary: {}\nOf size: {}\n".format(tokens, vocab_size))

        self.tokens, self.vocab_size = self.add_extra_tokens(tokens, vocab_size)
        max_length = np.max([len(ismiles) for ismiles in all_selfies_tokens])
        print(
            f"Maximum length of tokenized SMILES: {max_length} tokens (termination spaces included)\n"
        )
        # [unk]の分も足す
        self.max_length = max_length + 1
        self.enum_tokens_train, self.enum_prop_train = (
            self.convert_enum_tokens_to_torch_tensor(enum_tokens_train, enum_prop_train)
        )
        self.enum_tokens_valid, self.enum_prop_valid = (
            self.convert_enum_tokens_to_torch_tensor(enum_tokens_valid, enum_prop_valid)
        )
        self.enum_tokens_test, self.enum_prop_test = (
            self.convert_enum_tokens_to_torch_tensor(enum_tokens_test, enum_prop_test)
        )

    def get_vocab(self, tokens):
        vocab = self.extract_vocab(tokens)
        vocab_file = os.path.join(self.save_dir, "Vocabulary.txt")
        with open(vocab_file, "w") as f:
            f.write(str(list(vocab)))
        with open(vocab_file, "r") as f:
            tokens = ast.literal_eval(f.read())
        return tokens

    def get_tokens(self, selfies_array):
        tokenized_selfies_list = list()
        for i_selfies in selfies_array:
            tokenized_selfies_tmp = self.selfies_tokenizer(i_selfies)
            tokenized_selfies_list.append(tokenized_selfies_tmp)
        return tokenized_selfies_list

    def selfies_tokenizer(self, selfies):
        selfies = selfies.split(sep="][")
        selfies[0] = selfies[0][1:]
        selfies[-1] = selfies[-1][:-1]
        selfies.insert(0, " ")
        selfies.extend(" ")
        selfies = [f"[{s}]" for s in selfies]
        return selfies

    def get_tokentoint(self, tokens):
        return dict((c, i) for i, c in enumerate(tokens))

    def get_inttotoken(self, tokens):
        return dict((i, c) for i, c in enumerate(tokens))

    def extract_vocab(self, lltokens):
        return set([itoken for ismiles in lltokens for itoken in ismiles])

    def add_extra_tokens(self, tokens, vocab_size):
        tokens.insert(0, "[unk]")
        tokens.insert(0, "[pad]")
        vocab_size = vocab_size + 2
        return tokens, vocab_size

    def int_vec_encode(self, tokenized_selfies_list):
        vocab_int_dict = self.get_tokentoint(self.tokens)
        int_selfies_array = np.zeros(
            (len(tokenized_selfies_list), self.max_length), dtype=np.int32
        )
        for idx, i_selfies in enumerate(tokenized_selfies_list):
            i_selfies_tmp = list()
            if len(i_selfies) <= self.max_length:
                i_selfies_tmp = ["[pad]"] * (self.max_length - len(i_selfies)) + i_selfies  # Force output vectors to have same length
            else:
                i_selfies_tmp = i_selfies[-self.max_length:]  # longer vectors are truncated (to be changed...)
            integer_encoded = [vocab_int_dict[itoken] if (itoken in self.tokens) else vocab_int_dict["[unk]"] for itoken in i_selfies_tmp]
            int_selfies_array[idx] = integer_encoded
        return int_selfies_array

    def convert_enum_tokens_to_torch_tensor(self, enum_tokens, enum_prop):
        enum_tokens = self.int_vec_encode(enum_tokens)
        torch_enum_tokens = torch.IntTensor(enum_tokens)
        torch_enum_prop = torch.FloatTensor(enum_prop)
        return torch_enum_tokens, torch_enum_prop

    def check_unique_tokens(self, enum_tokens_train, enum_tokens_valid, enum_tokens_test):
        train_unique_tokens = self.extract_vocab(enum_tokens_train)
        valid_unique_tokens = self.extract_vocab(enum_tokens_valid)
        test_unique_tokens = self.extract_vocab(enum_tokens_test)
        print("Number of tokens only present in a training set:", end="")
        print(len(train_unique_tokens), end="\n\n")

        print("Number of tokens only present in a validation set", end="")
        print(len(valid_unique_tokens))
        print("Is the validation set a subset of the training set:", end="")
        print(valid_unique_tokens.issubset(train_unique_tokens))
        print("What are the tokens by which they differ:", end="")
        print(valid_unique_tokens.difference(train_unique_tokens), end="\n\n")

        print("Number of tokens only present in a test set:", end="")
        print(len(test_unique_tokens))
        print("Is the test set a subset of the training set:", end="")
        print(test_unique_tokens.issubset(train_unique_tokens))
        print("What are the tokens by which they differ:", end="")
        print(test_unique_tokens.difference(train_unique_tokens))
        print("Is the test set a subset of the validation set:", end="")
        print(test_unique_tokens.issubset(valid_unique_tokens))
        print("What are the tokens by which they differ:", end="")
        print(test_unique_tokens.difference(valid_unique_tokens), end="\n\n")