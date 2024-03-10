import os
import ast

import numpy as np
import torch

from .augm import Augmentation


class Token:
    aliphatic_organic = ["B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    aromatic_organic = ["b", "c", "n", "o", "s", "p"]
    bracket = ["[", "]"]  # includes isotope, symbol, chiral, hcount, charge, class
    bond = ["-", "=", "#", "$", "/", "\\", "."]
    lrb = ["%"]  # long ring bonds '%TWODIGITS'
    terminator = [" "]  # SPACE - start/end of SMILES
    wildcard = ["*"]
    oov = ["oov"]  # out-of-vocabulary tokens

    def __init__(self, augmentation=True, poly=False):
        if poly:
            print("Setup Polymer Tokens.")
            self.aliphatic_organic += ["*"]
        else:
            print("Setup Molecule Tokens.")

        self.augmentation = augmentation
        if augmentation == True:
            print(f"***Data augmentation is {augmentation}***\n")
            self.canonical = False
            self.rotation = True
        else:
            print("***No data augmentation has been required.***\n")
            self.canonical = True
            self.rotation = False

    def get_vocab(self, tokens):
        vocab = self.extract_vocab(tokens)
        vocab_file = os.path.join(self.save_dir, "Vocabulary.txt")
        with open(vocab_file, "w") as f:
            f.write(str(list(vocab)))
        with open(vocab_file, "r") as f:
            tokens = ast.literal_eval(f.read())
        return tokens

    def get_tokens(self, smiles_array, split_l=1):
        tokenized_smiles_list = list()
        for ismiles in smiles_array.tolist():
            tokenized_smiles_tmp = self.smiles_tokenizer(ismiles)
            tokenized_smiles_list.append(
                [
                    "".join(tokenized_smiles_tmp[i : i + split_l])
                    for i in range(0, len(tokenized_smiles_tmp) - split_l + 1, 1)
                ]
            )
        return tokenized_smiles_list

    def smiles_tokenizer(self, smiles):
        smiles = smiles.replace("\n", "")  # avoid '\n' if exists in smiles
        # '[...]' as single token
        smiles = smiles.replace(self.bracket[0], " " + self.bracket[0]).replace(
            self.bracket[1], self.bracket[1] + " "
        )
        # '%TWODIGITS' as single token
        lrb_print = [
            smiles[ic : ic + 3]
            for ic, ichar in enumerate(smiles)
            if ichar == self.lrb[0]
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
            for inac in self.bracket + self.lrb:
                if inac in ifrag:
                    ifrag_tag = True
                    break
            if ifrag_tag == False:
                # check for Cl, Br in alphatic branches to not dissociate letters (e.g. Cl -> C, l is prohibited)
                for iaa in self.aliphatic_organic[7:9]:
                    ifrag = ifrag.replace(iaa, " " + iaa + " ")
                ifrag_tmp = ifrag.split(" ")
                for iifrag_tmp in ifrag_tmp:
                    if (
                        iifrag_tmp != self.aliphatic_organic[7]
                        and iifrag_tmp != self.aliphatic_organic[8]
                    ):  # not 'Cl' and not 'Br'
                        splitted_smiles.extend(
                            iifrag_tmp
                        )  # automatic split char by char
                    else:
                        splitted_smiles.extend([iifrag_tmp])
            else:
                splitted_smiles.extend([ifrag])  # keep the original token size
        return (
            self.terminator + splitted_smiles + self.terminator
        )  # add start + ... + end of SMILES

    def get_tokentoint(self, tokens):
        return dict((c, i) for i, c in enumerate(tokens))

    def get_inttotoken(self, tokens):
        return dict((i, c) for i, c in enumerate(tokens))

    def extract_vocab(self, lltokens):
        return set([itoken for ismiles in lltokens for itoken in ismiles])

    def add_extra_tokens(self, tokens, vocab_size):
        tokens.insert(0, "unk")
        tokens.insert(0, "pad")
        vocab_size = vocab_size + 2
        return tokens, vocab_size

    def int_vec_encode(self, tokenized_smiles_list):
        token_to_int = self.get_tokentoint(self.tokens)
        int_smiles_array = np.zeros(
            (len(tokenized_smiles_list), self.max_length), dtype=np.int32
        )
        for idx, ismiles in enumerate(tokenized_smiles_list):
            ismiles_tmp = list()
            if len(ismiles) <= self.max_length:
                ismiles_tmp = ["pad"] * (
                    self.max_length - len(ismiles)
                ) + ismiles  # Force output vectors to have same length
            else:
                ismiles_tmp = ismiles[
                    -self.max_length :
                ]  # longer vectors are truncated (to be changed...)
            integer_encoded = [
                token_to_int[itoken] if (itoken in self.tokens) else token_to_int["unk"]
                for itoken in ismiles_tmp
            ]
            int_smiles_array[idx] = integer_encoded
        return int_smiles_array

    def convert_enum_tokens_to_torch_tensor(self, enum_tokens, enum_prop):
        enum_tokens = self.int_vec_encode(enum_tokens)
        # torch_enum_tokens = torch.IntTensor(enum_tokens)
        # torch_enum_prop = torch.FloatTensor(enum_prop)
        torch_enum_tokens = torch.from_numpy(np.array(enum_tokens, dtype=np.int32))
        torch_enum_prop = torch.from_numpy(np.array(enum_prop, dtype=np.float32))
        return torch_enum_tokens, torch_enum_prop


class TrainToken(Token):
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
        poly=False,
    ):
        super().__init__(augmentation=augmentation, poly=poly)
        self.save_dir = save_dir
        self.smiles_train = smiles_train
        self.smiles_valid = smiles_valid
        self.smiles_test = smiles_test
        self.prop_train = prop_train
        self.prop_valid = prop_valid
        self.prop_test = prop_test

    def setup(self):
        print("Enumerated SMILES:")
        enum_smiles_train, self.enum_card_train, enum_prop_train = Augmentation(
            self.smiles_train,
            self.prop_train,
            canon=self.canonical,
            rotate=self.rotation,
        )
        enum_tokens_train = self.get_tokens(enum_smiles_train)
        enum_smiles_valid, self.enum_card_valid, enum_prop_valid = Augmentation(
            self.smiles_valid,
            self.prop_valid,
            canon=self.canonical,
            rotate=self.rotation,
        )
        enum_tokens_valid = self.get_tokens(enum_smiles_valid)
        enum_smiles_test, self.enum_card_test, enum_prop_test = Augmentation(
            self.smiles_test, self.prop_test, canon=self.canonical, rotate=self.rotation
        )
        enum_tokens_test = self.get_tokens(enum_smiles_test)

        all_smiles_tokens = enum_tokens_train + enum_tokens_valid + enum_tokens_test
        tokens = self.get_vocab(all_smiles_tokens)
        vocab_size = len(tokens)
        self.check_unique_tokens(enum_tokens_train, enum_tokens_valid, enum_tokens_test)
        print("Full vocabulary: {}\nOf size: {}\n".format(tokens, vocab_size))

        self.tokens, self.vocab_size = self.add_extra_tokens(tokens, vocab_size)
        max_length = np.max([len(ismiles) for ismiles in all_smiles_tokens])
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

    def check_unique_tokens(
        self, enum_tokens_train, enum_tokens_valid, enum_tokens_test
    ):
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


class InferacneToken(Token):
    def __init__(
        self, smiles_list, max_length, augmentation=True, input_dir=".", poly=False
    ):
        super().__init__(augmentation=augmentation, poly=poly)
        self.input_dir = input_dir
        self.max_length = max_length
        self.smiles = np.array(smiles_list)
        self.prop = np.array([[np.nan] * len(smiles_list)]).flatten()

    def setup(self):
        print("Enumerated SMILES:")
        enum_smiles, self.enum_card, enum_prop = Augmentation(
            self.smiles, self.prop, canon=self.canonical, rotate=self.rotation
        )
        enum_tokens = self.get_tokens(enum_smiles)
        with open(os.path.join(self.input_dir, "Vocabulary.txt"), mode="r") as f:
            tokens = f.read()[1:-1].replace("'", "").split(", ")

        vocab_size = len(tokens)
        self.tokens, self.vocab_size = self.add_extra_tokens(tokens, vocab_size)
        self.enum_tokens, self.enum_prop = self.convert_enum_tokens_to_torch_tensor(
            enum_tokens, enum_prop
        )
