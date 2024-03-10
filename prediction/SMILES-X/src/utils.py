import os
import random

import numpy as np
from scipy.ndimage.interpolation import shift
from sklearn.model_selection import train_test_split
import torch

np.set_printoptions(precision=3)


def random_split(smiles_input, prop_input, lengths=[0.8, 0.1, 0.1], random_state=42):
    prop_input = prop_input.reshape(-1, 1)
    train_ratio = lengths[0]
    validation_ratio = lengths[1]
    test_ratio = lengths[2]
    if train_ratio + validation_ratio + test_ratio != 1:
        print("Make sure the sum of the ratios is 1.")
        return
    print(f"Train/valid/test splits:{train_ratio:.2f}/{validation_ratio:.2f}/{test_ratio:.2f}\n\n")
    x_train, x_test, y_train, y_test = train_test_split(
        smiles_input,
        prop_input,
        test_size=1 - train_ratio,
        shuffle=True,
        random_state=random_state,
    )
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_test,
        y_test,
        test_size=test_ratio / (test_ratio + validation_ratio),
        shuffle=True,
        random_state=random_state,
    )
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def mean_median_result(x_cardinal_tmp, y_pred_tmp):
    x_card_cumsum = np.cumsum(x_cardinal_tmp)
    x_card_cumsum_shift = shift(x_card_cumsum, 1, cval=0)
    y_mean = np.array(
        [
            np.mean(y_pred_tmp[x_card_cumsum_shift[cenumcard] : ienumcard])
            for cenumcard, ienumcard in enumerate(x_card_cumsum.tolist())
        ]
    )
    y_std = np.array(
        [
            np.std(y_pred_tmp[x_card_cumsum_shift[cenumcard] : ienumcard])
            for cenumcard, ienumcard in enumerate(x_card_cumsum.tolist())
        ]
    )
    return y_mean, y_std


def seed_everything(seed=1):
    random.seed(seed)  # Python標準のrandomモジュールのシードを設定
    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # ハッシュ生成のためのシードを環境変数に設定
    np.random.seed(seed)  # NumPyの乱数生成器のシードを設定
    torch.manual_seed(seed)  # PyTorchの乱数生成器のシードをCPU用に設定
    torch.cuda.manual_seed(seed)  # PyTorchの乱数生成器のシードをGPU用に設定
    torch.backends.cudnn.deterministic = True  # PyTorchの畳み込み演算の再現性を確保


# def random_split(smiles_input, prop_input, random_state):
#     np.random.seed(seed=random_state)
#     full_idx = np.array([x for x in range(smiles_input.shape[0])])
#     train_idx = np.random.choice(
#         full_idx, size=math.ceil(0.8 * smiles_input.shape[0]), replace=False
#     )
#     x_train = smiles_input[train_idx]
#     y_train = prop_input[train_idx].reshape(-1, 1)

#     valid_test_idx = full_idx[np.isin(full_idx, train_idx, invert=True)]
#     valid_test_len = math.ceil(0.5 * valid_test_idx.shape[0])
#     valid_idx = valid_test_idx[:valid_test_len]
#     test_idx = valid_test_idx[valid_test_len:]

#     x_valid = smiles_input[valid_idx]
#     y_valid = prop_input[valid_idx].reshape(-1, 1)
#     x_test = smiles_input[test_idx]
#     y_test = prop_input[test_idx].reshape(-1, 1)

#     print(
#         "Train/valid/test splits: {0:0.2f}/{1:0.2f}/{2:0.2f}\n\n".format(
#             x_train.shape[0] / smiles_input.shape[0],
#             x_valid.shape[0] / smiles_input.shape[0],
#             x_test.shape[0] / smiles_input.shape[0],
#         )
#     )

#     return x_train, x_valid, x_test, y_train, y_valid, y_test
