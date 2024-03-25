import os
import math
import random

import numpy as np
from sklearn.preprocessing import RobustScaler
from scipy.ndimage.interpolation import shift
import torch

np.set_printoptions(precision=3)


def random_split(smiles_input, prop_input, random_state, scaling=True):
    np.random.seed(seed=random_state)
    full_idx = np.array([x for x in range(smiles_input.shape[0])])
    train_idx = np.random.choice(
        full_idx, size=math.ceil(0.8 * smiles_input.shape[0]), replace=False
    )
    x_train = smiles_input[train_idx]
    y_train = prop_input[train_idx].reshape(-1, 1)

    valid_test_idx = full_idx[np.isin(full_idx, train_idx, invert=True)]
    valid_test_len = math.ceil(0.5 * valid_test_idx.shape[0])
    valid_idx = valid_test_idx[:valid_test_len]
    test_idx = valid_test_idx[valid_test_len:]
    x_valid = smiles_input[valid_idx]
    y_valid = prop_input[valid_idx].reshape(-1, 1)
    x_test = smiles_input[test_idx]
    y_test = prop_input[test_idx].reshape(-1, 1)

    if scaling == True:
        scaler = RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(5.0, 95.0),
            copy=True,
        )
        scaler_fit = scaler.fit(y_train)
        print("Scaler: {}".format(scaler_fit))
        y_train = scaler.transform(y_train)
        y_valid = scaler.transform(y_valid)
        y_test = scaler.transform(y_test)

    print(
        "Train/valid/test splits: {0:0.2f}/{1:0.2f}/{2:0.2f}\n\n".format(
            x_train.shape[0] / smiles_input.shape[0],
            x_valid.shape[0] / smiles_input.shape[0],
            x_test.shape[0] / smiles_input.shape[0],
        )
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test, scaler


def mean_median_result(x_cardinal_tmp, y_pred_tmp):
    x_card_cumsum = np.cumsum(x_cardinal_tmp)
    x_card_cumsum_shift = shift(x_card_cumsum, 1, cval=0)

    y_mean = np.array(
        [
            np.mean(y_pred_tmp[x_card_cumsum_shift[cenumcard] : ienumcard])
            for cenumcard, ienumcard in enumerate(x_card_cumsum.tolist())
        ]
    )

    y_med = np.array(
        [
            np.median(y_pred_tmp[x_card_cumsum_shift[cenumcard] : ienumcard])
            for cenumcard, ienumcard in enumerate(x_card_cumsum.tolist())
        ]
    )

    return y_mean, y_med


def seed_everything(seed=42):
    random.seed(seed)  # Python標準のrandomモジュールのシードを設定
    os.environ["PYTHONHASHSEED"] = str(seed)  # ハッシュ生成のためのシードを環境変数に設定
    np.random.seed(seed)  # NumPyの乱数生成器のシードを設定
    torch.manual_seed(seed)  # PyTorchの乱数生成器のシードをCPU用に設定
    torch.cuda.manual_seed(seed)  # PyTorchの乱数生成器のシードをGPU用に設定
    torch.backends.cudnn.deterministic = True  # PyTorchの畳み込み演算の再現性を確保
