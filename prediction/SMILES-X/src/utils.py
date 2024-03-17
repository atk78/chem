import numpy as np
from scipy.ndimage.interpolation import shift
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from . import utils
from .dataset import Data

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


def evaluation_model(model, enum_smiles, enum_prop, card):
    model.eval()
    data = Data(enum_smiles, enum_prop)
    if len(enum_prop) < 10000:
        batch_size = len(enum_prop)
    else:
        batch_size = 10000
    dataloader = DataLoader(
        data, batch_size=batch_size, shuffle=False, drop_last=False
    )
    y_pred_list = []
    y_list = []
    for dataset in dataloader:
        x, y = dataset[0], list(dataset[1].detach().numpy().copy().flatten())
        with torch.no_grad():
            y_pred = model.forward(x)
        y_pred = list(y_pred.detach().numpy().copy().flatten())
        y_pred_list.extend(y_pred)
        y_list.extend(y)
    card = np.array(card)
    y_pred, _ = utils.mean_median_result(card, y_pred_list)
    y, _ = utils.mean_median_result(card, y_list)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)
    return y, y_pred, mae, rmse, r2


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
