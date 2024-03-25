import logging
import datetime
import sys

import numpy as np
from scipy.ndimage.interpolation import shift
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler

from . import utils, plot
from .dataset import Data

np.set_printoptions(precision=3)


def random_split(
    smiles_input,
    y_input,
    train_ratio=0.8,
    validation_ratio=0.1,
    test_ratio=0.1,
    random_state=42,
    scaling=True
):
    scaler = None
    y_input = y_input.reshape(-1, 1)
    if train_ratio + validation_ratio + test_ratio != 1:
        raise RuntimeError("Make sure the sum of the ratios is 1.")
    logging.info(f"Train/valid/test splits:{train_ratio:.2f}/{validation_ratio:.2f}/{test_ratio:.2f}")
    x_train, x_test, y_train, y_test = train_test_split(
        smiles_input,
        y_input,
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
    if scaling:
        scaler = RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(5.0, 95.0),
            copy=True
        )
        y_train = scaler.fit_transform(y_train)
        y_valid = scaler.transform(y_valid)
        y_test = scaler.transform(y_test)
    return x_train, x_valid, x_test, y_train, y_valid, y_test, scaler


def mean_std_result(x_cardinal_tmp, y_pred_tmp):
    x_card_cumsum = np.cumsum(x_cardinal_tmp)
    x_card_cumsum_shift = shift(x_card_cumsum, 1, cval=0)
    y_mean = np.array(
        [
            np.mean(y_pred_tmp[x_card_cumsum_shift[cenumcard]: ienumcard])
            for cenumcard, ienumcard in enumerate(x_card_cumsum.tolist())
        ]
    )
    y_std = np.array(
        [
            np.std(y_pred_tmp[x_card_cumsum_shift[cenumcard]: ienumcard])
            for cenumcard, ienumcard in enumerate(x_card_cumsum.tolist())
        ]
    )
    return y_mean, y_std


def model_evaluation(
    img_dir: str,
    data_name: str,
    model,
    x_train: torch.Tensor,
    x_valid: torch.Tensor,
    x_test: torch.Tensor,
    y_train: torch.Tensor,
    y_valid: torch.Tensor,
    y_test: torch.Tensor,
    enum_card_train: np.ndarray,
    enum_card_valid: np.ndarray,
    enum_card_test: np.ndarray,
    scaler=None,
    loss_func="MAE",
):
    y_train, y_pred_train, mae_train, rmse_train, r2_train = compute_metrics(
        model, x_train, y_train, enum_card_train, scaler
    )
    y_valid, y_pred_valid, mae_valid, rmse_valid, r2_valid = compute_metrics(
        model, x_valid, y_valid, enum_card_valid, scaler
    )
    y_test, y_pred_test, mae_test, rmse_test, r2_test = compute_metrics(
        model, x_test, y_test, enum_card_test, scaler
    )
    logging.info("For the training set:")
    logging.info(
        f"MAE: {mae_train:.4f} RMSE: {rmse_train:.4f} R^2: {r2_train:.4f}"
    )
    logging.info("For the validation set:")
    logging.info(
        f"MAE: {mae_valid:.4f} RMSE: {rmse_valid:.4f} R^2: {r2_valid:.4f}"
    )
    logging.info("For the test set:")
    logging.info(
        f"MAE: {mae_test:.4f} RMSE: {rmse_test:.4f} R^2: {r2_test:.4f}"
    )

    if loss_func == "MAE":
        loss_train, loss_valid, loss_test = mae_train, mae_valid, mae_test
    else:
        loss_train, loss_valid, loss_test = rmse_train, rmse_valid, rmse_test

    plot.plot_obserbations_vs_predictions(
        observations=(y_train, y_valid, y_test),
        predictions=(y_pred_train, y_pred_valid, y_pred_test),
        loss=(loss_train, loss_valid, loss_test),
        r2=(r2_train, r2_valid, r2_test),
        loss_func=loss_func,
        img_dir=img_dir,
        data_name=data_name,
    )


def compute_metrics(model, enum_smiles, enum_y, card, scaler):
    model.eval()
    data = Data(enum_smiles, enum_y)
    if len(enum_y) < 10000:
        batch_size = len(enum_y)
    else:
        batch_size = 10000
    dataloader = DataLoader(
        data, batch_size=batch_size, shuffle=False, drop_last=False
    )
    y_pred_list = []
    y_list = []
    for dataset in dataloader:
        x, y = dataset[0], dataset[1].detach().numpy().copy().flatten()
        with torch.no_grad():
            y_pred = model.forward(x)
        y_pred = y_pred.detach().numpy().copy().flatten()
        if scaler is not None:
            y = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_pred_list.extend(y_pred)
        y_list.extend(y)
    card = np.array(card)
    y_pred, _ = utils.mean_std_result(card, y_pred_list)
    y, _ = utils.mean_std_result(card, y_list)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)
    return y, y_pred, mae, rmse, r2


def log_setup(save_dir, name, verbose):
    """Setting up the logging format and files.

    Parameters
    ----------
    save_dir: str
        The directory where the logfile will be saved.
    name: str
        The name of the operation (train, inference, interpretation).
    verbose: bool
        Whether of now to printout the logs into console.

    Returns
    -------
    logger: logger
        Logger instance.
    logfile: str
        File to save the logs to.
    """

    # Setting up logging
    current_datetime = datetime.datetime.now()
    str_datetime = current_datetime.strftime("%Y%m%d%H%M%S")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_format = "%(asctime)s | %(levelname)s |    %(message)s"
    formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Remove existing handlers if any
    logger.handlers.clear()

    # Logging to the file
    logfile = f"{save_dir}/{name}_{str_datetime}.log"
    file_handler = logging.FileHandler(filename=logfile, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Logging to console
    if verbose:
        handler_stdout = logging.StreamHandler(sys.stdout)
        handler_stdout.setLevel(logging.INFO)
        handler_stdout.setFormatter(formatter)
        logger.addHandler(handler_stdout)

    return logger, logfile
