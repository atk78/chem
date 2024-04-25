import logging

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from . import plot
from .augm import mean_std_result


np.set_printoptions(precision=3)


def model_evaluation(
    model: Module,
    img_dir: str,
    graphs_train: Data,
    graphs_valid: Data,
    graphs_test: Data,
    enum_card_train: list,
    enum_card_valid: list,
    enum_card_test: list,
    scaler=None,
    loss_func="MAE",
):
    y_train, y_pred_train, mae_train, rmse_train, r2_train = compute_metrics(
        model, graphs_train, enum_card_train, scaler
    )
    y_valid, y_pred_valid, mae_valid, rmse_valid, r2_valid = compute_metrics(
        model, graphs_valid, enum_card_valid, scaler
    )
    y_test, y_pred_test, mae_test, rmse_test, r2_test = compute_metrics(
        model, graphs_test, enum_card_test, scaler
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
    )


def compute_metrics(
    model: Module,
    enum_graph: Data,
    card: list,
    scaler
):
    model.eval()
    y_pred_list = []
    y_list = []
    for dataset in enum_graph:
        y = dataset.y.detach().numpy().copy()
        with torch.no_grad():
            y_pred = model.forward(dataset)
        y_pred = y_pred.detach().numpy().copy()
        if scaler is not None:
            y = scaler.inverse_transform(y).flatten()
            y_pred = scaler.inverse_transform(y_pred).flatten()
        y_pred_list.extend(y_pred)
        y_list.extend(y)
    card = np.array(card)
    y_pred, _ = mean_std_result(card, y_pred_list)
    y, _ = mean_std_result(card, y_list)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)
    return y, y_pred, mae, rmse, r2
