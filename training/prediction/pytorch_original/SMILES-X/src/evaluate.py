from logging import Logger
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

from src import plot, augm
from src.data import SmilesXData
from src.model import SmilesX


np.set_printoptions(precision=3)


def model_evaluation(
    model: SmilesX,
    logger: Logger,
    output_dir: Path,
    smilesX_data: SmilesXData,
    device="cpu",
):
    img_dir = output_dir.joinpath("images")
    img_dir.mkdir()
    model_dir = output_dir.joinpath("model")
    result = dict()
    for phase, dataset in smilesX_data.tensor_datasets.items():
        dataloader = DataLoader(
            dataset,
            batch_size=smilesX_data.batch_size,
            shuffle=False,
            drop_last=False
        )
        result[phase] = compute_metrics(
            model,
            model_dir,
            dataloader,
            smilesX_data.enum_cards[phase],
            smilesX_data.scaler,
            device
        )
        logger.info(f"For the {phase} set:")
        logger.info(
            f"MAE: {result[phase]['MAE']:.4f} RMSE: {result[phase]['RMSE']:.4f} R^2: {result[phase]['R2']:.4f}"
        )
    plot.plot_obserbations_vs_predictions(result, img_dir=img_dir)


def compute_metrics(
    model: SmilesX,
    model_dir: Path,
    dataloader: DataLoader,
    enum_cards: list[int],
    scaler: RobustScaler | None = None,
    device="cpu",
):
    pth_path_list = list(model_dir.glob("*.pth"))
    y_list, y_pred_list = [], []
    for X, y in dataloader:
        X = X.to(device)
        y = y.cpu().detach().numpy()
        for i, pth_path in enumerate(pth_path_list):
            model.load_state_dict(torch.load(pth_path))
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                y_pred_tmp = model.forward(X)
            y_pred_tmp = y_pred_tmp.cpu().detach().numpy()
            if i == 0:
                y_pred = y_pred_tmp.copy()
            else:
                y_pred = np.concatenate([y_pred, y_pred_tmp], axis=1)
        if len(pth_path_list) != 1:
            y_pred = y_pred.mean(axis=1).reshape(-1, 1)
        if scaler is not None:
            y = scaler.inverse_transform(y)
            y_pred = scaler.inverse_transform(y_pred)
        y_list.extend(list(y))
        y_pred_list.extend(list(y_pred))
    y_list, _ = augm.mean_std_result(enum_cards, y_list)
    y_pred_list, _ = augm.mean_std_result(enum_cards, y_pred_list)
    mae = float(mean_absolute_error(y_list, y_pred_list))
    rmse = float(root_mean_squared_error(y_list, y_pred_list))
    r2 = float(r2_score(y_list, y_pred_list))
    result = {
        "y": y_list,
        "y_pred": y_pred_list,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
    return result
