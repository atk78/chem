from logging import Logger
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import RobustScaler
from torch_geometric.loader import DataLoader

from src import plot
from src.data import GraphData
from src.model import MolecularGNN


np.set_printoptions(precision=3)


def model_evaluation(
    model: MolecularGNN,
    logger: Logger,
    output_dir: Path,
    graph_data: GraphData,
    device="cpu",
):
    img_dir = output_dir.joinpath("images")
    img_dir.mkdir(exist_ok=True)
    model_dir = output_dir.joinpath("model")
    result = dict()
    for phase, graph_dataset in graph_data.graph_datasets.items():
        dataloader = DataLoader(
            graph_dataset,
            batch_size=graph_data.batch_size,
            shuffle=False,
            drop_last=False
        )
        result[phase] = compute_metrics(
            model, model_dir, dataloader, graph_data.scaler, device
        )
        logger.info(f"For the {phase} set:")
        logger.info(
            f"MAE: {result[phase]['MAE']:.4f} RMSE: {result[phase]['RMSE']:.4f} R^2: {result[phase]['R2']:.4f}"
        )

    plot.plot_obserbations_vs_predictions(result, img_dir=img_dir)


def compute_metrics(
    model: MolecularGNN,
    output_dir: Path,
    dataloader: DataLoader,
    scaler: RobustScaler | None = None,
    device="cpu",
):
    pth_path_list = list(output_dir.glob("*.pth"))
    y_list, y_pred_list = [], []

    for dataset in dataloader:
        dataset = dataset.to(device)
        y = dataset.y.cpu().detach().numpy()
        y_pred = np.zeros_like(y)
        for i, pth_path in enumerate(pth_path_list):
            model.load_state_dict(torch.load(pth_path))
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                y_pred_tmp = model.forward(dataset)
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


# def compute_metrics(
#     model: MolecularGNN,
#     graph,
#     scaler: RobustScaler | None = None,
#     device="cpu",
# ):
#     model.eval()
#     y_pred_list = []
#     y_list = []
#     for dataset in graph:
#         dataset = dataset.to(device)
#         y = dataset.y.cpu().detach().numpy().copy()
#         with torch.no_grad():
#             y_pred = model.forward(dataset)
#         y_pred = y_pred.cpu().detach().numpy().copy()
#         if scaler is not None:
#             y = scaler.inverse_transform(y)
#             y_pred = scaler.inverse_transform(y_pred)
#         # if y.shape[1] > 1:
#         #     y = ((y**2).sum(axis=1))**0.5
#         #     y_pred = ((y_pred**2).sum(axis=1))**0.5
#         y_list.extend(list(y.flatten()))
#         y_pred_list.extend(list(y_pred.flatten()))
#     mae = float(mean_absolute_error(y, y_pred))
#     rmse = float(root_mean_squared_error(y, y_pred))
#     r2 = float(r2_score(y, y_pred))
#     result = {
#         "y": y,
#         "y_pred": y_pred,
#         "MAE": mae,
#         "RMSE": rmse,
#         "R2": r2
#     }
#     return result
