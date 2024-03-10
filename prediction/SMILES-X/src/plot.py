import os
import shutil

import matplotlib.pyplot as plt
import numpy as np


def plot_hitory_loss(loss, r2, loss_func, save_dir=".", data_name=""):
    if os.path.exists(os.path.join(save_dir, "images")):
        shutil.rmtree(os.path.join(save_dir, "images"))
        os.makedirs(os.path.join(save_dir, "images"))
    else:
        os.makedirs(os.path.join(save_dir, "images"))

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    ax1.plot(loss, label=loss_func)
    ax2.plot(r2, label="$R^2$", color="orange")
    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()
    handlers = handler1 + handler2
    labels = label1 + label2
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(loss_func)
    ax2.set_ylabel("$R^2$")
    ax2.set_ylim([0, 1])
    ax1.legend(handlers, labels, loc="upper right", bbox_to_anchor=(1, 1))
    fig.savefig(
        os.path.join(save_dir, f"images/History_{data_name}.png"),
        bbox_inches="tight"
    )


def plot_obserbations_vs_predictions(
    observations, predictions, loss, r2, loss_func, save_dir=".", data_name=""
):
    y_train, y_valid, y_test = observations
    y_pred_train, y_pred_valid, y_pred_test = predictions
    loss_train, loss_valid, loss_test = loss
    r2_train, r2_valid, r2_test = r2

    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.size"] = 14
    axis_min = min(np.min(y_train), np.min(y_valid), np.min(y_test),
                   np.min(y_pred_train), np.min(y_pred_valid), np.min(y_pred_test))
    axis_max = max(np.max(y_train), np.max(y_valid), np.max(y_test),
                   np.max(y_pred_train), np.max(y_pred_valid), np.max(y_pred_test))
    axis_min -= 0.1 * (axis_max - axis_min)
    axis_max += 0.1 * (axis_max - axis_min)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()
    ax.scatter(
        y_train,
        y_pred_train,
        color="royalblue",
        edgecolors="black",
        alpha=0.7,
        label="train"
    )
    ax.scatter(
        y_valid,
        y_pred_valid,
        color="green",
        edgecolors="black",
        alpha=0.7,
        label="valid"
    )
    ax.scatter(
        y_test,
        y_pred_test,
        color="orange",
        edgecolors="black",
        alpha=0.7,
        label="test"
    )
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color="black")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Predictions")
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
    ax.text(0.05, 0.95, f"Train {loss_func}: {loss_train:.4f}", transform=ax.transAxes, fontsize=10)
    ax.text(0.05, 0.91, f"Train $R^2$: {r2_train:.4f}", transform=ax.transAxes, fontsize=10)
    ax.text(0.05, 0.87, f"Valid {loss_func}: {loss_valid:.4f}", transform=ax.transAxes, fontsize=10)
    ax.text(0.05, 0.83, f"Valid $R^2$: {r2_valid:.4f}", transform=ax.transAxes, fontsize=10)
    ax.text(0.05, 0.79, f"Test {loss_func}: {loss_test:.4f}", transform=ax.transAxes, fontsize=10)
    ax.text(0.05, 0.75, f"Test $R^2$: {r2_test:.4f}", transform=ax.transAxes, fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.savefig(os.path.join(save_dir, f"images/Plot_{data_name}.png"))
