import os

import matplotlib.pyplot as plt
import numpy as np


config = {
    "font.family": "Arial",
    "font.size": 12,
    "xtick.direction": "in",      # x軸の目盛りの向き
    "ytick.direction": "in",      # y軸の目盛りの向き
    "xtick.minor.visible": True,  # x軸補助目盛りの追加
    "ytick.minor.visible": True,  # y軸補助目盛りの追加
    "xtick.top": True,            # x軸上部の目盛り
    "ytick.right": True,          # y軸左部の目盛り
    "legend.fancybox": False,     # 凡例の角
    "legend.framealpha": 1,       # 枠の色の塗りつぶし
    "legend.edgecolor": "black"   # 枠の色
}


def plot_history_loss(train_loss, train_r2, valid_loss, valid_r2, loss_func, save_dir="", data_name=""):
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    plt.rcParams.update(config)
    fig = plt.figure(figsize=(6, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax1.plot(train_loss, color="black", label="train")
    ax1.plot(valid_loss, color="red", linestyle="--", label="valid")
    ax1.set_ylabel(loss_func)
    ax1.legend()
    ax2.plot(train_r2, color="black", label="train")
    ax2.plot(valid_r2, color="red", linestyle="--", label="valid")
    ax2.set_ylabel("$R^2$")
    ax2.set_xlabel("Epochs")
    ax2.legend()
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
    plt.rcParams.update(config)
    axis_min = min(np.min(y_train), np.min(y_valid), np.min(y_test),
                   np.min(y_pred_train), np.min(y_pred_valid), np.min(y_pred_test))
    axis_max = max(np.max(y_train), np.max(y_valid), np.max(y_test),
                   np.max(y_pred_train), np.max(y_pred_valid), np.max(y_pred_test))
    axis_min -= 0.1 * (axis_max - axis_min)
    axis_max += 0.1 * (axis_max - axis_min)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    for ax, y, y_pred, loss, r2, phase, color in zip(
        [ax1, ax2, ax3],
        [y_train, y_valid, y_test],
        [y_pred_train, y_pred_valid, y_pred_test],
        [loss_train, loss_valid, loss_test],
        [r2_train, r2_valid, r2_test],
        ["Training", "Validation", "Test"],
        ["royalblue", "green", "orange"]
    ):
        ax.scatter(y, y_pred, color=color, edgecolor="black", alpha=0.7, zorder=0)
        ax.plot([axis_min, axis_max], [axis_min, axis_max], color="black", zorder=1)
        ax.set_xlim(left=axis_min, right=axis_max)
        ax.set_ylim(bottom=axis_min, top=axis_max)
        ax.set_title(f"{phase}: $R^2$ = {r2:.2f}, {loss_func} = {loss:.2f}")
        ax.set_xlabel("Observations")
        ax.set_ylabel("Predictions")
    ax4.scatter(y_train, y_pred_train, color="royalblue", edgecolor="black", alpha=0.7, zorder=0, label="Training")
    ax4.scatter(y_valid, y_pred_valid, color="green", edgecolor="black", alpha=0.7, zorder=0, label="Validation")
    ax4.scatter(y_test, y_pred_test, color="orange", edgecolor="black", alpha=0.7, zorder=0, label="Test")
    ax4.plot([axis_min, axis_max], [axis_min, axis_max], color="black", zorder=1)
    ax4.set_xlim(left=axis_min, right=axis_max)
    ax4.set_ylim(bottom=axis_min, top=axis_max)
    ax4.set_title("All")
    ax4.set_xlabel("Observations")
    ax4.set_ylabel("Predictions")
    ax4.legend(loc="upper left")
    fig.savefig(os.path.join(save_dir, f"images/Plot_{data_name}.png"), bbox_inches="tight")
