from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


plot_config = {
    "font.family": "sans-serif",
    "font.size": 12,
    "xtick.direction": "in",  # x軸の目盛りの向き
    "ytick.direction": "in",  # y軸の目盛りの向き
    "xtick.minor.visible": True,  # x軸補助目盛りの追加
    "ytick.minor.visible": True,  # y軸補助目盛りの追加
    "xtick.top": True,  # x軸上部の目盛り
    "ytick.right": True,  # y軸左部の目盛り
    "legend.fancybox": False,  # 凡例の角
    "legend.framealpha": 1,  # 枠の色の塗りつぶし
    "legend.edgecolor": "black",  # 枠の色
}


# def plot_history_loss(
#     train_loss: np.ndarray,
#     train_r2: np.ndarray,
#     valid_loss: np.ndarray,
#     valid_r2: np.ndarray,
#     loss_func: str,
#     img_dir: str = ""
# ):
#     plt.rcParams.update(config)
#     fig = plt.figure(figsize=(6, 8))
#     ax1 = fig.add_subplot(2, 1, 1)
#     ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
#     ax1.plot(train_loss, color="black", label="train")
#     ax1.plot(valid_loss, color="red", linestyle="--", label="valid")
#     ax1.set_ylabel(loss_func)
#     ax1.legend()
#     ax2.plot(train_r2, color="black", label="train")
#     ax2.plot(valid_r2, color="red", linestyle="--", label="valid")
#     ax2.set_ylabel("$R^2$")
#     ax2.set_xlabel("Epochs")
#     ax2.legend()
#     fig.tight_layout()
#     fig.savefig(
#         fname=os.path.join(img_dir, "History.png"),
#         bbox_inches="tight"
#     )


def plot_obserbations_vs_predictions(
    result: dict,
    img_dir: Path,
):
    plt.rcParams.update(plot_config)
    all_y = []
    colors = ["royalblue", "green", "orange"]
    for value in result.values():
        all_y.extend(value["y"])
        all_y.extend(value["y_pred"])
    axis_min, axis_max = min(all_y), max(all_y)
    axis_min -= 0.1 * (axis_max - axis_min)
    axis_max += 0.1 * (axis_max - axis_min)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i, phase in enumerate(result.keys()):
        q, r = divmod(i, 2)
        plot_axis(
            ax[q, r],
            result[phase]["y"], result[phase]["y_pred"],
            axis_min, axis_max,
            colors[i]
        )
        ax[q, r].set_title(
            f"{phase}: $R^2$ = {result[phase]['R2']:.2f}, RMSE = {result[phase]['RMSE']:.2f}"
        )
        plot_axis(
            ax[1, 1],
            result[phase]["y"], result[phase]["y_pred"],
            axis_min, axis_max,
            colors[i],
            label=phase
        )
    ax[1, 1].set_title("All")
    ax[1, 1].set_xlabel("Observations")
    ax[1, 1].set_ylabel("Predictions")
    ax[1, 1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(
        fname=img_dir.joinpath("Plot.png"),
        bbox_inches="tight"
    )


def plot_axis(ax, y_true, y_pred, axis_min, axis_max, color, label=None):
    ax.scatter(
        x=y_true,
        y=y_pred,
        c=color,
        edgecolor="black",
        alpha=0.7,
        zorder=0,
        label=label
    )
    ax.plot(
        [axis_min, axis_max], [axis_min, axis_max], c="black", zorder=1
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=axis_min, right=axis_max)
    ax.set_ylim(bottom=axis_min, top=axis_max)
    ax.set_xlabel("Observations")
    ax.set_ylabel("Predictions")
    ax.set_aspect(aspect="equal", adjustable="box")
