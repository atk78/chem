import os
import matplotlib.pyplot as plt

config = {
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


def plot_minibatch_loss(train_loss, valid_loss, img_dir=""):
    plt.rcParams.update(config)
    train_step = train_loss[:, 0]
    train_loss = train_loss[:, 1]
    valid_step = valid_loss[:, 0]
    valid_loss = valid_loss[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(train_step, train_loss, label="training loss", zorder=0)
    ax.scatter(
        valid_step,
        valid_loss,
        s=50,
        color="orange",
        edgecolors="black",
        zorder=1,
        marker="^",
        label="validation loss",
    )
    ax.legend()
    ax.set_xlabel("# of batchs")
    ax.set_ylabel("Loss function")
    fig.savefig(fname=os.path.join(img_dir, "History.png"))


def plot_reconstruction_rate(reconstruction_rate, img_dir):
    plt.rcParams.update(config)
    step = reconstruction_rate[:, 0]
    reconstruction_rate = reconstruction_rate[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(step, reconstruction_rate, label="reconstruction rate")
    ax.legend()
    ax.set_xlabel("# of batchs")
    ax.set_ylabel("Reconstruction rate")
    fig.savefig(fname=os.path.join(img_dir, "Reconstruction_rate.png"))
