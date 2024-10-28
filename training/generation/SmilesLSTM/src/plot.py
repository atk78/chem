import os

import matplotlib.pyplot as plt


config = {
    "font.family": "Arial",
    "font.size": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.right": True,
    "legend.fancybox": False,
    "legend.framealpha": 1,
    "legend.edgecolor": "black"
}


def plot_minibatch_loss(train_loss, valid_loss, img_dir):
    train_step = train_loss[:, 0]
    train_loss = train_loss[:, 1]
    valid_step = valid_loss[:, 0]
    valid_loss = valid_loss[:, 1]
    fig = plt.figure()
    plt.rcParams.update(config)
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
    # ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1))
    ax.legend(loc="best")
    ax.set_xlabel("# of batchs")
    ax.set_ylabel("Loss")
    plt.tight_layout()
    fig.savefig(fname=os.path.join(img_dir, "History.png"))
    plt.show()
