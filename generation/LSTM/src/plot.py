import os

import matplotlib.pyplot as plt


def plot_minibatch_loss(train_loss, valid_loss, img_dir):
    train_step = train_loss.iloc[:, 0]
    train_loss = train_loss.iloc[:, 1]
    valid_step = valid_loss.iloc[:, 0]
    valid_loss = valid_loss.iloc[:, 1]
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
    plt.show()
