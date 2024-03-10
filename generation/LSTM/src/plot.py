import matplotlib.pyplot as plt


def plot_minibatch_loss(valid_loss_list):
    # valid_t, valid_loss = list(zip(*valid_loss_list))
    fig = plt.figure()
    ax = fig.add_subplot()
    # ax.plot(valid_t, valid_loss, label="validation loss", marker="*")
    ax.plot(valid_loss_list, label="validation loss", marker="*")
    ax.legend()
    ax.set_xlabel("# of updates")
    ax.set_ylabel("Loss function")
    plt.show()

