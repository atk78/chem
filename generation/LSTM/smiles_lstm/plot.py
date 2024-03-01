import matplotlib.pyplot as plt


def plot_minibatch_loss(train_loss_list, valid_loss_list):
    train_t, train_loss = list(zip(*train_loss_list))
    valid_t, valid_loss = list(zip(*valid_loss_list))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(train_t, train_loss, label="train loss")
    ax.plot(valid_t, valid_loss, label="validation loss", marker="*")
    ax.legend()
    ax.set_xlabel("# of updates")
    ax.set_ylabel("Loss function")
    ax.show()
