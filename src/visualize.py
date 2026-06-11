def plot_training(loss_history, accuracy_history, save_path='training_curves.png'):
    # Imported here so training works without matplotlib installed
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax_loss.plot(loss_history)
    ax_loss.set_ylabel('Loss')

    ax_acc.plot(accuracy_history)
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_xlabel('Epoch')

    fig.suptitle('Training Loss and Accuracy')
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
