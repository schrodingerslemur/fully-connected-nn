import numpy as np

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

    print(f'Saved training curves to {save_path}')
    return save_path

def plot_decision_boundary(X, y, w1, b1, w2, b2, save_path='decision_boundary.png', title='Decision Boundary'):
    # Imported here so training works without matplotlib installed
    import matplotlib.pyplot as plt

    # Grid covering the data range
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 200),
                         np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Forward pass through the trained 2-layer network (argmax doesn't need softmax)
    hidden = np.maximum(0, grid @ w1 + b1)
    predictions = np.argmax(hidden @ w2 + b2, axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, predictions, alpha=0.3, cmap='brg')
    ax.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap='brg')
    ax.set_title(title)

    fig.savefig(save_path)
    plt.close(fig)
    return save_path
