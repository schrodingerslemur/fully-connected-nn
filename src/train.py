import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init() 

import sys
sys.path.append('../')

from network import Layer_Dense
from loss import Activation_Softmax_Loss_CategoricalCrossentropy
from activation import Activation_ReLU
from optimizer import Optimizer_Adam

def load_data(samples=100, classes=3):
    X, y = spiral_data(samples=samples, classes=classes) # type: ignore
    return X, y

def train(X, y, input_neurons, hidden_neurons, output_neurons, n_epochs=10001, learning_rate=0.005, decay=5e-7):
    # Create network, activation, optimizer and loss layers
    dense1 = Layer_Dense(input_neurons, hidden_neurons)
    activation1 = Activation_ReLU()

    dense2 = Layer_Dense(hidden_neurons, output_neurons)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=decay)

    # Per-epoch metric history for plotting
    loss_history = []
    accuracy_history = []

    # Weight snapshots for boundary frames, copied because the optimizer updates arrays in place
    weight_snapshots = []

    # ~10 evenly spaced progress lines regardless of n_epochs
    print_every = max(1, n_epochs // 10)

    # Training loop
    for epoch in range(n_epochs):
        # Forward pass
        dense1.forward(X)
        activation1.forward(dense1.output)

        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        # Accuracy check
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        loss_history.append(loss)
        accuracy_history.append(accuracy)

        # Print epoch, accuracy and loss
        if epoch % print_every == 0 or epoch == n_epochs - 1:
            print(f'epoch: {epoch}, acc: {(accuracy*100):.2f}%, loss: {loss:.3f}')
            weight_snapshots.append((epoch, dense1.weights.copy(), dense1.biases.copy(),
                                     dense2.weights.copy(), dense2.biases.copy()))
            
        # Backward passes
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Optimization
        optimizer.pre_update_params() 
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params() 

    return dense1.weights, dense1.biases, dense2.weights, dense2.biases, loss_history, accuracy_history, weight_snapshots

if __name__ == '__main__':
    X,y = load_data()
    w1, b1, w2, b2, loss_history, accuracy_history, weight_snapshots = train(X, y, 2, 64, 3) # 3 different classes

    with open('model.txt', 'w') as f:
        for name, array in (('w1', w1), ('b1', b1), ('w2', w2), ('b2', b2)):
            f.write(f'{name} {array.shape}:\n')
            f.write(np.array2string(array, precision=4, suppress_small=True, max_line_width=120))
            f.write('\n\n')
    print('Saved weights to model.txt')

    if '--plot' in sys.argv:
        import os
        from visualize import plot_training, plot_decision_boundary
        plot_training(loss_history, accuracy_history)

        os.makedirs('boundary_frames', exist_ok=True)
        for epoch, sw1, sb1, sw2, sb2 in weight_snapshots:
            plot_decision_boundary(X, y, sw1, sb1, sw2, sb2,
                                   save_path=f'boundary_frames/epoch_{epoch:05d}.png',
                                   title=f'Decision Boundary - Epoch {epoch}')
        print(f'Saved {len(weight_snapshots)} boundary frames in boundary_frames/')
