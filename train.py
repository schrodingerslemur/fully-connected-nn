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

def train(X, y, input_neurons, hidden_neurons, output_neurons, n_epochs=10001, learning_rate=0.05, decay=5e-7):
    # Create network, activation, optimizer and loss layers
    dense1 = Layer_Dense(input_neurons, hidden_neurons)
    activation1 = Activation_ReLU()

    dense2 = Layer_Dense(hidden_neurons, output_neurons)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=decay)

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

        # Print epoch, accuracy and loss
        if not epoch % 100:
            print(f'epoch: {epoch}, acc: {(accuracy*100):.2f}%, loss: {loss:.3f}')
            
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

    return dense1.weights, dense1.biases, dense2.weights, dense2.biases
    
if __name__ == '__main__':
    X,y = load_data()
    w1, b1, w2, b2 = train(X, y, 2, 64, 3) # 3 different classes
    print('w1:', w1)
    print('b1:', b1)
    print('w2:', w2)
    print('b2:', b2)
