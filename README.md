# Fully Connected Neural Network

This repository implements a fully connected neural network from scratch using only Python and NumPy, without relying on traditional machine learning or neural network libraries.

## Features

- **Customizable Neural Network Architecture:** Define your own network structure with varying layers and neurons.
- **Activation Functions:** Includes ReLU and Sigmoid.
- **Loss Functions:** Implementations of Mean Squared Error (MSE) and Cross-Entropy Loss.
- **Optimizers:** Gradient Descent with custom learning rates.
- **Training Script:** Train your model on custom datasets.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy

### Installation

Clone the repository:

```bash
git clone https://github.com/schrodingerslemur/fully-connected-nn.git
cd fully-connected-nn
```

Install the required Python packages:

```bash
pip install numpy
```

### Usage

1. **Define the Model:**
   Customize your network in \`network.py\`.

2. **Choose Activation, Loss, and Optimizer:**
   Specify your choices in \`activation.py\`, \`loss.py\`, and \`optimizer.py\`.

3. **Train the Model:**
   Run the \`train.py\` script to train the model:

   ```bash
   python train.py
   ```

4. **Evaluate the Model:**
   Add evaluation scripts to validate your model's performance.

### Implementation Example

Here’s an example of how to implement and use the classes in this repository:

```python
import numpy as np
from network import Layer_Dense
from loss import Activation_Softmax_Loss_CategoricalCrossentropy
from activation import Activation_ReLU
from optimizer import Optimizer_Adam

def load_data():
    # Example data - replace with your own
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([0, 1, 1, 0])
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
        dense1.forward(X)
        activation1.forward(dense1.output)

        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y)

        if not epoch % 100:
            print(f'epoch: {epoch}, acc: {(accuracy*100):.2f}%, loss: {loss:.3f}')
            
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params() 
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    return dense1.weights, dense1.biases, dense2.weights, dense2.biases
    
if __name__ == '__main__':
    X, y = load_data()
    w1, b1, w2, b2 = train(X, y, 2, 64, 3) # Adjust parameters as needed
    print('w1:', w1)
    print('b1:', b1)
    print('w2:', w2)
    print('b2:', b2)
```
### File Overview

- `activation.py`: Contains activation functions.
- `loss.py`: Defines loss functions.
- `network.py`: Implements the neural network architecture.
- `optimizer.py`: Includes the optimizer logic.
- `train.py`: Script for training the neural network.

### Contributing

Feel free to submit issues or pull requests if you have suggestions for improving the code.

### License

This project is licensed under the MIT License.
