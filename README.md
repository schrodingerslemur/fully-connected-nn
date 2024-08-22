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

\`\`\`bash
git clone https://github.com/schrodingerslemur/fully-connected-nn.git
cd fully-connected-nn
\`\`\`

### Usage

1. **Define the Model:**
   Customize your network in \`network.py\`.

2. **Choose Activation, Loss, and Optimizer:**
   Specify your choices in \`activation.py\`, \`loss.py\`, and \`optimizer.py\`.

3. **Train the Model:**
   Run the \`train.py\` script to train the model:

   \`\`\`bash
   python train.py
   \`\`\`

4. **Evaluate the Model:**
   Add evaluation scripts to validate your model's performance.

### File Overview

- \`activation.py\`: Contains activation functions.
- \`loss.py\`: Defines loss functions.
- \`network.py\`: Implements the neural network architecture.
- \`optimizer.py\`: Includes the optimizer logic.
- \`train.py\`: Script for training the neural network.

### Contributing

Feel free to submit issues or pull requests if you have suggestions for improving the code.

### License

This project is licensed under the MIT License.
EOL
