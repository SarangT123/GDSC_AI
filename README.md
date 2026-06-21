# nnlib — Neural Networks from Scratch

A minimal neural network library built from scratch using only NumPy. Designed for learning how backpropagation and gradient descent work under the hood.

## Installation

```bash
pip install -e .
```

Requirements: `numpy`

## Quickstart

```python
from nnlib import Sequential, Data, save_network

# 784 inputs → 128 hidden (ReLU) → 10 outputs (softmax)
model = Sequential([784, 128, 10])

data = Data(x_b=X_batch, y_b_target=y_onehot_batch)
model.learn(data, learning_rate=0.1)

# Evaluate
test_data = Data(x_b=X_test, y_b_target=y_test)
print(f"Accuracy: {model.accuracy(test_data):.2%}")

save_network(model, "model.nn")
```

## Architecture

| Module | Description |
|--------|-------------|
| `Sequential` | Feed-forward neural network |
| `Layer` | Single layer with forward/backward pass |
| `activations` | Sigmoid, ReLU, Softmax, Tanh, Linear |
| `losses` | Square loss, categorical/binary cross-entropy |
| `Data` | Batched data container |
| `save.py` | Pickle-based save/load |

## Examples

```
examples/
├── beginner/placement/     # Placement prediction (sigmoid, square loss)
└── intermediate/mnist/     # MNIST digit classification (ReLU+softmax, CCEL)
```

## License

GPL v3
