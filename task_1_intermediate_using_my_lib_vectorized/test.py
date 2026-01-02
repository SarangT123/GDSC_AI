# test the mnist test dataset accuracy
import numpy as np
from neural_network_vectorized import NeuralNetwork
from neural_network_vectorized import load_network
import yaml
from mnist import MNIST


def load_mnist(path='mnist_files', kind='test'):
    """Load MNIST data from `path`"""
    mndata = MNIST(path)
    mndata.gz = True
    if kind == 'train':
        images, labels = mndata.load_training()
    else:
        images, labels = mndata.load_testing()
    return np.array(images), np.array(labels)


# Load MNIST test data
test_data = load_mnist('mnist_files', kind='t10k')
X_test, y_test_labels = test_data
X_test = X_test.astype(np.float32) / 255.0
def to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


y_test = to_one_hot(y_test_labels)


# Load model
model = load_network("mnist_model.nn")


# Evaluate accuracy
predictions = model.calculate_output(X_test)
predicted_labels = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_labels == y_test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")