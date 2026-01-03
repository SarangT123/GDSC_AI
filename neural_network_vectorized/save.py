import pickle
import numpy as np
from . import NeuralNetwork
def save_network(means,stds,model:NeuralNetwork,filename="model.nn"):
    with open(filename, 'wb') as f:
        pickle.dump({
            'model': model,
        }, f)


def load_network(filename="model.nn") -> tuple[NeuralNetwork, np.ndarray, np.ndarray]:
    """
    Docstring for load_network
    
    :param filename: filename
    :return: Loaded NeuralNetwork model, means, and stds
    :rtype: tuple[NeuralNetwork, ndarray[_AnyShape, dtype[Any]], ndarray[_AnyShape, dtype[Any]]]
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return (data['model'])