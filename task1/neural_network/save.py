import pickle
from .model import NeuralNetwork
def save_network(network:NeuralNetwork,filename="model.nn"):
    with open(filename, 'wb') as f:
        pickle.dump(network, f)


def load_network(filename="model.nn")->NeuralNetwork:
    with open(filename, 'rb') as f:
        network = pickle.load(f)
    return network