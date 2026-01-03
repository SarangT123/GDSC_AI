import pickle
import numpy as np
def save_network(means,stds,model,filename="model.nn"):
    with open(filename, 'wb') as f:
        pickle.dump({
            'model': model,
        }, f)


def load_network(filename="model.nn") -> tuple:
    """
    Docstring for load_network
    
    :param filename: filename
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return (data['model'])