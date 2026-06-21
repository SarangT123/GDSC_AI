import pickle


def save_network(model, filename="model.nn", **metadata):
    with open(filename, 'wb') as f:
        pickle.dump({'model': model, **metadata}, f)


def load_network(filename="model.nn"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['model']
