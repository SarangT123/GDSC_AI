from mnist import MNIST
import numpy as np
import yaml
import pickle

from loom import Sequential, Data, sigmoid, sigmoid_derivative, square_loss, square_loss_derivative


mndata = MNIST('mnist_files')
mndata.gz = True

images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

with open('config.yaml') as f:
    config = yaml.safe_load(f)

print("Configuration:")
print(yaml.dump(config, default_flow_style=False))
if input("Proceed? (y/n): ").lower() != 'y':
    exit()

X_train = np.array(images, dtype=np.float32) / 255.0
X_test  = np.array(test_images, dtype=np.float32) / 255.0


def to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


y_train = to_one_hot(labels)
y_test  = to_one_hot(test_labels)

if config['network-functions']['model-type'] == "1":
    model = Sequential(layer_sizes=config['network-architecture'])
elif config['network-functions']['model-type'] == "2":
    model = Sequential(
        layer_sizes=config['network-architecture'],
        activation=sigmoid,
        da_dz=sigmoid_derivative,
        output_activation=sigmoid,
        output_da_dz=sigmoid_derivative,
        dL_da=square_loss_derivative,
        dL_dz_output=None,
        loss_fn=square_loss,
        weight_init='xavier',
    )
else:
    print("Invalid model-type.")
    exit()

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Architecture: {config['network-architecture']}")

epochs = config['training-parameters']['epochs']
batch_size = config['training-parameters']['batch_size']
lr = config['training-parameters']['learning_rate']
num_samples = X_train.shape[0]

for epoch in range(epochs):
    indices = np.random.permutation(num_samples)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    for i in range(0, num_samples, batch_size):
        batch = Data(
            x_b=X_shuffled[i:i + batch_size],
            y_b_target=y_shuffled[i:i + batch_size],
        )
        model.learn(batch, lr)

    sample = min(1000, num_samples)
    outputs = model.forward(X_train[:sample])
    preds = np.argmax(outputs, axis=1)
    acc = np.mean(preds == np.argmax(y_train[:sample], axis=1))
    print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {acc:.2%}")

outputs = model.forward(X_test)
preds = np.argmax(outputs, axis=1)
actuals = np.argmax(y_test, axis=1)
test_acc = np.mean(preds == actuals)
print(f"\nTest Accuracy: {test_acc:.2%}")

with open('mnist_model.nn', 'wb') as f:
    pickle.dump({'model': model}, f)
print("Model saved.")
