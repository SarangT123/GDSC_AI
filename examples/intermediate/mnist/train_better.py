from mnist import MNIST
import numpy as np
import yaml
import pickle

from nnlib import Sequential, Data


mndata = MNIST('mnist_files')
mndata.gz = True

images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()


def to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


with open('config.yaml') as f:
    config = yaml.safe_load(f)

print(yaml.dump(config, default_flow_style=False))
if input("Proceed? (y/n): ").lower() != 'y':
    exit()

X_train = np.array(images, dtype=np.float32) / 255.0
X_test  = np.array(test_images, dtype=np.float32) / 255.0
y_train = to_one_hot(labels)
y_test  = to_one_hot(test_labels)

X_train_extended = []
y_train_extended = []
for img, label in zip(X_train, y_train):
    X_train_extended.append(img)
    y_train_extended.append(label)

    img_2d = img.reshape(28, 28)

    left = np.roll(img_2d, -1, axis=1)
    left[:, -1] = 0
    X_train_extended.append(left.flatten())
    y_train_extended.append(label)

    right = np.roll(img_2d, 1, axis=1)
    right[:, 0] = 0
    X_train_extended.append(right.flatten())
    y_train_extended.append(label)

    up = np.roll(img_2d, -1, axis=0)
    up[-1, :] = 0
    X_train_extended.append(up.flatten())
    y_train_extended.append(label)

    down = np.roll(img_2d, 1, axis=0)
    down[0, :] = 0
    X_train_extended.append(down.flatten())
    y_train_extended.append(label)

    noise = np.random.normal(0, 0.1, img_2d.shape)
    noisy = np.clip(img_2d + noise, 0, 1)
    X_train_extended.append(noisy.flatten())
    y_train_extended.append(label)

X_train = np.array(X_train_extended, dtype=np.float32)
y_train = np.array(y_train_extended, dtype=np.float32)

model = Sequential(layer_sizes=config['network-architecture'])

print(f"Train (augmented): {X_train.shape[0]}, Test: {X_test.shape[0]}")

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
