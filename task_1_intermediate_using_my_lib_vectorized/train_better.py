# python-mnist is used to handle the filetype that mnist data gives
from mnist import MNIST
import numpy as np
import yaml
import pickle

from neural_network_vectorized_multiple_models import Model_ReLU_CCEL_Softmax, Model_Sig_SL




# Load MNIST
mndata = MNIST('mnist_files')
mndata.gz = True

images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()



# Load 
config_file = 'config.yaml'
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

print("Configuration Loaded:")
print(yaml.dump(config, default_flow_style=False))

if input("Proceed with these settings? (y/n): ").lower() != 'y':
    print("Exiting.")
    exit()

print(config)

if config['network-functions']['model-type'] == "1":
    NeuralNetwork, Data = Model_ReLU_CCEL_Softmax.NeuralNetwork, Model_ReLU_CCEL_Softmax.Data
elif config['network-functions']['model-type'] == "2":
    NeuralNetwork, Data = Model_Sig_SL.NeuralNetwork, Model_Sig_SL.Data
else:
    print("Invalid model type specified in configuration.")
    exit()
    


# Preprocess data
X_train = np.array(images, dtype=np.float32) / 255.0
X_test  = np.array(test_images, dtype=np.float32) / 255.0




def to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


y_train = to_one_hot(labels)
y_test  = to_one_hot(test_labels)

# We will combine original , left shifted, right shifted, up shifted, down shifted and noisy images and extend the labels accordingly
X_train_extended = []
y_train_extended = []
for img, label in zip(X_train, y_train):
    X_train_extended.append(img)  # Original
    y_train_extended.append(label)

    img_2d = img.reshape(28, 28)

    # Left shift
    left_shifted = np.roll(img_2d, -1, axis=1)
    left_shifted[:, -1] = 0
    X_train_extended.append(left_shifted.flatten())
    y_train_extended.append(label)

    # Right shift
    right_shifted = np.roll(img_2d, 1, axis=1)
    right_shifted[:, 0] = 0
    X_train_extended.append(right_shifted.flatten())
    y_train_extended.append(label)

    # Up shift
    up_shifted = np.roll(img_2d, -1, axis=0)
    up_shifted[-1, :] = 0
    X_train_extended.append(up_shifted.flatten())
    y_train_extended.append(label)

    # Down shift
    down_shifted = np.roll(img_2d, 1, axis=0)
    down_shifted[0, :] = 0
    X_train_extended.append(down_shifted.flatten())
    y_train_extended.append(label)

    # Noisy image
    noise = np.random.normal(0, 0.1, img_2d.shape)
    noisy_img = np.clip(img_2d + noise, 0, 1)
    X_train_extended.append(noisy_img.flatten())
    y_train_extended.append(label)
X_train = np.array(X_train_extended, dtype=np.float32)
y_train = np.array(y_train_extended, dtype=np.float32)


print(f"Train samples: {X_train.shape[0]}")
print(f"Test samples : {X_test.shape[0]}")
print(f"Input size   : {X_train.shape[1]}")



# Create model
layer_sizes = config['network-architecture']
model = NeuralNetwork(layer_sizes)

print(f"\nNeural Network Architecture: {layer_sizes}")



# Training parameters
epochs = config['training-parameters']['epochs']
batch_size = config['training-parameters']['batch_size']
learning_rate = config['training-parameters']['learning_rate']

num_samples = X_train.shape[0]




# Training loop
print("\nStarting training...\n")

for epoch in range(epochs):

    # Shuffle dataset
    indices = np.random.permutation(num_samples)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    # Mini-batch training
    for i in range(0, num_samples, batch_size):
        x_b = X_shuffled[i:i + batch_size]
        y_b = y_shuffled[i:i + batch_size]

        data = Data(x_b=x_b, y_b_target=y_b)

        # Forward + backward + update
        model.learn(data, learning_rate)

    # ---- Training accuracy (sampled) ----
    sample_size = min(1000, num_samples)
    outputs = model.calculate_output(X_train[:sample_size])

    predictions = np.argmax(outputs, axis=1)
    actuals = np.argmax(y_train[:sample_size], axis=1)

    accuracy = np.mean(predictions == actuals) * 100

    print(f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {accuracy:.2f}%")



# Test evaluation
print("\n" + "=" * 50)
print("FINAL EVALUATION ON TEST SET")
print("=" * 50)

outputs = model.calculate_output(X_test)
predictions = np.argmax(outputs, axis=1)
actuals = np.argmax(y_test, axis=1)

test_accuracy = np.mean(predictions == actuals) * 100

print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Correct predictions: {np.sum(predictions == actuals)}/{len(X_test)}")



# Per-digit accuracy
confusion_matrix = np.zeros((10, 10), dtype=int)

for actual, predicted in zip(actuals, predictions):
    confusion_matrix[actual][predicted] += 1

print("\nPer-digit accuracy:")
for digit in range(10):
    total = confusion_matrix[digit].sum()
    correct = confusion_matrix[digit][digit]
    acc = (correct / total * 100) if total > 0 else 0
    print(f"Digit {digit}: {acc:.2f}% ({correct}/{total})")



# Save model

print("\nSaving model...")
with open('mnist_model.nn', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as 'mnist_model.nn'")
print("\nTraining complete!")
