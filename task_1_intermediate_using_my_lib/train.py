# python-mnist is used to handle the filetype that mnist data gives, alternatively tensorflow/keras can be used as well or direct import using gzip which i dont wanna deal with 

from mnist import MNIST
# this is bloated i will replace this later
import numpy as np
from neural_network import NeuralNetwork, DataPoint
import yaml

mndata = MNIST('mnist_files')
mndata.gz = True
images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# Config file loading
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
# print config to verify
print("Configuration Loaded:")
print(yaml.dump(config, default_flow_style=False))
if input("Proceed with these settings? (y/n): ").lower() != 'y':
    print("Exiting. Please adjust the config.yaml file as needed.")
    exit()


# Normalize features
X_train = np.array(images[:1000], dtype=np.float32) / 255.0  # Normalize to [0, 1]
X_test = np.array(test_images[:1000], dtype=np.float32) / 255.0

"""
The output needs onehot encoding since we are using 
np.argmax() in the NN class 
"""
def to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

y_train = to_one_hot(labels[:1000])
y_test = to_one_hot(test_labels[:1000])

print(f"Number of features: {X_train.shape[1]}")
print(f"Number of samples: {X_train.shape[0]}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Convert to DataPoint objects
train_data = []
for i in range(len(X_train)):
    dp = DataPoint()
    dp.inputs = X_train[i]
    dp.expected_outputs = y_train[i]  # No extra array wrapper!
    train_data.append(dp)

test_data = []
for i in range(len(X_test)):
    dp = DataPoint()
    dp.inputs = X_test[i]
    dp.expected_outputs = y_test[i]
    test_data.append(dp)

# Create the neural network
layer_sizes = config['network-architecture']
model = NeuralNetwork(layer_sizes)

print(f"\nNeural Network Architecture: {layer_sizes}")

# Train the model
print("\nStarting training...")

epochs = config['training-parameters']['epochs']
batch_size = config['training-parameters']['batch_size']
learn_rate = config['training-parameters']['learning_rate']

for epoch in range(epochs):
    # Shuffle training data each epoch
    indices = np.random.permutation(len(train_data))
    shuffled_train = [train_data[i] for i in indices]
    
    # Train in batches
    for i in range(0, len(shuffled_train), batch_size):
        batch = shuffled_train[i:i+batch_size]
        model.learn(batch, learn_rate)
    
    # Print progress every epoch
    if (epoch + 1) % 1 == 0:
        correct = 0
        sample_size = min(1000, len(train_data))
        for dp in train_data[:sample_size]:
            output = model.calculate_output(dp.inputs)
            prediction = np.argmax(output)  # Get digit with highest probability
            actual = np.argmax(dp.expected_outputs)  # Get actual digit
            if prediction == actual:
                correct += 1
        
        accuracy = (correct / sample_size) * 100
        print(f"Epoch {epoch+1}/{epochs} - Training Accuracy: {accuracy:.2f}%")

# Final evaluation on test set
print("\n" + "="*50)
print("FINAL EVALUATION ON TEST SET")
print("="*50)

correct = 0
for dp in test_data:
    output = model.calculate_output(dp.inputs)
    prediction = np.argmax(output)
    actual = np.argmax(dp.expected_outputs)
    if prediction == actual:
        correct += 1

test_accuracy = (correct / len(test_data)) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Correct predictions: {correct}/{len(test_data)}")

# Per-digit accuracy
print("\nPer-digit accuracy:")
confusion_matrix = np.zeros((10, 10), dtype=int)
for dp in test_data:
    output = model.calculate_output(dp.inputs)
    prediction = np.argmax(output)
    actual = np.argmax(dp.expected_outputs)
    confusion_matrix[actual][prediction] += 1

for digit in range(10):
    digit_total = confusion_matrix[digit].sum()
    digit_correct = confusion_matrix[digit][digit]
    digit_accuracy = (digit_correct / digit_total * 100) if digit_total > 0 else 0
    print(f"Digit {digit}: {digit_accuracy:.2f}% ({digit_correct}/{digit_total})")

# Save the model
import pickle

print("\nSaving model...")
with open('mnist_model.nn', 'wb') as f:
    pickle.dump({
        'model': model
    }, f)

print("Model saved as 'mnist_model.nn'")
print("\nTraining complete!")