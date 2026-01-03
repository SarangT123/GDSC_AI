import numpy as np
from neural_network_vectorized import NeuralNetwork, Data
import yaml


# Config file loading
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Load CSV 
data = []
filename = 'Placement_BeginnerTask01.csv'
with open(filename, 'r') as f:
    lines = f.readlines()
    headers = lines[0].strip().split(',')
    print(f"Headers: {headers}")
    
    for line in lines[1:]:  # Skip header
        values = line.strip().split(',')
        data.append(values)

print(f"Loaded {len(data)} rows")

# Convert to numpy array and preprocess
print("Preprocessing...")

# Extract columns (skip StudentID column 0)
processed_data = []
for row in data:
    processed_row = []
    
    # CGPA (column 1)
    processed_row.append(float(row[1]))
    
    # Internships (column 2)
    processed_row.append(int(row[2]))
    
    # Projects (column 3)
    processed_row.append(int(row[3]))
    
    # Workshops/Certifications (column 4)
    processed_row.append(int(row[4]))
    
    # AptitudeTestScore (column 5)
    processed_row.append(int(row[5]))
    
    # SoftSkillsRating (column 6)
    processed_row.append(float(row[6]))
    
    # ExtracurricularActivities (column 7): Yes=1, No=0
    processed_row.append(1 if row[7] == 'Yes' else 0)
    
    # PlacementTraining (column 8): Yes=1, No=0
    processed_row.append(1 if row[8] == 'Yes' else 0)
    
    # SSC_Marks (column 9)
    processed_row.append(int(row[9]))
    
    # HSC_Marks (column 10)
    processed_row.append(int(row[10]))
    
    # PlacementStatus (column 11): Placed=1, Not Placed=0
    target = 1 if row[11] == 'Placed' else 0
    
    processed_data.append((processed_row, target))

# Step 3: Normalize features
X = np.array([row[0] for row in processed_data])
y = np.array([row[1] for row in processed_data])

if config['training-parameters']['normalization']:
    # Calculate mean and std for each feature
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    # Avoid division by zero
    stds = np.where(stds == 0, 1, stds)
    X_normalized = (X - means) / stds
else:
    X_normalized = X
    means = np.zeros(X.shape[1])
    stds = np.ones(X.shape[1])


print(f"Number of features: {X_normalized.shape[1]}")
print(f"Number of samples: {X_normalized.shape[0]}")

# Step 4: Train-test split 
split = config['training-parameters']['train_test_split']
split_index = int(split * len(X_normalized))

X_train = X_normalized[:split_index]
y_train = y[:split_index]
X_test = X_normalized[split_index:]
y_test = y[split_index:]

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Step 5: Convert labels to one-hot encoding for softmax output
# For binary classification: [1, 0] = Not Placed, [0, 1] = Placed
y_train_onehot = np.zeros((len(y_train), 2))
y_train_onehot[np.arange(len(y_train)), y_train] = 1

y_test_onehot = np.zeros((len(y_test), 2))
y_test_onehot[np.arange(len(y_test)), y_test] = 1

# Step 6: Create the neural network
# Modify layer sizes to have 2 output neurons for binary classification
layer_sizes = config['network-architecture']
if layer_sizes[-1] == 1:
    layer_sizes[-1] = 2  # Change output to 2 neurons for softmax
    print(f"Modified output layer to 2 neurons for softmax classification")

model = NeuralNetwork(layer_sizes)

print(f"\nNeural Network Architecture: {layer_sizes}")

# Step 7: Train the model
print("\nStarting training...")

epochs = config['training-parameters']['epochs']
batch_size = config['training-parameters']['batch_size']
learn_rate = config['training-parameters']['learning_rate']

for epoch in range(epochs):
    # Shuffle training data each epoch
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train_onehot[indices]
    
    # Train in batches
    for i in range(0, len(X_train_shuffled), batch_size):
        batch_end = min(i + batch_size, len(X_train_shuffled))
        X_batch = X_train_shuffled[i:batch_end]
        y_batch = y_train_shuffled[i:batch_end]
        
        # Create Data object for batch
        batch_data = Data(X_batch, y_batch)
        model.learn(batch_data, learn_rate)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        # Evaluate on training set
        sample_size = min(1000, len(X_train))
        train_sample_data = Data(X_train[:sample_size], y_train_onehot[:sample_size])
        train_accuracy = model.accuracy(train_sample_data) * 100
        train_loss = model.avg_loss(train_sample_data)
        
        print(f"Epoch {epoch+1}/{epochs} - Training Accuracy: {train_accuracy:.2f}% - Loss: {train_loss:.4f}")

# Step 8: Evaluate on test set
print("\n" + "="*50)
print("FINAL EVALUATION ON TEST SET")
print("="*50)

test_data = Data(X_test, y_test_onehot)
test_accuracy = model.accuracy(test_data) * 100
test_loss = model.avg_loss(test_data)

print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Additional metrics
predictions = model.predict(X_test)
true_labels = y_test

# Confusion matrix components
true_positives = np.sum((predictions == 1) & (true_labels == 1))
true_negatives = np.sum((predictions == 0) & (true_labels == 0))
false_positives = np.sum((predictions == 1) & (true_labels == 0))
false_negatives = np.sum((predictions == 0) & (true_labels == 1))

print(f"\nConfusion Matrix:")
print(f"True Positives (Placed correctly): {true_positives}")
print(f"True Negatives (Not Placed correctly): {true_negatives}")
print(f"False Positives (Incorrectly predicted Placed): {false_positives}")
print(f"False Negatives (Incorrectly predicted Not Placed): {false_negatives}")

if (true_positives + false_positives) > 0:
    precision = true_positives / (true_positives + false_positives)
    print(f"\nPrecision: {precision:.2f}")

if (true_positives + false_negatives) > 0:
    recall = true_positives / (true_positives + false_negatives)
    print(f"Recall: {recall:.2f}")

# Step 9: Save the model
import pickle

print("\nSaving model...")
with open('placement_model.nn', 'wb') as f:
    pickle.dump({
        'model': model,
    }, f)

print("Model saved as 'placement_model.nn'")
print("\nTraining complete!")