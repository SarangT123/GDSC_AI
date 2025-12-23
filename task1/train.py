import numpy as np
from neural_network import NeuralNetwork, DataPoint
import yaml


#config file loading
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

# step 3 : normalize features
X = np.array([row[0] for row in processed_data])
y = np.array([row[1] for row in processed_data])

if config['training-parameters']['normalization']:
    # calculate mean and std for each feature
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    X_normalized = (X - means) / stds
else:
    X_normalized = X


print(f"Number of features: {X_normalized.shape[1]}")
print(f"Number of samples: {X_normalized.shape[0]}")

# step 4 train-test split 
split = config['training-parameters']['train_test_split']
split_index = int(split * len(X_normalized))

X_train = X_normalized[:split_index]
y_train = y[:split_index]
X_test = X_normalized[split_index:]
y_test = y[split_index:]

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Step 5: Convert to DataPoint objects
train_data = []
for i in range(len(X_train)):
    dp = DataPoint()
    dp.inputs = X_train[i]
    dp.expected_outputs = np.array([y_train[i]])
    train_data.append(dp)

test_data = []
for i in range(len(X_test)):
    dp = DataPoint()
    dp.inputs = X_test[i]
    dp.expected_outputs = np.array([y_test[i]])
    test_data.append(dp)

# Step 6: Create the neural network
layer_sizes = config['neural-network-architecture']
model = NeuralNetwork(layer_sizes)

print(f"\nNeural Network Architecture: {layer_sizes}")

# Step 7: Train the model
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
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        correct = 0
        sample_size = min(1000, len(train_data))
        for dp in train_data[:sample_size]:
            output = model.calculate_output(dp.inputs)
            prediction = 1 if output[-1] > 0.5 else 0  # Remove [0] here too
            if prediction == dp.expected_outputs[0]:
                correct += 1
        
        accuracy = (correct / sample_size) * 100
        print(f"Epoch {epoch+1}/{epochs} - Training Accuracy: {accuracy:.2f}%")

# Step 8: Evaluate on test set
print("\n" + "="*50)
print("FINAL EVALUATION ON TEST SET")
print("="*50)

correct = 0
for dp in test_data:
    output = model.calculate_output(dp.inputs)
    prediction = 1 if output[-1] > 0.5 else 0  
    if prediction == dp.expected_outputs[0]:
        correct += 1

test_accuracy = (correct / len(test_data)) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Step 9: Save the model (using pickle)
import pickle

print("\nSaving model...")
with open('placement_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'means': means,
        'stds': stds
    }, f)

print("Model saved as 'placement_model.pkl'")
print("\nTraining complete!")