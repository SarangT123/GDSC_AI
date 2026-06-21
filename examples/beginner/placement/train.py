import pickle
import numpy as np
import yaml

from nnlib import Sequential, Data, sigmoid, sigmoid_derivative, square_loss, square_loss_derivative


with open('config.yaml') as f:
    config = yaml.safe_load(f)

filename = 'Placement_BeginnerTask01.csv'
with open(filename) as f:
    lines = f.readlines()
    headers = lines[0].strip().split(',')
    print(f"Headers: {headers}")

data = []
for line in lines[1:]:
    values = line.strip().split(',')
    data.append(values)

print(f"Loaded {len(data)} rows")

processed_data = []
for row in data:
    processed_row = [
        float(row[1]),
        int(row[2]),
        int(row[3]),
        int(row[4]),
        int(row[5]),
        float(row[6]),
        1 if row[7] == 'Yes' else 0,
        1 if row[8] == 'Yes' else 0,
        int(row[9]),
        int(row[10]),
    ]
    target = 1 if row[11] == 'Placed' else 0
    processed_data.append((processed_row, target))

X = np.array([r[0] for r in processed_data])
y = np.array([r[1] for r in processed_data])

if config['training-parameters']['normalization']:
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    X = (X - means) / stds

split = config['training-parameters']['train_test_split']
split_idx = int(split * len(X))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_data = Data(x_b=X_train, y_b_target=y_train.reshape(-1, 1))
test_data = Data(x_b=X_test, y_b_target=y_test.reshape(-1, 1))

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

epochs = config['training-parameters']['epochs']
batch_size = config['training-parameters']['batch_size']
lr = config['training-parameters']['learning_rate']

print(f"\nArchitecture: {config['network-architecture']}")
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n")

for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    for i in range(0, len(X_train), batch_size):
        batch_idx = indices[i:i + batch_size]
        batch = Data(x_b=X_train[batch_idx], y_b_target=y_train[batch_idx].reshape(-1, 1))
        model.learn(batch, lr)

    if (epoch + 1) % 10 == 0:
        acc = model.accuracy(train_data)
        print(f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {acc:.2%}")

test_acc = model.accuracy(test_data)
print(f"\nTest Accuracy: {test_acc:.2%}")

with open('placement_model.nn', 'wb') as f:
    pickle.dump({'model': model, 'means': means, 'stds': stds}, f)
print("Model saved.")
