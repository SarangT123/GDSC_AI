import pickle
import numpy as np

with open('placement_model.nn', 'rb') as f:
    data = pickle.load(f)
model, means, stds = data['model'], data['means'], data['stds']
print("Model loaded.")

filename = 'Placement_BeginnerTask01.csv'
with open(filename) as f:
    headers = f.readline().strip().split(',')
headers = headers[1:-1]

inputs = []
for h in headers:
    val = float(input(f"Enter value for {h}: "))
    inputs.append(val)

x = np.array([(np.array(inputs) - means) / stds])
prob = model.forward(x)[0, 0]

if prob >= 0.5:
    print(f"Likely placed ({prob:.2%} probability)")
else:
    print(f"Unlikely placed ({prob:.2%} probability)")
