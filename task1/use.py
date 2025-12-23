from neural_network import NeuralNetwork, DataPoint, Layer, save_network, load_network
import numpy as np
# Loading the netwrork from placement_model.pkl
network, means, stds = load_network('placement_model.nn')
print("Network loaded successfully.")


# We have 10 inputs for the 10 questions
# load headers from placement_BegginerTask01.csv
filename = 'Placement_BeginnerTask01.csv'
with open(filename, 'r') as f:
    lines = f.readlines()
    headers = lines[0].strip().split(',')
    print(f"Headers: {headers}")

headers.pop(0)  # Remove StudentID  
headers.pop(-1) # Remove PlacementStatus

inputs = []
for i in range(0,len(headers)):
    user_input = input(f"Enter value for {headers[i]}: ")  # +1 to skip StudentID
    inputs.append(float(user_input))

# Convert inputs to numpy array
inputs_array = np.array(inputs)

# Calculate output
output = network.calculate_output((inputs_array - means) / stds)
print(f"Network output: {output}")
placement_probability = output[0]
if placement_probability >= 0.5:
    print(f"The model predicts that you are likely to get placed with a probability of {placement_probability*100:.2f}%.")
else:
    print(f"The model predicts that you are unlikely to get placed with a probability of {placement_probability*100:.2f}%.")

