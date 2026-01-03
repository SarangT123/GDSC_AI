# Note
remember that use.py wont give as accurate results as test.py becuase it is not the same stroke length or details as mnist dataset

configuration file usage is as follows :

example config file : 
```yaml
network-architecture:
  - 784   # Input layer 
  - 256   # Hidden layer 1
  - 128   # Hidden layer 2
  - 64    # Hidden layer 3
  - 32    # Hidden layer 4
  - 10    # Output layer

network-functions:
  model-type: "1"
  # model 1 uses ReLU and Softmax(output) as activation functions and Cross-Entropy as loss function
  # model 2 uses Sigmoid as activation function and squared error as loss function

training-parameters:
  epochs: 20
  batch_size: 32
  learning_rate: 0.1
```
network-architecture : defines the architecture of the neural network
network-functions : defines which model to use
training-parameters : defines the training parameters such as epochs, batch size and learning rate
