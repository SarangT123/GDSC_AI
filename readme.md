The documentation (Handwritten) is available at :  [Documentation](documentation/documentation.md)
(This is documenting the very first version of the neural network library created from scratch using only numpy)


## How to navigate the repository
- `neural_network/` : Contains the implementation of the neural network from scratch using only numpy with scalars this is extremely slow but this works for small tasks such as task 1 in the beginners track.
- `neural_network_vectorized/` : Contains the implementation of the neural network from scratch using only numpy with vectorization this is much faster and can be used for larger tasks such as task 1 in intermediate track.
- `neural_network_vectorized_multiple_models/` : Is an extension of the previous folder which has two different implementations including the ones the previous folder has, which used sigmoid activation in all layers and squared error loss function for cost calculation. This adds another model which uses relu activation in hidden layers, softmax activation in output layer and cross entropy loss function for cost calculation. This is more suitable for classification tasks such as task 1 in intermediate track.

- Every task must be ran from it's respective folder to avoid import errors.

- `task1_beginner` : Contains the code for task 1 in beginner's track use `train.py` to train the model and `use.py` to use the trained model for predictions.
- `task1_intermediate_using_my_lib` : Contains the code for task 1 in intermediate track using the custom neural network library created in `neural_network/` and uses scalars this is extremely slow and this is just an attempt not reccommended for actual use.
- `task1_intermediate_using_my_lib_vectorized` : Contains the code for task 1 in intermediate track using the custom neural network library created in `neural_network_vectorized/` and uses vectorization this is much faster and can be used for actual use. there are two training file available one trains using the mnist dataset as it is and the other one shifts and adds noise to the dataset to artificially increase the size of the dataset and improve generalization. the config file can be used to change training parameters. and `use.py` to use the trained model for predictions. The config file usage is documented in the readme present in the folder. link : [readme](task_1_intermediate_using_my_lib_vectorized/readme.md)



### Requirements
- numpy
- pyyaml
- python-mnist
### Installation
```bash
pip install -r requirements.txt
```
### Usage
- Navigate to the respective task folder.
- Use `train.py` to train the model.
- Use `use.py` to use the trained model for predictions.



# Todo
- ~saving a model~
- ~Uploading of training data in a suitable format~
- ~Customizing training parameters~
- ~Documenting backpropagation algorithm used~
- ~implementing different activation functions~
- ~implementing different loss functions~
- ~implementing different optimization algorithms~
- ~implementing regularization techniques~
- ~implementing validation during training~


Note : There are some more features that im working on but due to the deadline i couldnt complete them in time, will add them later.