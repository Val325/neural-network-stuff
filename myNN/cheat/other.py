#https://medium.com/@waleedmousa975/building-a-neural-network-from-scratch-using-numpy-and-math-libraries-a-step-by-step-tutorial-in-608090c20466
import numpy as np

np.random.seed(0)

# create a toy dataset
#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y = np.array([[0], [1], [1], [0]])

X = np.array([
  [180, 120],  # Fat
  [180, 60],   # Skin
  [175, 110],   # Fat
  [170, 50], # Skin
  [170, 120],  # Fat
  [170, 60],   # Skin
  [175, 130],   # Fat
  [165, 50], # Skin
  [180, 100],  # Fat
  [180, 63],   # Skin
  [175, 130],   # Fat
  [170, 55], # Skin
  [173, 120],  # Fat
  [170, 58],   # Skin
  [177, 130],   # Fat
  [165, 54], # Skin
])

y = np.array([
  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0], 
])

# initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(0)
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

parameters = initialize_parameters(2, 3, 1)

# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# forward propagation
def forward_propagation(X, parameters):
    X = np.array(X)
    X.astype(np.float)
    # retrieve the parameters
    W1, b1, W2, b2 = parameters
    
    # compute the activation of the hidden layer
    Z1 = np.dot(W1, X.T) + b1
    A1 = sigmoid(Z1)
    
    # compute the activation of the output layer
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    
    return A2, cache

A2, cache = forward_propagation(X, parameters)

# binary cross-entropy loss function
def binary_cross_entropy_loss(A2, y):
    m = y.shape[0]
    loss = -(1/m) * np.sum(y*np.log(A2) + (1-y)*np.log(1-A2))
    return loss

# backward propagation
def backward_propagation(parameters, cache, X, y):
    m = y.shape[0]
    
    # retrieve the intermediate values
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    A2 = cache["A2"]
    
    # compute the derivative of the loss with respect to A2
    dA2 = - (y/A2) + ((1-y)/(1-A2))
    
    # compute the derivative of the activation function of the output layer
    dZ2 = dA2 * (A2 * (1-A2))
    
    # compute the derivative of the weights and biases of the output layer
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # compute the derivative of the activation function of the hidden layer
    dA1 = np.dot(parameters["W2"].T, dZ2)
    dZ1 = dA1 * (A1 * (1-A1))
    
    # compute the derivative of the weights and biases of the hidden layer
    dW1 = (1/m) * np.dot(dZ1, X)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return gradients

# update parameters
def update_parameters(parameters, gradients, learning_rate):
    # retrieve the gradients
    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]
    
    # retrieve the weights and biases
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # update the weights and biases
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

# train the neural network
def train(X, y, hidden_layer_size, num_iterations, learning_rate):
    # initialize the weights and biases
    parameters = initialize_parameters(X.shape[1], hidden_layer_size, 1)
    
    for i in range(num_iterations):
        # forward propagation
        A2, cache = forward_propagation(X, parameters)
        
        # compute the loss
        loss = binary_cross_entropy_loss(A2, y)
        
        # backward propagation
        gradients = backward_propagation(parameters, cache, X, y)
        
        # update the parameters
        parameters = update_parameters(parameters, gradients, learning_rate)
        
        if i % 1000 == 0:
            print(f"iteration {i}: loss = {loss}")
    
    return parameters

parameters = train(X, y, hidden_layer_size=4, num_iterations=10000, learning_rate=0.1)
