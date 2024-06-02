import numpy as np
import array
from numpy import asarray

def sigmoid(x):
    #print("x: ", x)]
    x = x.astype(float)
    return 1 / (1 + np.exp(-x))

def softmax_grad(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)


def loss_rnn(y_hat, y):
    """
    Cross-entropy loss function - Calculating difference between 2 probability distributions.
    First, calculate cross-entropy loss for each time step with np.sum, which returns a numpy array
    Then, sum across individual losses of all time steps with sum() to get a scalar value.
    :param y_hat: predicted value
    :param y: expected value - true label
    :return: total loss
    """
    return np.sum(-np.sum(y[i] * np.log(y_hat[i]) for i in range(len(y))))

def loss_rnn_derivative_softmax(y_hat, y):
    return y_hat - y 

def mse_loss(y_pred, y_true):
    """
    Calculates the mean squared error (MSE) loss between predicted and true values.
    
    Args:
    - y_pred: predicted values
    - y_true: true values
    
    Returns:
    - mse_loss: mean squared error loss
    """
    n = len(y_pred)
    #print("x ",y_pred)
    #print("y ",y_true)
    #mse_loss = np.sum((y_pred - y_true) ** 2) #/ amountData
    #mse_loss = np.sum((y_true - y_pred) ** 2) #/ amountData      
    mse_loss = np.sum((y_pred - y_true) ** 2) / 2 #(2*n)
    return mse_loss

def mse_derivative(y_true, y_pred): 
    n = len(y_pred)
    #return -2 * np.sum(y_true - y_pred) / n
    return (y_pred - y_true) #/ n 

# binary cross-entropy loss function
def binary_cross_entropy_loss(y_pred, y_true):
    m = len(y_true)
    loss = -(1/m) * np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    return loss

def binary_cross_entropy_derivative(y_pred, y_true):
    return - (y_true/y_pred) + ((1-y_true)/(1-y_pred))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def softmax(xs):
    return np.exp(xs) / np.sum(np.exp(xs))

def normalization(minNum, maxNum, x):
    return (x - minNum) / (maxNum - minNum)

def FindMaxNumpyArray(arr, axis):
    largest = arr[0][axis]
    for i in range(len(arr)):
        if arr[i][axis] > largest:
            largest = arr[i][axis]

    return largest

def FindMinNumpyArray(arr, axis):
    largest = arr[0][axis]
    for i in range(len(arr)):
        if arr[i][axis] < largest:
            largest = arr[i][axis]

    return largest

def NormalizeArray(arr, MaxElem, axis):
    for i in range(len(arr)):
        arr[i][axis] = arr[i][axis] / MaxElem  
    
    return arr
