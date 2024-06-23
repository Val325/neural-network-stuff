import numpy as np
import array
from numpy import asarray

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
    #n = y_pred.shape[1] 
    #print("MSE loss : ", n)
    mse_loss = np.sum((y_true - y_pred) ** 2) / n      
    return mse_loss

def mse_derivative(y_true, y_pred): 
    N = len(y_pred)
    #N = y_pred.shape[1] 
    #print("MSE loss derive: ", y_pred.shape[1])
    #print("y_true len: ", len(y_true))
    #print("y_pred len: ", len(y_pred))

    #return -2 * (y_true - y_pred) / N
    return -2 * (y_true - y_pred) #) / N

# binary cross-entropy loss function
def binary_cross_entropy_loss(y_pred, y_true):
    m = len(y_true)
    loss = -(1/m) * np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    return loss

def binary_cross_entropy_derivative(y_pred, y_true):
    return - (y_true/y_pred) + ((1-y_true)/(1-y_pred))
#def binary_cross_entropy(y_pred, y_true):
#    N = len(y_pred)
#    return np.squeeze(-np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))/N)

#def binary_cross_entropy_derivative(y_pred, y_true):
    #print("inputX: ", inputX)
    #print("y_pred: ", y_pred)
    #print("y_true: ", y_true)

#    return y_pred * (y_pred - y_true).mean() 

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
