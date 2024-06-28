import numpy as np
import math
from functions import *
import math
import os
import random
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.ndimage import rotate

cropimagesize = (48, 48)

path_cats_train = "dataset/training_set/training_set/cats"
path_dogs_train = "dataset/training_set/training_set/dogs"

dir_list_cat_train = os.listdir(path_cats_train)
dir_list_dog_train = os.listdir(path_dogs_train)

print("Files and directories in ", path_cats_train, " len: ", len(dir_list_cat_train))
print("Files and directories in ", path_dogs_train, " len: ", len(dir_list_dog_train))

label_dog = np.array([0, 1])
label_cat = np.array([1, 0])

train_size = len(dir_list_cat_train) + len(dir_list_dog_train) 
process_amount = 0

train_size_backprop = 300
train_dataset = []
true_label = []
broken_image = 0

#preprocessing.normalize()
def load_image_numpy(path):
    global process_amount 
    image = Image.open(path)
    image = image.resize(cropimagesize)
    #image = ImageOps.grayscale(image)
    #image = (image-np.min(image))/(np.max(image)-np.min(image))
    #image = (image - np.min(image))/np.ptp(image)
    #np.interp(image, (image.min(), image.max()), (0, +1))
    #image /= (np.max(image)/255.0)
    #return preprocessing.normalize(np.array(image))

    x_min = np.array(image).min(axis=(1, 2), keepdims=True)
    x_max = np.array(image).max(axis=(1, 2), keepdims=True)

    image = (np.array(image) - x_min)/(x_max-x_min)
    #print("image: ", image)
    process_amount += 1
    print("loading: ",((process_amount / train_size_backprop) * 100),end="\r")
    return image

for i in dir_list_cat_train:
    if process_amount > (train_size_backprop / 2):
        break

    if i.lower().endswith(('.jpg')):
        dir_image = path_cats_train + "/" + i
        image = load_image_numpy(dir_image) #Image.open(dir_image)
        if np.isnan(image).any():
            broken_image += 1
            continue
    
        train_dataset.append(np.array(image))
        true_label.append(label_cat)

for i in dir_list_dog_train:
    if process_amount > train_size_backprop:
        break

    if i.lower().endswith(('.jpg')):
        dir_image = path_dogs_train + "/" + i 
        image = load_image_numpy(dir_image) #Image.open(dir_image)
        if np.isnan(image).any():
            broken_image += 1
            continue
        
        train_dataset.append(np.array(image))
        true_label.append(label_dog)


amount = len(train_dataset) 
x_data = np.asarray(train_dataset)
y_data = np.asarray(true_label)

class ConvulutionNeuralNetwork:
    def __init__(self, learning_rate, epochs, size_input, neuron_hidden, size_output):

        self.learn_rate = learning_rate
        self.epoch = epochs
        self.size_input = size_input
        self.size_output = size_output
        self.num_neuron_hidden = neuron_hidden
        """
           convW - (num_conv, filterX, filterY, channels) 
           convB - (num_conv, 1, 1, 1) 
        """
        self.Wconv1 = np.random.randn(3, 5, 5, 3)
        self.bconv1 = np.random.randn(3, 1, 1, 1)

        self.Wconv2 = np.random.randn(3, 5, 5, 3)
        self.bconv2 = np.random.randn(3, 1, 1, 1)

        self.w1 = np.array(np.random.randn(self.num_neuron_hidden, size_input) * 0.01, dtype=np.float64)
        self.w2 = np.array(np.random.randn(self.size_output, self.num_neuron_hidden) * 0.01, dtype=np.float64) 

        self.b1 = np.array(np.zeros((self.num_neuron_hidden)), dtype=np.float64)
        self.b2 = np.array(np.zeros((self.size_output)), dtype=np.float64)

    def zero_pad(self, X, pad):
        """
        Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
        as illustrated in Figure 1.
    
        Argument:
        X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
        pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
        Returns:
        X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """
    
        X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
        return X_pad
    def convulution(self, layer_prev, W, b, hparameters):
        """
        Implements the forward propagation for a convolution function
    
        Arguments:
        A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights, numpy array of shape (numConv, channels, f, f)
        b -- Biases, numpy array of shape (n_C, 1, 1, 1)
        hparameters -- python dictionary containing "stride" and "pad"
        
        Returns:
        Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward() function
        """
    
        # Retrieve dimensions from A_prev's shape (≈1 line)  
        (m, n_H_prev, n_W_prev, n_C_prev) = layer_prev.shape

        # Retrieve dimensions from W's shape (≈1 line)
        (numConv, channels, f, f) = W.shape
        # Retrieve information from "hparameters" (≈2 lines)
        stride = hparameters['stride']
        pad = hparameters['pad']  
        # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
        n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
        n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    
        # Initialize the output volume Z with zeros. (≈1 line)
        Z = np.zeros((numConv, m, n_H, n_W, channels))

        # Create A_prev_pad by padding A_prev
        A_prev_pad = self.zero_pad(layer_prev, pad)
    
        for i in range(m):                                 # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i]                     # Select ith training example's padded activation
            for conv in range(numConv):
                for h in range(n_H):                           # loop over vertical axis of the output volume
                    for w in range(n_W):                       # loop over horizontal axis of the output volume
                        for c in range(n_C_prev):                   # loop over channels (= #filters) of the output volume
                            # Find the corners of the current "slice" (≈4 lines)
                            vert_start = h * stride
                            vert_end = vert_start + f
                            horiz_start = w * stride
                            horiz_end = horiz_start + f
                            # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                            a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                            # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                            Z[conv, i, h, w, c] = np.sum(np.multiply(a_slice_prev, W[conv].T) + b[conv]) 
                                        

        # Making sure your output shape is correct
        #assert(Z.shape == (m, n_H, n_W, n_C))
    
        # Save information in "cache" for the backprop
        cache = (layer_prev, W, b, hparameters)
    
        return np.array(Z), cache

#(numConv, channels, f, f)
Wconv = np.random.randn(3, 3, 5, 5)
#(numConv, 1, 1, 1)
bconv = np.random.randn(3, 1, 1, 1)

hparameters = {
    "pad" : 0,
    "stride": 1
}
#np.array([load_image_numpy("dataset/test_set/test_set/cats/cat.4001.jpg")])
ConvNetwork = ConvulutionNeuralNetwork(0.1, 10, 243, 10, 2)
convData, convBackprop = ConvNetwork.convulution(np.array([load_image_numpy("dataset/test_set/test_set/cats/cat.4001.jpg")]), Wconv, bconv, hparameters)
print("conv.shape: ", np.array(convData).shape)
