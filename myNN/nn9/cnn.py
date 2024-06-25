import numpy as np
import math
from functions import *
import math
import os
import random
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn import preprocessing
cropimagesize = (48, 48)

path_cats_train = "dataset/training_set/training_set/cats"
path_dogs_train = "dataset/training_set/training_set/dogs"

dir_list_cat_train = os.listdir(path_cats_train)
dir_list_dog_train = os.listdir(path_dogs_train)

print("Files and directories in ", path_cats_train, " len: ", len(dir_list_cat_train))
print("Files and directories in ", path_dogs_train, " len: ", len(dir_list_dog_train))

label_dog = np.array([0, 1])
label_cat = np.array([1, 0])


train_dataset = []
true_label = []

#preprocessing.normalize()
def load_image_numpy(path):
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
    return image
"""
for i in dir_list_sad_train:
    if i.lower().endswith(('.png')):
        dir_image = path_sad_train + "/" + i
        image = Image.open(dir_image)
        image = image.resize(cropimagesize)
        #image = ImageOps.grayscale(image)
        train_dataset.append(np.array(image).flatten())
        true_label.append(label_sad)

for i in dir_list_smile_train:
    if i.lower().endswith(('.png')):
        dir_image = path_smile_train + "/" + i
        image = Image.open(dir_image)
        image = image.resize(cropimagesize)
        #image = ImageOps.grayscale(image)
        train_dataset.append(np.array(image).flatten())
        true_label.append(label_smile)

print("len dataset: ", len(train_dataset))
print("len label: ", len(true_label))

amount = len(train_dataset) 
# preprocessing.normalize()
x_data = np.asarray(train_dataset)
x_data = (x_data - np.min(x_data))/np.ptp(x_data)
y_data = np.asarray(true_label) 
print("x_data.shape: ", x_data.shape)
print("len(x_data[0]): ", len(x_data[0]))
lenght_data = len(x_data[0]) 
"""
class ConvulutionNeuralNetwork:
    def __init__(self, learning_rate, epochs, size_input, neuron_hidden, size_output):

        self.learn_rate = learning_rate
        self.epoch = epochs
        self.size_input = size_input
        self.size_output = size_output
        self.num_neuron_hidden = neuron_hidden

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
        W -- Weights, numpy array of shape (C, f, f, n_C_prev)
        b -- Biases, numpy array of shape (n_C, 1, 1, 1)
        hparameters -- python dictionary containing "stride" and "pad"
        
        Returns:
        Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward() function
        """
    
        # Retrieve dimensions from A_prev's shape (≈1 line)  
        (m, n_H_prev, n_W_prev, n_C_prev) = layer_prev.shape
    
        # Retrieve dimensions from W's shape (≈1 line)
        (n_C, f, f, n_C_prev) = W.shape
        # Retrieve information from "hparameters" (≈2 lines)
        stride = hparameters['stride']
        pad = hparameters['pad']  
        # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
        n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
        n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    
        # Initialize the output volume Z with zeros. (≈1 line)
        Z = np.zeros((m, n_H, n_W, n_C))

        # Create A_prev_pad by padding A_prev
        A_prev_pad = self.zero_pad(layer_prev, pad)
    
        for i in range(m):                                 # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i]                     # Select ith training example's padded activation
            for h in range(n_H):                           # loop over vertical axis of the output volume
                for w in range(n_W):                       # loop over horizontal axis of the output volume
                    for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                        Z[i, h, w, c] = np.sum(np.multiply(a_slice_prev, W[c]) + b[c]) 
                                        

        # Making sure your output shape is correct
        assert(Z.shape == (m, n_H, n_W, n_C))
    
        # Save information in "cache" for the backprop
        cache = (layer_prev, W, b, hparameters)
    
        return Z, cache
    def pool_forward(self, A_prev, hparameters, mode = "max"):
        """
        Implements the forward pass of the pooling layer
    
        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        hparameters -- python dictionary containing "f" and "stride"
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
        """
        # Retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
        # Retrieve hyperparameters from "hparameters"
        f = hparameters["f"]
        stride = hparameters["stride"]
    
        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev
    
        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))              
    
        for i in range(m):                           # loop over the training examples
            for h in range(n_H):                     # loop on the vertical axis of the output volume
                for w in range(n_W):                 # loop on the horizontal axis of the output volume
                    for c in range (n_C):            # loop over the channels of the output volume
                    
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                    
                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                        # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                        if mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
    
    
        # Store the input and hparameters in "cache" for pool_backward()
        cache = (A_prev, hparameters)
    
        # Making sure your output shape is correct
        assert(A.shape == (m, n_H, n_W, n_C))
        return A, cache
    def conv_backward(self, dZ, cache):
        """
        Implement the backward propagation for a convolution function
    
        Arguments:
        dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
        db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
        """
    
        # Retrieve information from "cache"
        (A_prev, W, b, hparameters) = cache
    
        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
        # Retrieve dimensions from W's shape
        (n_C, f, f, n_C_prev) = W.shape
    
        # Retrieve information from "hparameters"
        stride = hparameters["stride"]
        pad = hparameters["pad"]
    
        # Retrieve dimensions from dZ's shape
        (m, n_H, n_W, n_C) = dZ.shape
    
        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
        dW = np.zeros((f, f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        # Pad A_prev and dA_prev
        A_prev_pad = self.zero_pad(A_prev, pad)
        dA_prev_pad = self.zero_pad(dA_prev, pad)
    
        for i in range(m):                       # loop over the training examples
        
            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
        
            for h in range(n_H):                   # loop over vertical axis of the output volume
                for w in range(n_W):               # loop over horizontal axis of the output volume
                    for c in range(n_C):           # loop over the channels of the output volume
                    
                        # Find the corners of the current "slice"
                        vert_start = h * stride

                        vert_end = vert_start + f
                        horiz_start = w * stride

                        horiz_end = horiz_start + f
                    
                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        #print("da_prev_pad.shape: ", da_prev_pad.shape)
                        #print("W.shape: ", W.shape)
                        #print("dZ.shape: ", dZ.shape)

                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[c] * dZ[i, h, w, c]
                        dW[:,:,:,c] += np.array(a_slice, dtype=float) * np.array(dZ[i, h, w, c], dtype=float)
                        db[:,:,:,c] += dZ[i, h, w, c]
                    
            # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            if pad is not 0:
                dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
            else:
                dA_prev[i, :, :, :] = da_prev_pad[:, :, :]

        # Making sure your output shape is correct
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
        return dA_prev, dW, db
    def create_mask_from_window(self, x):
        """
        Creates a mask from an input matrix x, to identify the max entry of x.
    
        Arguments:
        x -- Array of shape (f, f)
    
        Returns:
        mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """
    
        mask = x == np.max(x)
    
        return mask
    def distribute_value(self, dz, shape):
        """
        Distributes the input value in the matrix of dimension shape
    
        Arguments:
        dz -- input scalar
        shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
        Returns:
        a -- Array of size (n_H, n_W) for which we distributed the value of dz
        """
    
        # Retrieve dimensions from shape (≈1 line)
        (n_H, n_W) = shape
    
        # Compute the value to distribute on the matrix (≈1 line)
        average = dz / (n_H * n_W)
    
        # Create a matrix where every entry is the "average" value (≈1 line)
        a = np.ones(shape) * average
        return a

    def pool_backward(self, dA, cache, mode = "max"):
        """
        Implements the backward pass of the pooling layer
    
        Arguments:
        dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
        cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
        Returns:
        dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
        """
    
    
        # Retrieve information from cache (≈1 line)
        (A_prev, hparameters) = cache
    
        # Retrieve hyperparameters from "hparameters" (≈2 lines)
        stride = hparameters["stride"]
        f = hparameters["f"]
    
        # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape
    
        # Initialize dA_prev with zeros (≈1 line)
        dA_prev = np.zeros(A_prev.shape)
    
        for i in range(m):                       # loop over the training examples
            # select training example from A_prev (≈1 line)
            a_prev = A_prev[i]
            for h in range(n_H):                   # loop on the vertical axis
                for w in range(n_W):               # loop on the horizontal axis
                    for c in range(n_C):           # loop over the channels (depth)
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h
                        vert_end = vert_start + f
                        horiz_start = w
                        horiz_end = horiz_start + f
                    
                        # Compute the backward propagation in both modes.
                        if mode == "max":
                            # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                            # Create the mask from a_prev_slice (≈1 line)
                            mask = self.create_mask_from_window(a_prev_slice)
                            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        
                        elif mode == "average":
                            # Get the value a from dA (≈1 line)
                            da = dA[i, h, w, c]
                            # Define the shape of the filter as fxf (≈1 line)
                            shape = (f, f)
                            # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += self.distribute_value(da, shape)
                         
        # Making sure your output shape is correct
        assert(dA_prev.shape == A_prev.shape)
    
        return dA_prev

    def feedforward(self, x):
        hparameters = {
            "pad" : 0,
            "stride": 1
        }
        Wconv1 = np.random.randn(3, 5, 5, 3)
        bconv1 = np.random.randn(3, 1, 1, 1)
        Conv1layer1 = self.convulution(x, Wconv1, bconv1, hparameters)

        #Wconv2 = np.random.randn(3, 5, 5, 3)
        #bconv2 = np.random.randn(3, 1, 1, 1)
        #Conv2layer1 = self.convulution(x, Wconv2, bconv2, hparameters)
        print("Conv1: ", np.array(Conv1layer1[0]).shape)
        #print("Conv2: ", np.array(Conv2layer1[0]).shape)

        #print("Conv[1][0]: ", np.array(Conv1[1][0]).shape)

        #print("Conv1[0].shape: ", np.array(Conv1, dtype=object).shape)

        hparameters = {
            "f" : 2,
            "stride": 2
        }
        pool1 = self.pool_forward( np.array(Conv1layer1[0], dtype=object), hparameters, "max")
        print("pool1.shape: ", np.array(pool1[0], dtype=object).shape)
        
        hparameters = {
            "pad" : 0,
            "stride": 1
        }
        W2 = np.random.randn(3, 5, 5, 3)
        b2 = np.random.randn(3, 1, 1, 1)
        Conv2, conv2_cache = self.convulution(np.array(pool1[0], dtype=object), W2, b2, hparameters)
        self.conv2_cache = conv2_cache 
        print("Conv2.shape: ", np.array(Conv2[0], dtype=object).shape)
        hparameters = {
            "f" : 2,
            "stride": 2
        }
        pool2, pool2_cache = self.pool_forward( np.array(Conv2, dtype=object), hparameters, "max")
        self.pool2_shape = pool2.shape 
        self.pool2_cacheBprop = pool2_cache 
        print("pool2.shape: ", np.array(pool2[0], dtype=object).shape)
        print("Flatten: ", np.array(pool2[0], dtype=object).flatten().shape)
        self.input_data = x
        self.cache_shape_flat = np.array(pool2[0], dtype=object).flatten().shape 
        self.z1 = np.array(np.dot(self.w1, np.array(pool2[0], dtype=object).flatten()) + self.b1,dtype=np.float32)
        self.sigmoid_hidden = sigmoid(self.z1)
        self.z2 = np.dot(self.w2, self.sigmoid_hidden) + self.b2
        self.sigmoid_output = softmax(self.z2)
        return self.sigmoid_output  
    def backpropogation(self, x, y):
        delta = softmax_gradient(self.z2) @ mse_derivative(y, x)
        
        grad_w2 = delta  
        grad_b2 = delta 
        grad_w2 = np.outer(grad_w2, self.sigmoid_hidden.T) 
        
        delta_input = (delta @ self.w2) * deriv_sigmoid(self.z1)
        grad_w1 = delta_input @ self.w1   #np.outer(delta_input, self.input_data.T)
        grad_w1_upd = self.w1.T @ delta_input 
        grad_b1 = delta_input 

        poolbackward2 = self.pool_backward(grad_w1_upd.reshape(self.pool2_shape), self.pool2_cacheBprop, "max") 
        print("poolbackward2.shape: ", poolbackward2.shape)
        conv2back = self.conv_backward(poolbackward2, self.conv2_cache)
        #conv2back = np.array([conv2back])
        print("conv2back.shape: ", np.array(conv2back, dtype=object)[0].shape)
        #Gradient descend
        self.w2 -= self.learn_rate * grad_w2 
        self.b2 -= self.learn_rate * grad_b2

        #Gradient descend        
        self.w1 -= self.learn_rate * grad_w1 
        self.b1 -= self.learn_rate * grad_b1 

    def train(self, x, y):
        self.backpropogation(x, y)
        #pass
        #for ep in range(self.epoch):
            

ConvNetwork = ConvulutionNeuralNetwork(0.01, 10, 243, 40, 2)
print("image.shape: ", np.array([load_image_numpy("dataset/training_set/training_set/cats/cat.1.jpg")]).shape)
#arrayCats = [load_image_numpy("dataset/training_set/training_set/cats/cat.1.jpg")]
#print("feed-forward", ConvNetwork.feedforward(np.array([load_image_numpy("dataset/training_set/training_set/cats/cat.1.jpg")])))
ConvNetwork.train(ConvNetwork.feedforward(np.array([load_image_numpy("dataset/training_set/training_set/cats/cat.1.jpg")])), label_cat)
#ConvNetwork.feedforward()
#np.random.seed(1)
#A_prev = np.random.randn(10, 4, 4, 3)
#W = np.random.randn(8, 2, 2, 3)
#b = np.random.randn(8, 1, 1, 1)
#hparameters = {"pad" : 2,
#               "stride": 1}
#Z, cache_conv = ConvNetwork.convulution(A_prev, W, b, hparameters) 
#print("Z.shape =", Z.shape)
#print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
#hparameters_pool = {"stride" : 1, "f": 4}
#A, cache = ConvNetwork.pool_forward(Z, hparameters_pool)
#print("mode = max")
#print("A =", A)
#print()

#A, cache = ConvNetwork.pool_forward(Z, hparameters_pool, mode = "average")
#print("mode = average")
#print("A =", A)

#dA, dW, db = ConvNetwork.conv_backward(Z, cache_conv)
#print("dA_mean =", np.mean(dA))
#print("dW_mean =", np.mean(dW))
#print("db_mean =", np.mean(db))

#x = np.random.randn(2,3)
#mask = ConvNetwork.create_mask_from_window(x)
#print('x = ', x)
#print("mask = ", mask)

#A_prev = np.random.randn(5, 5, 3, 2)
#hparameters = {"stride" : 1, "f": 2}
#A, cache = ConvNetwork.pool_forward(A_prev, hparameters)
#dA = np.random.randn(5, 4, 2, 2)

#dA_prev = ConvNetwork.pool_backward(dA, cache, "max")
#print("mode = max")
#print('mean of dA = ', np.mean(dA))
#print('dA_prev[1,1] = ', dA_prev[1,1])  
#print()
#dA_prev = ConvNetwork.pool_backward(dA, cache, "average")
#print("mode = average")
#print('mean of dA = ', np.mean(dA))
#print('dA_prev[1,1] = ', dA_prev[1,1])
