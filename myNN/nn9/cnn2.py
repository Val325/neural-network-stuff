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

train_size_backprop = 10
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
        self.Wconv1 = np.random.randn(3, 3, 5, 5)
        self.bconv1 = np.random.randn(3, 1, 1, 1)

        self.Wconv2 = np.random.randn(3, 3, 5, 5)
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
        if X.ndim == 5:
            X_pad = np.pad(X, ((0,0),(0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
        if X.ndim == 4:         
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
        if layer_prev.ndim == 5: 
            (convN, m, n_H_prev, n_W_prev, n_C_prev) = layer_prev.shape
        if layer_prev.ndim == 4:         
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
            #print("a_prev_pad: ", a_prev_pad.ndim)
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
                             
                            if a_prev_pad.ndim == 3: 
                                a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                            if a_prev_pad.ndim == 4: 
                                a_slice_prev = a_prev_pad[:,vert_start:vert_end, horiz_start:horiz_end, :]
                            #print("a_slice_prev.shape ", a_slice_prev.)
                            # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                            Z[conv, i, h, w, c] = np.sum(np.multiply(a_slice_prev, W[conv].T) + b[conv]) 
                                        

        # Making sure your output shape is correct
        #assert(Z.shape == (m, n_H, n_W, n_C))
    
        # Save information in "cache" for the backprop
        cache = (layer_prev, W, b, hparameters)
    
        return np.array(Z), cache
    def pool_forward(self, A_prev, hparameters, mode = "max"):
        """
        Implements the forward pass of the pooling layer
    
        Arguments:
        A_prev -- Input data, numpy array of shape (convNum, m, n_H_prev, n_W_prev, n_C_prev)
        hparameters -- python dictionary containing "f" and "stride"
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
        """
        # Retrieve dimensions from the input shape
        (convNum, m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
        # Retrieve hyperparameters from "hparameters"
        f = hparameters["f"]
        stride = hparameters["stride"]
    
        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev
    
        # Initialize output matrix A
        A = np.zeros((convNum, m, n_H, n_W, n_C))              
         
        for i in range(m):                           # loop over the training examples
            for conv in range(convNum):
                for h in range(n_H):                     # loop on the vertical axis of the output volume
                    for w in range(n_W):                 # loop on the horizontal axis of the output volume
                        for c in range (n_C):            # loop over the channels of the output volume
                    
                            # Find the corners of the current "slice" (≈4 lines)
                            vert_start = h * stride
                            vert_end = vert_start + f
                            horiz_start = w * stride
                            horiz_end = horiz_start + f
                    
                            # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                            a_prev_slice = A_prev[conv,i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                            # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                            if mode == "max":
                                A[conv, i, h, w, c] = np.max(a_prev_slice)
                            elif mode == "average":
                                A[conv, i, h, w, c] = np.mean(a_prev_slice)
    
    
        # Store the input and hparameters in "cache" for pool_backward()
        cache = (A_prev, hparameters)
    
        # Making sure your output shape is correct
        #assert(A.shape == (m, n_H, n_W, n_C))
        return np.array(A), cache
    def conv_backward(self, dZ, cache):
        """
        Implement the backward propagation for a convolution function
    
        Arguments:
        dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (numConv, channels, f, f)
        cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (numConv, channels, f, f)
        db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (channels, 1, 1, 1)
        """
    
        # Retrieve information from "cache"
        (A_prev, W, b, hparameters) = cache
    
        # Retrieve dimensions from A_prev's shape
        #print("A_prev.shape ", A_prev.shape)
        if A_prev.ndim == 5: 
            (convN ,m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape 
        if A_prev.ndim == 4:         
            (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
    
        # Retrieve dimensions from W's shape
        (numConv, channels, f, f) = W.shape
        #print("(n_C, f, f, n_C_prev)")
        #print("W.shape: ", W.shape) 
    
        # Retrieve information from "hparameters"
        stride = hparameters["stride"]
        pad = hparameters["pad"]
    
        # Retrieve dimensions from dZ's shape
        (con, m, n_H, n_W, n_C) = dZ.shape
        #print("dZ.shape: ", dZ.shape)  
        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
        dW = np.zeros((numConv, channels, f, f))
        db = np.zeros((n_C, 1, 1, 1))

        # Pad A_prev and dA_prev
        A_prev_pad = self.zero_pad(A_prev, pad)
        dA_prev_pad = self.zero_pad(dA_prev, pad)


        for i in range(m):                       # loop over the training examples
            for conv in range(numConv): 
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
                            #print("vert_start: ", vert_start)
                            #print("horiz_end: ", horiz_end)
                            # Use the corners to define the slice from a_prev_pad
                            #print("a_slice.shape: ", a_prev_pad[i].shape)
                            if a_prev_pad.ndim == 3:
                                a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]
                                #print("a_slice: ", A_prev_pad.shape)
                            if a_prev_pad.ndim == 4:
                                a_slice = a_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end,:]
                            #a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] 

                            # Update gradients for the window and the filter's parameters using the code formulas given above
                            
                            #print("W.shape: ", W.shape)
                            #print("dZ.shape: ", dZ.shape)
                            #rotate(x, angle=180)
                            
                            if horiz_end <= n_W and vert_end <= n_H:
                                da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[conv].T * dZ[conv, i, h, w, c]
                                dW[conv] += rotate(np.array(a_slice, dtype=float), angle=180).T * np.array(dZ[conv, i, h, w, c], dtype=float)
                                #image_conv.shape:  (1, 48, 48, 3)
                                db[conv] += dZ[conv, i, h, w, c]
               
            # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            if pad != 0:
                dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
            else:
                dA_prev[i, :, :, :] = da_prev_pad[:, :, :]

        # Making sure your output shape is correct
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
        #print("db: ", db) 
        return np.array(dA_prev), np.array(dW), db
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
        #print("A_prev: ", A_prev.shape) 
        # Retrieve hyperparameters from "hparameters" (≈2 lines)
        stride = hparameters["stride"]
        f = hparameters["f"]
        #print("A_prev.shape pool_backward: ", A_prev.shape) 
        # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
        convNum ,m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        if dA.ndim == 5: 
           (convN, m, n_H, n_W, n_C) = dA.shape 
        if dA.ndim == 4:         
           (m, n_H, n_W, n_C) = dA.shape 
            
        # Initialize dA_prev with zeros (≈1 line)
        dA_prev = np.zeros(A_prev.shape)
    
        for i in range(m):                       # loop over the training examples
            # select training example from A_prev (≈1 line)
            a_prev = A_prev[:,i]
            for conv in range(convNum):
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
                                #print("dA: ", dA.shape) 
                                a_prev_slice = a_prev[conv, vert_start:vert_end, horiz_start:horiz_end, c]
                            
                                # Create the mask from a_prev_slice (≈1 line)
                                mask = self.create_mask_from_window(a_prev_slice)
                                # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                                if dA.ndim == 5: 
                                    dA_prev[conv, i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[conv, i, h, w, c])
                                if dA.ndim == 4:         
                                    dA_prev[conv, i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        
                            elif mode == "average":
                                # Get the value a from dA (≈1 line)
                                da = dA[i, h, w, c]
                                # Define the shape of the filter as fxf (≈1 line)
                                shape = (f, f)
                                # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                                dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += self.distribute_value(da, shape)
                         
        # Making sure your output shape is correct
        assert(dA_prev.shape == A_prev.shape)
         
        return np.array(dA_prev)

    def feedforward(self, x):
        hparameters = {
            "pad" : 0,
            "stride": 1
        }
        #print("x: ", x.shape) 
        Conv1layer1, conv1_cache = self.convulution(x, self.Wconv1, self.bconv1, hparameters)
        self.conv1_cache = conv1_cache
        #Wconv2 = np.random.randn(3, 5, 5, 3)
        #bconv2 = np.random.randn(3, 1, 1, 1)
        #Conv2layer1 = self.convulution(x, Wconv2, bconv2, hparameters)
        #print("Conv1: ", np.array(Conv1layer1).shape)
        #print("Conv2: ", np.array(Conv2layer1[0]).shape)

        #print("Conv[1][0]: ", np.array(Conv1[1][0]).shape)

        #print("Conv1[0].shape: ", np.array(Conv1, dtype=object).shape)

        hparameters = {
            "f" : 2,
            "stride": 2
        }
        pool1, pool1_cache = self.pool_forward( np.array(Conv1layer1, dtype=object), hparameters, "max")
        self.pool1_shape = pool1.shape 
        self.pool1_cacheBprop = pool1_cache 
        #print("pool1.shape: ", np.array(pool1, dtype=object).shape)
        
        hparameters = {
            "pad" : 0,
            "stride": 1
        }

        Conv2, conv2_cache = self.convulution(np.array(pool1, dtype=object), self.Wconv2, self.bconv2, hparameters)
        self.conv2_cache = conv2_cache 
        #print("Conv2.shape: ", np.array(Conv2, dtype=object).shape)
        hparameters = {
            "f" : 2,
            "stride": 2
        }
        pool2, pool2_cache = self.pool_forward( np.array(Conv2, dtype=object), hparameters, "max")
        self.pool2_shape = pool2.shape 
        self.pool2_cacheBprop = pool2_cache 
        #print("pool2.shape: ", np.array(pool2, dtype=object).shape)
        #print("Flatten: ", np.array(pool2, dtype=object).flatten().shape)
        self.input_data = x
        self.cache_shape_flat = np.array(pool2, dtype=object).flatten().shape 
        self.z1 = np.array(np.dot(self.w1, np.array(pool2, dtype=object).flatten()) + self.b1,dtype=np.float32)
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
        #print("poolbackward2.shape: ", poolbackward2.shape)
        conv2back, grad_W_conv2, grad_b_conv2 = self.conv_backward(poolbackward2, self.conv2_cache)
        #conv2back = np.array([conv2back])
        #print("conv2back: ", np.array(conv2back, dtype=object).shape)
        poolbackward1 = self.pool_backward(np.array(conv2back, dtype=object), self.pool1_cacheBprop, "max")
        #print("poolbackward1.shape: ", poolbackward1.shape)
        
        conv1back, grad_W_conv1, grad_b_conv1 = self.conv_backward(poolbackward1, self.conv1_cache)

        #Gradient descend
        self.w2 -= self.learn_rate * grad_w2.astype('float32') 
        self.b2 -= self.learn_rate * grad_b2.astype('float32')

        #Gradient descend        
        self.w1 -= self.learn_rate * grad_w1.astype('float32') 
        self.b1 -= self.learn_rate * grad_b1.astype('float32')
        
        #print("grad_b_conv1: ", grad_b_conv1.shape)
        #print("self.bconv1: ", self.bconv1.shape)
        self.Wconv2 = self.learn_rate * grad_W_conv2.astype('float32') 
        self.bconv2 = self.learn_rate * grad_b_conv2.astype('float32') 

        self.Wconv1 -= self.learn_rate * grad_W_conv1.astype('float32') 
        self.bconv1 -= self.learn_rate * grad_b_conv1.astype('float32') 



    def train(self, x, y):
        pred = []
        #pass
        for ep in range(self.epoch):
            amountlearn = 0
            for item_train in range(len(x)):
                p = self.feedforward(np.array([x[item_train]]))
                pred.append(p)
                self.backpropogation(p, y[item_train])
                amountlearn += 1
                print("train passed: ", amountlearn, end="\r")
            
            amountlearn = 0
            print("-------------------------")
            print("epoch: ", ep)
            print("loss: ", mse_loss(pred, y))
            print("cat prob: ", self.feedforward(np.array([load_image_numpy("dataset/test_set/test_set/cats/cat.4001.jpg")])))
            print("dog prob: ", self.feedforward(np.array([load_image_numpy("dataset/test_set/test_set/dogs/dog.4001.jpg")])))

            pred.clear()

#(numConv, channels, f, f)
#Wconv = np.random.randn(3, 3, 5, 5)
#(numConv, 1, 1, 1)
#bconv = np.random.randn(3, 1, 1, 1)
"""
hparameters = {
    "pad" : 0,
    "stride": 1
}
"""
#np.array([load_image_numpy("dataset/test_set/test_set/cats/cat.4001.jpg")])
#print("train_dataset len: ", len(train_dataset))
#print("true_label len: ", len(true_label))
ConvNetwork = ConvulutionNeuralNetwork(0.1, 10, 729, 10, 2)
ConvNetwork.train(x_data, y_data)
#image_conv = np.array([load_image_numpy("dataset/test_set/test_set/cats/cat.4001.jpg")])
#print("image_conv.shape: ", np.array(image_conv).shape)
#convData, convBackprop = ConvNetwork.convulution(image_conv, Wconv, bconv, hparameters)
#grad_conv, gradW, gradb = ConvNetwork.conv_backward(image_conv, convBackprop)
#print("conv.shape: ", np.array(convData)[0][0].shape)
#print("conv len: ", len(convData))
#data = Image.fromarray(np.array(convData, dtype=np.uint8)[0][0])
#data2 = Image.fromarray(np.array(convData, dtype=np.uint8)[1][0])
#data3 = Image.fromarray(np.array(convData, dtype=np.uint8)[2][0])
#data.save("convolution/conv_pic1.png") 
#data2.save("convolution/conv_pic2.png") 
#data3.save("convolution/conv_pic3.png") 

#grad Wconv (3, 3, 5, 5)
#print("convBackprop.shape: ", np.array(convBackprop[1]).shape)
#print("grad_conv_image", grad_conv.shape)
#print("grad_conv_w", gradW.shape)
