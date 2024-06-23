import os
import random
from PIL import Image, ImageOps
import numpy as np
from functions import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math
np.random.seed(2)

plt.style.use('_mpl-gallery')
fig, ax = plt.subplots()

errors = []
epochsArr = []

amountData = 0
cropimagesize = (25, 25)

path_sad_train = "dataset/sad"
path_smile_train = "dataset/smile"

dir_list_sad_train = os.listdir(path_sad_train)
dir_list_smile_train = os.listdir(path_smile_train)

print("Files and directories in ", path_sad_train, " len: ", len(dir_list_sad_train))
print("Files and directories in ", path_smile_train, " len: ", len(dir_list_smile_train))

label_sad = np.array([0, 1])
label_smile = np.array([1, 0])
#label_sad = 0
#label_smile = 1

train_dataset = []
true_label = []

#preprocessing.normalize()
def load_image_numpy(path):
    image = Image.open(path) 
    image = image.resize(cropimagesize)
    image = ImageOps.grayscale(image)
    #image = (image-np.min(image))/(np.max(image)-np.min(image))
    #image = (image - np.min(image))/np.ptp(image)
    #np.interp(image, (image.min(), image.max()), (0, +1))
    #image /= (np.max(image)/255.0)
    return preprocessing.normalize(np.array(image)).flatten()

for i in dir_list_sad_train:
    if i.lower().endswith(('.png')):
        dir_image = path_sad_train + "/" + i
        image = Image.open(dir_image)
        image = image.resize(cropimagesize)
        image = ImageOps.grayscale(image)
        train_dataset.append(np.array(image).flatten())
        true_label.append(label_sad)

for i in dir_list_smile_train:
    if i.lower().endswith(('.png')):
        dir_image = path_smile_train + "/" + i
        image = Image.open(dir_image)
        image = image.resize(cropimagesize)
        image = ImageOps.grayscale(image)
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

debug_input_layer = False
debug_hidden_layer = False
debug_output_layer = False
debug_backpropogation = False

class NeuralNetwork:
    def __init__(self, lr, ep, size_input):
        self.learn_rate = lr
        self.epoch = ep
        self.size_input = size_input 
        self.num_neuron = 10
                
        lower_w1, upper_w1 = -(math.sqrt(6.0) / math.sqrt(size_input + self.size_input)), (math.sqrt(6.0) / math.sqrt(size_input + self.size_input))
        self.w1 = np.random.uniform(lower_w1, upper_w1, size=(self.num_neuron, size_input))

        lower_w2, upper_w2 = -(math.sqrt(6.0) / math.sqrt(self.size_input + self.size_input)), (math.sqrt(6.0) / math.sqrt(self.size_input + self.size_input))
        self.w2 = np.random.uniform(lower_w2, upper_w2, size=(2, self.num_neuron))

        self.b1 = np.random.uniform(0.0, 1.0, size=(self.num_neuron))
        self.b2 = np.random.uniform(0.0, 1.0, size=(self.num_neuron))

    def feedforward(self, x):
        self.input_data = x
        self.input_sum = np.array([np.sum(np.dot(self.w1[index].T, x)) + self.b1[index] for index in range(len(self.w1))])
        self.input_sigmoid = sigmoid(self.input_sum)
        if debug_input_layer: 
            print("self.input_sum: ", self.input_sum)
            print("self.input_sigmoid: ", self.input_sigmoid)

        self.output_sum = np.array([np.sum(np.dot(self.w2[index].T, self.input_sigmoid)) + self.b2[index] for index in range(len(self.w2))])
        self.output_sigmoid = sigmoid(self.output_sum)
        if debug_hidden_layer: 
            print("self.hidden_sum: ", self.output_sum)
            print("self.hidden_sigmoid: ", self.output_sigmoid)
        

        return self.output_sigmoid  

    def backpropogation(self, x, y, error, elem):
        if debug_backpropogation:
            print("w1 len", len(self.w1))
            print("w1 shape", self.w1.shape)
            print("w1[0] len", len(self.w1[0]))
            print("w2 len", len(self.w2))
            print("w2[0] len", len(self.w2[0]))
            print("w2 shape", self.w2.shape)
            #print("Input data: ", self.input_data)
            #print("self.input_sum len: ", len(self.input_sum))
            #print("hidden sigmoid len: ", len(self.hidden_sigmoid))
            print("self.output_sum len: ", self.output_sum.shape)
            print("self.output_sigmoid len: ", self.output_sigmoid.shape)
            print("y: ", y)
            print("x: ", x)
            print("delta: ", )
            #print("sad: ", load_image_numpy("dataset/test/sad/1.png"))
            #print("smile: ", load_image_numpy("dataset/test/smile/1.png")) 
            #print("black: ", load_image_numpy("dataset/test/black.png"))
            #print("noise: ", load_image_numpy("dataset/test/noise.png")) 
            #print("white: ", load_image_numpy("dataset/test/white.png"))
        for i in range(len(self.w2)):
            for j in range(len(self.w2[0])):
        #        left_elem = self.learn_rate * mse_derivative(y, x)[elem][0] * deriv_sigmoid(self.output_sum[0]) * self.hidden_sigmoid[i] * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i] * self.w3[0][i]
       #         right_elem = self.learn_rate * mse_derivative(y, x)[elem][1] * deriv_sigmoid(self.output_sum[1]) * self.hidden_sigmoid[i] * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i] * self.w3[1][i]
        #        self.w2[i][j] -= left_elem + right_elem 
                self.w2[i][j] -= self.learn_rate * mse_derivative(y, x)[elem][0] * deriv_sigmoid(self.output_sum[0]) * self.input_sigmoid[i] * deriv_sigmoid(self.output_sum[i]) 
                self.w2[i][j] -= self.learn_rate * mse_derivative(y, x)[elem][1] * deriv_sigmoid(self.output_sum[1]) * self.input_sigmoid[i] * deriv_sigmoid(self.output_sum[i]) 


        for i in range(len(self.b2)):
            for j in range(2):
                self.b2[i] -= self.learn_rate * mse_derivative(y, x)[elem][j] * deriv_sigmoid(self.output_sum[j]) 
        
        for i in range(len(self.w1)):
            for j in range(len(self.w1[0])):
                self.w1[i][j] -= self.learn_rate * mse_derivative(y, x)[elem][0] * deriv_sigmoid(self.output_sum[0]) * deriv_sigmoid(self.input_sum[i]) * self.w2[0][i] * self.input_data[i] 
                self.w1[i][j] -= self.learn_rate * mse_derivative(y, x)[elem][1] * deriv_sigmoid(self.output_sum[1]) * deriv_sigmoid(self.input_sum[i]) * self.w2[1][i] * self.input_data[i]
        #        left_elem = self.learn_rate * mse_derivative(y, x)[elem][0] * deriv_sigmoid(self.output_sum[0]) * self.hidden_sigmoid[i] * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i] * self.w2[i][i] * self.input_data[i]
        #        right_elem = self.learn_rate * mse_derivative(y, x)[elem][1] * deriv_sigmoid(self.output_sum[1]) * self.hidden_sigmoid[i] * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i] * self.w2[i][i] * self.input_data[i]
        #        self.w1[i][j] -= left_elem + right_elem 

        for i in range(len(self.b2)):
            for i in range(2):
                self.b1[i] -= self.learn_rate * mse_derivative(y, x)[elem][i] * deriv_sigmoid(self.input_sum[i]) 

    def train(self, x, y):
        all_pred = [] 
        for epoch in range(self.epoch):
            
            for pred_i in range(len(x)):
                pred_x = np.array(self.feedforward(x[pred_i]))
                all_pred.append(pred_x)
            
            error = mse_loss(np.array(all_pred), y)

            print("--------------")
            print("epoch: ", epoch)
            print("error: ", error)
            print("sad: ", network.feedforward(load_image_numpy("dataset/test/sad/1.png")))
            print("smile: ", network.feedforward(load_image_numpy("dataset/test/smile/1.png"))) 
            print("black: ", network.feedforward(load_image_numpy("dataset/test/black.png")))
            print("noise: ", network.feedforward(load_image_numpy("dataset/test/noise.png"))) 
            print("white: ", network.feedforward(load_image_numpy("dataset/test/white.png"))) 

            self.backpropogation(np.array(all_pred), y, error, len(all_pred) - 1)
            all_pred = []
            #print("len all_pred after: ", len(all_pred))



network = NeuralNetwork(2.0, 250, lenght_data)
#network.feedforward(x_data[0])
#print("true_label[0]: ", true_label[0])
network.train(x_data, true_label)


