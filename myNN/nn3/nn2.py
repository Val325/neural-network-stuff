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

path_sad_train = "dataset/training_set/training_set/cats"
path_smile_train = "dataset/training_set/training_set/dogs"

dir_list_sad_train = os.listdir(path_sad_train)
dir_list_smile_train = os.listdir(path_smile_train)

print("Files and directories in ", path_sad_train, " len: ", len(dir_list_sad_train))
print("Files and directories in ", path_smile_train, " len: ", len(dir_list_smile_train))

label_sad = [1, 0]
label_smile = [0, 1]

train_dataset = []
true_label = []

#preprocessing.normalize()
def load_image_numpy(path):
    image = Image.open(path) 
    image = image.resize(cropimagesize)
    image = ImageOps.grayscale(image)
    return preprocessing.normalize(np.array(image)).flatten()

for i in dir_list_sad_train:
    if i.lower().endswith(('.jpg')):
        dir_image = path_sad_train + "/" + i
        image = Image.open(dir_image)
        image = image.resize(cropimagesize)
        image = ImageOps.grayscale(image)
        train_dataset.append(np.array(image).flatten())
        true_label.append(label_sad)

for i in dir_list_smile_train:
    if i.lower().endswith(('.jpg')):
        dir_image = path_smile_train + "/" + i
        image = Image.open(dir_image)
        image = image.resize(cropimagesize)
        image = ImageOps.grayscale(image)
        train_dataset.append(np.array(image).flatten())
        true_label.append(label_smile)

print("len dataset: ", len(train_dataset))
print("len label: ", len(true_label))

amount = len(train_dataset) 

x_data = preprocessing.normalize(np.asarray(train_dataset))
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
        
        self.num_neuron_input = 10
        self.num_neuron_hidden = 10
        self.num_neuron_hidden_two = 10

        #scale = 1/max(1., (2+2)/2.)
        #limit = math.sqrt(3.0 * scale)
        
        #self.w1 = np.random.uniform(-limit, limit, size=(self.num_neuron, size_input))
        #print("w1: ", self.w1)
        #self.w2 = np.random.uniform(-limit, limit, size=(self.num_neuron, self.num_neuron))
        #print("w2: ", self.w2)
        #self.w3 = np.random.uniform(-limit, limit, size=(2, self.num_neuron))
        #print("w3: ", self.w3)
        
        lower_w1, upper_w1 = -(math.sqrt(6.0) / math.sqrt(size_input + self.size_input)), (math.sqrt(6.0) / math.sqrt(size_input + self.size_input))
        self.w1 = np.random.uniform(lower_w1, upper_w1, size=(self.num_neuron, size_input))
        #print("w1: ", self.w1)
        lower_w2, upper_w2 = -(math.sqrt(6.0) / math.sqrt(self.size_input + self.size_input)), (math.sqrt(6.0) / math.sqrt(self.size_input + self.size_input))
        self.w2 = np.random.uniform(lower_w2, upper_w2, size=(self.num_neuron, self.num_neuron))
        #print("w2: ", self.w2)
        lower_w3, upper_w3 = -(math.sqrt(6.0) / math.sqrt(self.size_input + 2)), (math.sqrt(6.0) / math.sqrt(self.size_input + 2))
        self.w3 = np.random.uniform(lower_w3, upper_w3, size=(2, self.num_neuron))
        #print("w3: ", self.w3)
        #self.w1 = np.random.rand(self.num_neuron, size_input)
        #self.w2 = np.array(np.random.rand(self.num_neuron, self.num_neuron), dtype=object) 
        #self.w3 = np.random.rand(2, self.num_neuron)

        #self.b1 = np.random.normal()
        #self.b2 = np.random.normal()
        #self.b3 = np.random.normal()
        self.b1 = np.random.uniform(0.0, 1.0, size=(self.num_neuron))
        #print("self.b1: ", self.b1)
        self.b2 = np.random.uniform(0.0, 1.0, size=(self.num_neuron)) 
        self.b3 = np.random.uniform(0.0, 1.0, size=(2))

    def feedforward(self, x):
        self.input_data = x
        self.input_sum = np.array([np.sum(np.dot(self.w1[index].T, x)) + self.b1[index] for index in range(len(self.w1))])
        self.input_sigmoid = sigmoid(self.input_sum)
        if debug_input_layer: 
            print("self.input_sum: ", self.input_sum)
            print("self.input_sigmoid: ", self.input_sigmoid)

        self.hidden_sum = np.array([np.sum(np.dot(self.w2[index].T, self.input_sigmoid)) + self.b2[index] for index in range(len(self.w2))])
        self.hidden_sigmoid = sigmoid(self.hidden_sum)
        if debug_hidden_layer: 
            print("self.hidden_sum: ", self.hidden_sum)
            print("self.hidden_sigmoid: ", self.hidden_sigmoid)
        
        self.output_sum = np.array([np.sum(np.dot(self.w3[index].T, self.hidden_sum)) + self.b3[index] for index in range(len(self.w3))])
        self.output_sigmoid = softmax(self.output_sum)
        if debug_output_layer: 
            print("self.output_sum: ", self.output_sum)
            print("self.output_sigmoid: ", self.output_sigmoid)
        return self.output_sigmoid  

    def backpropogation(self, x, y, error, elem):
        if debug_backpropogation:
            print("w1 len", len(self.w1))
            print("w1[0] len", len(self.w1[0]))
            print("w2 len", len(self.w2))
            print("w2[0] len", len(self.w2[0]))
            print("w3 len", len(self.w3))
            print("w3[0] len", len(self.w3[0]))
            print("w3[0]: ", self.w3[0])
            print("w3[1]: ", self.w3[1])
            print("w3[0][0]: ", self.w3[1][0])
            #print("self.input_sum len: ", len(self.input_sum))
            #print("hidden sigmoid len: ", len(self.hidden_sigmoid))
            print("self.output_sum len: ", self.output_sum.shape)
            print("self.hidden_sigmoid len: ", self.hidden_sigmoid.shape)
            print("y: ", y)
            print("x: ", x)
        #print("derivative: ", mse_derivative(y, x))
        #print("derivative[0]: ", mse_derivative(y, x)[0])
        #print("derivative[0][0]: ", mse_derivative(y, x)[0][0])
        #print("derivative[0][1]: ", mse_derivative(y, x)[0][1])
        #print("self.w3: ", self.w3)
        
        #print("self.w3[0]: ", self.w3[0])
        #print("self.w3[1]: ", self.w3[1])
        #print("np.sum(mse_derivative(y, x)[elem][0]: ", mse_derivative(y, x)[elem][0])
        #print("np.sum(mse_derivative(y, x)[elem][1]: ", mse_derivative(y, x)[elem][1])
 

        #print("before: ", self.w3[0][2])
        for i in range(len(self.w3[0])):
            #for j in range(2):
            self.w3[0][i] -= self.learn_rate * mse_derivative(y, x)[elem][0] * deriv_sigmoid(self.output_sum[0]) * self.hidden_sigmoid[i]
            self.w3[1][i] -= self.learn_rate * mse_derivative(y, x)[elem][1] * deriv_sigmoid(self.output_sum[1]) * self.hidden_sigmoid[i]
                #left = self.learn_rate * mse_derivative(y, x)[elem][0] * deriv_sigmoid(self.output_sum[0]) * self.hidden_sigmoid[i]
                #right = self.learn_rate * mse_derivative(y, x)[elem][1] * deriv_sigmoid(self.output_sum[1]) * self.hidden_sigmoid[i]
                #self.w3[j][i] = left + right
        #print("after ---")
        #print("self.w3[0]: ", self.w3[0])
        #print("self.w3[1]: ", self.w3[1])

        #print("after: ", self.w3[0][2])
        #print("len b3", self.b3.shape)
        for i in range(len(self.output_sum)): 
            self.b3[i] -= self.learn_rate * mse_derivative(y, x)[elem][i] * deriv_sigmoid(self.output_sum[i]) 

        for i in range(len(self.w2)):
            for j in range(len(self.w2[0])):
        #        left_elem = self.learn_rate * mse_derivative(y, x)[elem][0] * deriv_sigmoid(self.output_sum[0]) * self.hidden_sigmoid[i] * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i] * self.w3[0][i]
       #         right_elem = self.learn_rate * mse_derivative(y, x)[elem][1] * deriv_sigmoid(self.output_sum[1]) * self.hidden_sigmoid[i] * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i] * self.w3[1][i]
        #        self.w2[i][j] -= left_elem + right_elem 
                self.w2[i][j] -= self.learn_rate * mse_derivative(y, x)[elem][0] * deriv_sigmoid(self.output_sum[0]) * self.hidden_sigmoid[i] * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i] * self.w3[0][i]
                self.w2[i][j] -= self.learn_rate * mse_derivative(y, x)[elem][1] * deriv_sigmoid(self.output_sum[1]) * self.hidden_sigmoid[i] * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i] * self.w3[1][i]


        for i in range(len(self.hidden_sum)):
            for i in range(2):
                self.b2[i] -= self.learn_rate * mse_derivative(y, x)[elem][i] * deriv_sigmoid(self.hidden_sum[i]) 
        
        for i in range(len(self.w1)):
            for j in range(len(self.w1[0])):
                self.w1[i][j] -= self.learn_rate * mse_derivative(y, x)[elem][0] * deriv_sigmoid(self.output_sum[0]) * self.hidden_sigmoid[i] * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i] * self.w2[i][i] * self.input_data[i] 
                self.w1[i][j] -= self.learn_rate * mse_derivative(y, x)[elem][1] * deriv_sigmoid(self.output_sum[1]) * self.hidden_sigmoid[i] * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i] * self.w2[i][i] * self.input_data[i]
        #        left_elem = self.learn_rate * mse_derivative(y, x)[elem][0] * deriv_sigmoid(self.output_sum[0]) * self.hidden_sigmoid[i] * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i] * self.w2[i][i] * self.input_data[i]
        #        right_elem = self.learn_rate * mse_derivative(y, x)[elem][1] * deriv_sigmoid(self.output_sum[1]) * self.hidden_sigmoid[i] * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i] * self.w2[i][i] * self.input_data[i]
        #        self.w1[i][j] -= left_elem + right_elem 

        for i in range(len(self.input_sum)):
            for i in range(2):
                self.b1[i] -= self.learn_rate * mse_derivative(y, x)[elem][i] * deriv_sigmoid(self.input_sum[i])

    def train(self, x, y):
        all_pred = [] 
        for epoch in range(self.epoch):
            
            for pred_i in range(len(x)):
                pred_x = np.array(self.feedforward(x[pred_i]))
                all_pred.append(pred_x)
            
            error = mse_loss(np.array(all_pred), np.array(y))

            print("--------------")
            print("epoch: ", epoch)
            print("error: ", error)
            print("cat: ", network.feedforward(load_image_numpy("dataset/test_set/test_set/cats/cat.4001.jpg")))
            print("dog: ", network.feedforward(load_image_numpy("dataset/test_set/test_set/dogs/dog.4001.jpg"))) 
            #print("black: ", network.feedforward(load_image_numpy("dataset/test/black.png")))
            #print("noise: ", network.feedforward(load_image_numpy("dataset/test/noise.png"))) 
            #print("white: ", network.feedforward(load_image_numpy("dataset/test/white.png"))) 

            self.backpropogation(np.array(all_pred), y, error, len(all_pred) - 1)
            all_pred = []
            #print("len all_pred after: ", len(all_pred))



network = NeuralNetwork(100.0, 100, lenght_data)
#network.feedforward(x_data[0])
#print("true_label[0]: ", true_label[0])
network.train(x_data, true_label)


