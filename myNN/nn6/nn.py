from functions import *
import numpy as np
import math
import os
import random
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math

random_state=None
seed = None if random_state is None else int(random_state)
rng = np.random.default_rng(seed=seed)

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

class NeuralNetwork:
    def __init__(self, learning_rate, epochs, size_input, neuron_hidden, size_output):
        self.learn_rate = learning_rate
        self.epoch = epochs
        self.size_input = size_input
        self.size_output = size_output
        self.num_neuron_hidden = neuron_hidden
        print("--------------------------------------------------")
        print("Neural network: ")
        print("--------------------------------------------------")
        print("Input: ", self.size_input)
        print("Hidden: ", self.num_neuron_hidden)
        print("Output: ", self.size_output)

        #lower_w1, upper_w1 = -(math.sqrt(6.0) / math.sqrt(size_input + self.size_input)), (math.sqrt(6.0) / math.sqrt(size_input + self.size_input))
        #self.w1 = np.random.uniform(lower_w1, upper_w1, size=(self.num_neuron_hidden, size_input))
        self.w1 = np.random.randn(self.num_neuron_hidden, size_input) * 0.01
        print("w1: ", self.w1)
        #self.w2 = np.random.uniform(lower_w1, upper_w1, size=(self.size_output, self.num_neuron_hidden))
        self.w2 = np.random.randn(self.size_output, self.num_neuron_hidden) * 0.01 
        print("w2: ", self.w2)
        #self.b1 = np.random.uniform(lower_w1, upper_w1, size=(self.num_neuron_hidden))
        #self.b2 = np.random.uniform(lower_w1, upper_w1, size=(self.size_output))
        self.b1 = np.zeros((self.num_neuron_hidden))
        self.b2 = np.zeros((self.size_output))
        print("--------------------------------------------------")


    def feedforward(self, x):
        self.input_data = x
        self.z1 = np.dot(self.w1, x) + self.b1
        self.sigmoid_hidden = sigmoid(self.z1)

        self.z2 = np.dot(self.w2, self.sigmoid_hidden) + self.b2
        #print("self.z2.shape: ", self.z2.shape)
        self.sigmoid_output = sigmoid(self.z2)
        #print("self.sigmoid_output.shape: ", self.sigmoid_output.shape)
        #print("-------------------------------------------------------------------------") 
        return self.sigmoid_output 


    def backpropogation(self, x, y, i):
        delta = mse_derivative(y, x) * deriv_sigmoid(self.z2)         

        grad_w2 = delta  
        grad_b2 = delta 
        grad_w2 = np.outer(grad_w2, self.sigmoid_hidden.T) 

        self.w2 -= self.learn_rate * grad_w2 
        self.b2 -= self.learn_rate * grad_b2

        delta_input = (delta @ self.w2) * deriv_sigmoid(self.z1)
        grad_w1 = np.outer(delta_input, self.input_data.T)
        grad_b1 = delta_input 
        
        self.w1 -= self.learn_rate * grad_w1 
        self.b1 -= self.learn_rate * grad_b1

    def train(self, x, y, all_train):
        #print("all: ", all_train)
        size_data = len(x)
        all_pred = []
        batch_size = 10
        #print("all: ", all_train[:batch])
        #print("num: iter: ", round(size_data / batch))
        num_batch = round(size_data / batch_size) 
        #print("all each 1: ", all_train[num_batch * 1:batch])
        #print("all each 2: ", all_train[num_batch * 2:batch])


        for ep in range(self.epoch):
            rng.shuffle(all_train)
            for index in range(num_batch):
                stop = index + batch_size

                x_batch, y_batch = all_train[index:stop, :-1], all_train[index:stop, -1:]
                for i in range(len(x_batch)):
                    #print("x_batch: ", x_batch[i][0][0], "y_batch: ", y_batch[i][0][0])
                    pred = self.feedforward(x_batch[i][0][0])
                #print("pred: ", pred)
                #print("y: ", y[index])
                    all_pred.append(np.array(pred))
                    self.backpropogation(pred, y_batch[i][0][0], index)
                    error = mse_loss(pred, y_batch[i][0][0]) 
            


            all_pred = []
            #print("self.w2 ", self.w2)


            if ep % 10 == 0:
                print("--------------------")
                print("epoch: ", ep)
                print("error", error)
                print("sad: ", network.feedforward(load_image_numpy("dataset/test/sad/1.png")))
                print("smile: ", network.feedforward(load_image_numpy("dataset/test/smile/1.png"))) 
                print("black: ", network.feedforward(load_image_numpy("dataset/test/black.png")))
                print("noise: ", network.feedforward(load_image_numpy("dataset/test/noise.png"))) 
                print("white: ", network.feedforward(load_image_numpy("dataset/test/white.png"))) 




network = NeuralNetwork(0.01, 100, 625, 625, 2)

all_train = []
#elem = [X[149],iris_array[149]]
for i in range(len(train_dataset)):
    elem = [[train_dataset[i]],[true_label[i]]]
    all_train.append(np.array(elem, dtype=object))

network.train(train_dataset, true_label, np.array(all_train))





