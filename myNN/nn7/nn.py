from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from functions import *
import numpy as np
import math
from sklearn.datasets import load_iris
import random
from sklearn import preprocessing

#import random
random_state=None
seed = None if random_state is None else int(random_state)
rng = np.random.default_rng(seed=seed)

with open("dataset/shakespeare.txt") as data:
    text_data = data.read()[0:3000].lower()

tokens = word_tokenize(text_data)
vocabulary = list(set(tokens))

print("size vocabulary: ", len(vocabulary))
print("size vector: ", len(vocabulary[0]))

# Generate one-hot encoded vectors for each word in the vocabulary
one_hot_encoded = []
for word in vocabulary:
    # Create a list of zeros with the length of the vocabulary
    encoding = [0] * len(tokens)
    
    # Get the index of the word in the vocabulary
    index = list(tokens).index(word)
    
    # Set the value at the index to 1 to indicate word presence
    encoding[index] = 1.0
    one_hot_encoded.append((word, encoding))

#print("one-hot encoding len: ", len(one_hot_encoded))
#print("one-hot encoding[3]: ", one_hot_encoded[3])

class NeuralNetworkRecurrent:
    def __init__(self, learning_rate, epochs, size_input, neuron_hidden, size_output):
        self.learn_rate = learning_rate
        self.epoch = epochs
        self.size_input = size_input
        self.size_output = size_output
        self.num_neuron_hidden = neuron_hidden
        self.recurcive_hidden = 0
        self.num_times = 0
        self.hidden_generation = None

        print("--------------------------------------------------")
        print("Neural network: ")
        print("--------------------------------------------------")
        print("Input: ", self.size_input)
        print("Hidden: ", self.num_neuron_hidden)
        print("Output: ", self.size_output)

        self.w1 = np.random.randn(self.num_neuron_hidden, size_input) * 0.01
        print("w1: ", self.w1)
        self.wh = np.random.randn(self.num_neuron_hidden, self.num_neuron_hidden) * 0.01
        print("self.wh: ", self.wh)

        self.w2 = np.random.randn(self.size_output, self.num_neuron_hidden) * 0.01 
        print("w2: ", self.w2)

        self.b1 = np.zeros((self.num_neuron_hidden, 1))
        self.b2 = np.zeros((self.size_output, 1))
        print("--------------------------------------------------")

    def preditc_sentence(self, word, n):
        tokens = []
        for time in range(n):
            h = np.tanh(np.dot(self.w1, word).astype('float64') + np.dot(self.wh, self.hidden[time-1]) + self.b1)
            o = softmax(np.dot(self.w2, h) + self.b2)
            tokens.append(tokens[np.argmax(o)]) 
        return tokens

    def feedforward(self, data):
        #print("data size: ", len(data))
        #print("x: ", data[0][0])
        #print("y: ", data[0][1])

        self.hidden = {}
        self.output = {}
        self.hidden[-1] = np.zeros((self.num_neuron_hidden, 1))
        self.loss = 0

        for time in range(len(data)):
            self.hidden[time] = np.tanh(np.dot(self.w1, data[time][1].T).astype('float64') + np.dot(self.wh, self.hidden[time-1]) + self.b1)
            self.output[time] = softmax(np.dot(self.w2, self.hidden[time]) + self.b2)
            #print("self.output[time].shape: ", self.output[time].shape)
            #print("data[time][1]: ", data[time][1].T.shape)
            self.loss += mse_loss(self.output[time], data[time][1].T)
        
        print("loss: ", self.loss)
        #print("generate sentense: ", tokens[np.argmax(data[time][1])], " ", tokens[np.argmax(self.output[time])])
        #print("start word: ", tokens[np.argmax(data[time][1])])
        #print("predict: ", tokens[np.argmax(self.output[time])])
        
        
        return self.output, self.hidden 
    def backpropogation(self, data, pred, hidden):
        self.dw1 = np.zeros_like(self.w1)
        self.dwh = np.zeros_like(self.wh)
        self.dw2 = np.zeros_like(self.w2)

        self.db1 = np.zeros_like(self.b1)
        self.db2 = np.zeros_like(self.b2)

        for time in reversed(range(len(data))):
            derivative_error = mse_derivative(data[time][1].T, pred[time]).astype('float64') 

            self.dw2 += np.dot(derivative_error, hidden[time].T).astype('float64')
            self.db2 += derivative_error
            
            self.db1 += (1 - hidden[time] * hidden[time]) * np.dot(self.w2.T, derivative_error).astype('float64')
            self.dwh += (1 - hidden[time] * hidden[time]) * np.dot(self.w2.T, derivative_error).astype('float64') * hidden[time-1] 
            self.dw1 += (1 - hidden[time] * hidden[time]) * np.dot(self.w2.T, derivative_error).astype('float64') * data[time][1].astype('float64') 
        
        self.b1 -= self.learn_rate * self.b1 
        self.b2 -= self.learn_rate * self.b2

        self.w2 -= self.learn_rate * self.dw2
        self.wh -= self.learn_rate * self.dwh
        self.w1 -= self.learn_rate * self.dw1

        self.hidden_generation = hidden[len(data)-1]
        for dparam in [self.dw1, self.dwh, self.dw2, self.db1, self.db2]:
            np.clip(dparam, -5, 5, dparam) 
    
    def train(self, x, y, all_train):
        #print("all: ", all_train)
        size_data = len(x)
        all_pred = []
        batch_size = 64
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
                    print("error", error) 


            all_pred = []
            #print("self.w2 ", self.w2)


            #if ep % 10 == 0:
            print("--------------------")
            print("epoch: ", ep)
            print("times: ", self.num_times)


network = NeuralNetworkRecurrent(0.1, 2, len(vocabulary), 30, len(vocabulary))


all_train = []
X = []
Y = []

#print("(one_hot_encoded[1][0]: ", one_hot_encoded[1][1])
for i in range(len(tokens)):

    #network.train(text_data[i], text_data[i+1], np.array(all_train))
    try:
        X.append(one_hot_encoded[i][1])
        Y.append(one_hot_encoded[i+1][1])
        #print("X: ", one_hot_encoded[i][1])
        #print("Y: ", one_hot_encoded[i+1][1])
        #print("argmax one_hot_encoded[i][1]: ", np.argmax(one_hot_encoded[i][1]))
        #print("argmax one_hot_encoded[i+1][1]: ", np.argmax(one_hot_encoded[i+1][1]))

        elem = [[one_hot_encoded[i][1]],[one_hot_encoded[i+1][1]]]
        #print("elem: ", elem)
    except IndexError:
        break
        print("end processing")
    all_train.append(np.array(elem, dtype=object))

#try:
    #print("len(all_train) : ", len(all_train))
    #network.train(X, Y, np.array(all_train))
#except IndexError:
#    print("end learning")


size_gen = 3
text = ""
text_data_pred = None
start_word = one_hot_encoded[0][1]
two_word = one_hot_encoded[1][1]
hprev = np.zeros((40,1))
for i in range(25):
    print("num: ", i)
    pred_output, pred_hidden = network.feedforward(all_train)
    network.backpropogation(all_train, pred_output, pred_hidden)
    start_pred = network.preditc_sentence(start_word, 7)
    print("generate: ", start_pred)

#two_pred = network.preditc_sentence(start_pred)
#three_pred = network.preditc_sentence(two_pred)

#print("two_word: ", two_word)
#three_word = network.feedforward(two_word)
#print("three_word: ", three_word)

#text = tokens[np.argmax(start_pred)] + " " + tokens[np.argmax(two_pred)] + " " + tokens[np.argmax(three_pred)]
#text = str(np.argmax(start_pred)) + " " + str(np.argmax(two_pred)) + " " + str(np.argmax(three_pred))

#for word in range(size_gen):




