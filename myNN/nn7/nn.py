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
    text_data = data.read().lower()

tokens = word_tokenize(text_data)[0:10]
vocabulary = list(set(tokens))

print("size tokens: ", len(tokens))

print("size vocabulary: ", len(vocabulary))
print("size vector vocabulary: ", len(vocabulary[0]))

# Generate one-hot encoded vectors for each word in the vocabulary
one_hot_encoded = []
for word in tokens:
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

    def preditc_word(self, word, index):
        index_token = 0
        #print("word: ", word)
        #print("wh2: ", self.w2)

        #for time in range(n):
        h = np.tanh(np.dot(self.w1, word[1]) + np.dot(self.wh, self.hidden[-1]) + self.b1)
        o = softmax(np.dot(self.w2, h) + self.b2)
        index_token = np.argmax(o[index])
            #print("o len: ", len(o))
            #try:
            #tokens.append(o) 
            #except IndexError:
            #    print("index error")
        return index_token

    def feedforward(self, data):
        #print("data size: ", data)
        #print("x: ", data[0][0])
        #print("y: ", data[0][1])

        self.hidden = {}
        self.sum_output = {}
        self.output = {}
        self.hidden[-1] = np.zeros((self.num_neuron_hidden, 1))
        self.loss = 0

        for time in range(len(data)):
            self.hidden[time] = np.tanh(np.dot(self.w1, data[time][0].T).astype('float64') + np.dot(self.wh, self.hidden[time-1]) + self.b1)
            self.sum_output[time] = np.dot(self.w2, self.hidden[time]) + self.b2 
            self.output[time] = softmax(self.sum_output[time])
            #print("self.output[time].shape: ", self.output[time].shape)
            #print("data[time][1]: ", data[time][1].T.shape)
            self.loss += loss_rnn(self.output[time], data[time][1].T)
        
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
        #grad_prev_h = {} #np.zeros_like(self.w2)
        grad_prev_h = 0

        for time in reversed(range(len(data))):
            derivative_error = loss_rnn_derivative_softmax(pred[time], data[time][1].T).astype('float64')  
            #print("derivative_error: ", derivative_error.shape) 
            #print("softmax: ", softmax_grad(pred[time]).shape)
            #print("hidden[time]: ", hidden[time].shape)

            self.dw2 += np.dot(derivative_error, hidden[time].T).astype('float64')
            self.db2 += derivative_error
            
            grad_h = np.dot(self.w2.T, derivative_error) + grad_prev_h
            grad_u = grad_h * (1 - hidden[time] * hidden[time])
            self.dwh += np.dot(grad_u, hidden[time-1].T) # (1 - hidden[time] * hidden[time]) * np.dot(self.w2.T, derivative_error).astype('float64') * hidden[time-1]
            
            self.db1 += grad_u #(1 - hidden[time] * hidden[time]) * self.dwh  #* np.dot(self.w2.T, derivative_error).astype('float64')
            self.dw1 += np.dot(self.wh.T, grad_u) #(1 - hidden[time] * hidden[time]) * np.dot(self.w2.T, derivative_error).astype('float64') * data[time][1].astype('float64')
            grad_prev_h = np.dot(self.wh.T, grad_u)
        
        self.b1 -= self.learn_rate * self.b1 
        self.b2 -= self.learn_rate * self.b2

        self.w2 -= self.learn_rate * self.dw2
        self.wh -= self.learn_rate * self.dwh
        self.w1 -= self.learn_rate * self.dw1

        self.hidden_generation = hidden[len(data)-1]
        for dparam in [self.dw1, self.dwh, self.dw2, self.db1, self.db2]:
            np.clip(dparam, -5, 5, dparam) 
    
    def train(self, x, y, all_train):
        for i in range(700):
            data_output, data_hidden = self.feedforward(all_train)
            print("next token 1: ", self.preditc_word(np.array(one_hot_encoded[1], dtype=object), 1))
            self.backpropogation(all_train, data_output, data_hidden)
        #pass
        #for ep in range(5):
            #for i in range(len(all_train)):
            #pred = network.feedforward(all_train)
            #print("pred: ", tokens[np.argmax(pred)])

network = NeuralNetworkRecurrent(0.05, 2, len(tokens), 10, len(tokens))


all_train = []
X = []
Y = []

#print("(one_hot_encoded[1][0]: ", one_hot_encoded[1][1])
print("tokens len: ", len(tokens))
for i in range(len(tokens)):

    #network.train(text_data[i], text_data[i+1], np.array(all_train))
    try:
        X.append(one_hot_encoded[i][1])
    #print(one_hot_encoded[i][1])
        Y.append(one_hot_encoded[i+1][1])
        #print("X: ", one_hot_encoded[i][1])
        #print("Y: ", one_hot_encoded[i+1][1])
        #print("argmax one_hot_encoded[i][1]: ", np.argmax(one_hot_encoded[i][1]))
        #print("argmax one_hot_encoded[i+1][1]: ", np.argmax(one_hot_encoded[i+1][1]))

        elem = [[one_hot_encoded[i][1]],[one_hot_encoded[i+1][1]]]
        #print("elem: ", elem)
    except IndexError:
    #    break
        print("end processing")
    all_train.append(np.array(elem, dtype=object))


print("all_train size: ", len(all_train))
#try:
    #print("len(all_train) : ", len(all_train))
    #network.train(X, Y, np.array(all_train))
#except IndexError:
#    print("end learning")


size_gen = 3
text = ""
text_data_pred = None
start_word = one_hot_encoded[1][1]
two_word = one_hot_encoded[1][1]
hprev = np.zeros((100,1))
arr_gen = []
network.train(X, Y, all_train)
    #arr_gen.append()
#oneword = network.preditc_sentence(start_word, 1)
#twoword = network.preditc_sentence(oneword[0], 1) 
#threeword = network.preditc_sentence(twoword[0], 1) 

#text = tokens[np.argmax(start_word)] + " " + tokens[np.argmax(oneword)] + " " + tokens[np.argmax(twoword)] + " " + tokens[np.argmax(threeword)] 
#print("text: ", text)
#start_pred = network.feedforward(start_word)
#print("generate: ", tokens[np.argmax(start_word)], " ", np.argmax(start_pred))


#text = tokens[np.argmax(start_word)] + " " 
#for t in arr_gen:
#    text += tokens[np.argmax(t)] + " "

#print("generate: ", text)

#two_pred = network.preditc_sentence(start_pred)
#three_pred = network.preditc_sentence(two_pred)

#print("two_word: ", two_word)
#three_word = network.feedforward(two_word)
#print("three_word: ", three_word)

#text = tokens[np.argmax(start_pred)] + " " + tokens[np.argmax(two_pred)] + " " + tokens[np.argmax(three_pred)]
#text = str(np.argmax(start_pred)) + " " + str(np.argmax(two_pred)) + " " + str(np.argmax(three_pred))

#for word in range(size_gen):




