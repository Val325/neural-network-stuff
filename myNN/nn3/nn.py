import os
import random
from PIL import Image, ImageOps
from functions import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
np.random.seed(0)

plt.style.use('_mpl-gallery')
fig, ax = plt.subplots()

errors = []
epochsArr = []

amountData = 0
cropimagesize = (50, 50)

path_cats_train = "dataset/training_set/training_set/cats"
path_dogs_train = "dataset/training_set/training_set/dogs"

path_cats_test = "dataset/test_set/test_set/cats"
path_dogs_test = "dataset/test_set/test_set/dogs"

dir_list_cats_train = os.listdir(path_cats_train)
dir_list_dogs_train = os.listdir(path_dogs_train)
dir_list_cats_test = os.listdir(path_cats_test)
dir_list_dogs_test = os.listdir(path_dogs_test)

print("Files and directories in ", path_cats_train, " len: ", len(dir_list_cats_train))
print("Files and directories in ", path_dogs_train, " len: ", len(dir_list_dogs_train))
print("Files and directories in ", path_cats_test, " len: ", len(dir_list_cats_test))
print("Files and directories in ", path_dogs_test, " len: ", len(dir_list_dogs_test))
#image = Image.open('dataset/test_set/test_set/cats/cat.4001.jpg')
#numpydata = np.array(image)
#print("Image data: ", numpydata.shape)

# [1, 0] - cat
# [0, 1] - dogs
label_cat = [1, 0]
label_dog = [0, 1]

train_dataset = []
true_label = []

#preprocessing.normalize()
def load_image_numpy(path):
    image = Image.open(path) 
    image = image.resize(cropimagesize)
    image = ImageOps.grayscale(image)
    return np.array(image).flatten()

#add cats
for i in dir_list_cats_train:
    if i.lower().endswith(('.jpg')):
        dir_image = path_cats_train + "/" + i
        image = Image.open(dir_image) 
        image = image.resize(cropimagesize)
        image = ImageOps.grayscale(image)
        train_dataset.append(np.array(image).flatten())
        true_label.append(label_cat)

print("len dataset with cats: ", len(train_dataset))
print("len label with cats: ", len(true_label))
x_data = np.asarray(train_dataset, dtype="object")

for i in dir_list_dogs_train:
    if i.lower().endswith(('.jpg')):
        dir_image = path_dogs_train + "/" + i
        image = Image.open(dir_image)
        image = image.resize(cropimagesize)
        image = ImageOps.grayscale(image)
        train_dataset.append(np.array(image).flatten())
        true_label.append(label_dog)

print("len dataset with cats and dogs: ", len(train_dataset))
print("len label with cats and dogs: ", len(true_label))

#print("before normalize()", train_dataset)
#x_data = np.asarray(train_dataset, dtype=np.float32)
#print("after normalize()", x_data)

y_data = np.asarray(true_label, dtype=object)

print("dataset image numpy", x_data[0].shape)
size_flatten = len(x_data[0].flatten()) 
amountData = len(x_data) 

y_data = np.array(true_label) 

debug_info_feedforward = False
debug_info_backpropogate = False

class NeuralNetwork:
    def __init__(self, size_input):
        # Hyperparametrs
        #self.learn_rate = 0.35
        self.learn_rate = 0.1
        self.size_input_vec = size_input
        self.size_output_vec = 2
        # Weights
        self.w1 = np.random.randn(self.size_input_vec, self.size_output_vec)
        print("len self.w1: ", len(self.w1))
        print("shape self.w1: ", self.w1.shape)

        #self.w1 = np.random.normal() 
        #print("w1", self.w1)
        self.w2 = np.random.randn(2, 1) 
        #self.w3 = np.random.randn(2, 1)

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        #self.b3 = np.random.normal()

    def feedforward(self, x):
        print("x shape", x.shape)
        print("x len", len(x))
        #print("x[0] len", len(x[0]))
        
        self.size_output_vec = len(x)
        self.w1 = np.random.randn(self.size_input_vec, self.size_output_vec)
        #for xElem in range(len(x)):
            #print("---------------------")
            #print("height: ", x[xElem][0])
            #print("weight: ", x[xElem][1])
            #print("w1: ", self.w1)
            #print("w2: ", self.w2)
            #self.hidden_sum = self.w1[xElem][0] * x[xElem][0] + self.w1[xElem][1] * x[xElem][1] + self.b1
            #print("self.w1", self.w1)
            #print("w1 (1): ", self.w1[0])
            #print("w1 (2): ", self.w1[1])

            #print("x", x[xElem])
        #maxElemHeights = FindMaxNumpyArray(x, 0)
        #maxElemWeights = FindMaxNumpyArray(x, 1)
        #x = NormalizeArray(x, maxElemHeights, 0) 
        #x = NormalizeArray(x, maxElemWeights, 1)        
        if debug_info_feedforward:
            #print("---------------------------------------------------")
            #print("-                 ITERATION:",xElem,"                  -")
            print("---------------------------------------------------")
            print("-                 INPUT LAYER                     -")
            print("---------------------------------------------------")
            print("x: ", x)
            print("self.w1", self.w1)
            print("self.b1", self.b1)
            #print("self.w1.T", self.w1.T)
            #print("Max Heights: ", maxElemHeights)
            #print("Max Weights: ", maxElemWeights)

            #print("x[xElem]", x[xElem])
        


        self.input_sum = np.sum(np.dot(self.w1.T, x) + self.b1)
            
        if debug_info_feedforward:                
            print("input_sum: ", self.input_sum)
            
        self.input_sigmoid = sigmoid(self.input_sum)

        if debug_info_feedforward:            
            print("input: ", self.input_sigmoid)
            #print("input sigmoid.T: ", self.input_sigmoid.T)
            #print("input sigmoid[0]: ", self.input_sigmoid[0].shape)
            #print("input sigmoid[1]: ", self.input_sigmoid[1].shape)
            
            #print("input sigmoid[1][0]: ", self.input_sigmoid[1][0].shape)
            #print("input sigmoid[1][0]: ", self.input_sigmoid[1][0].shape)
            
            #print("input sigmoid[0][0]: ", self.input_sigmoid[0][0])
            #print("input sigmoid[0][0]: ", self.input_sigmoid[0][1])
            
            #print("input sigmoid[1][0]: ", self.input_sigmoid[1][0])
            #print("input sigmoid[1][0]: ", self.input_sigmoid[1][1])

            print("w1", self.w1)
            print("w2", self.w2)
            print("---------------------------------------------------")
            print("-                 HIDDEN LAYER                    -")
            print("---------------------------------------------------")
            #print("x: ", x[xElem])

        self.hidden_sum = np.sum(np.dot(self.w2, self.input_sigmoid)  + self.b2)
            
        #if debug_info_feedforward:             
        #    print("hidden_sum: ", self.hidden_sum)
            
        self.hidden_sigmoid = sigmoid(self.hidden_sum)
            
        if debug_info_feedforward: 
            print("hidden: ", self.hidden_sigmoid)
            print("---------------------------------------------------")
            print("-                 OUTPUT LAYER                    -")
            print("---------------------------------------------------")
            #print("x: ", x[xElem])
            
        #self.output_sum = np.sum(np.dot(self.w3, self.hidden_sigmoid))  + self.b3
        #if debug_info_feedforward:             
        #    print("output_sum: ", self.output_sum)
            
        #self.output_sigmoid = sigmoid(self.output_sum)
        #if debug_info_feedforward: 
        #    print("output: ", self.output_sigmoid)
        #self.hidden_sum = np.dot() + self.b2 
        #self.hidden = sigmoid(self.hidden_sum)
            #print("hidden sum: ", self.hidden_sum)
            #print("hidden: ", self.hidden)
            #print("self.w2: ", self.w2)
        #self.output_sum = np.dot(self.w2.T, self.hidden) + self.b3
        #self.output = sigmoid(self.output_sum)
            #print("output sum: ", self.output_sum)
            #print("output : ", self.output)
        #print("---------------------")
        return self.hidden_sigmoid
    
    def predict(self, x):
        #print("height: ", x[0])
        #print("weight: ", x[1])
        #self.hidden_sum = self.w1 * x[0] + self.w1[1] * x[1] + self.b1
        #self.hidden_sum = np.dot(self.w1.T, x) 
        #self.hidden = sigmoid(self.hidden_sum)
        #print("hidden sum: ", self.hidden_sum)
        #print("hidden: ", self.hidden)
        #self.output_sum = np.dot(self.w2, self.hidden)
        #self.output = sigmoid(self.output_sum)
        #print("w1 predict: ", self.w1)
        #print("w2 predict: ", self.w2)
        #print("w1[0] weight: ", self.w1[0])
        #print("w1[1] weight: ", self.w1[1])
        #print("bias: ", self.b1)
        #print("manual calc: ", self.w1[0] * x[0] + self.w1[1] * x[1] + self.b1)
        self.input_sum = np.sum(np.dot(self.w1, x) + self.b1)
        print("input_sum: ", self.input_sum)
        self.input_sigmoid = sigmoid(self.input_sum)
        print("input_sigmoid: ", self.input_sigmoid)
        self.hidden_sum = np.sum(np.dot(self.w2, self.input_sigmoid) + self.b2)
        print("hidden_sum: ", self.hidden_sum)
        print("hidden_softmax: ", softmax(self.input_sum))
        self.hidden_sigmoid = sigmoid(self.hidden_sum)
        print("hidden_sigmoid: ", self.hidden_sigmoid)
        self.output_sum = np.dot(self.w3.T, self.hidden_sigmoid + self.b3)
        print("output_sum: ", self.output_sum)
        return self.output_sum

    def predict_sigmoid(self, x):
        #print("self.w1.T ", self.w1.T)
        #print("x ", x)
        self.input_sum = np.dot(self.w1.T, x) + self.b1
        self.input_sigmoid = softmax(self.input_sum)
        return self.input_sigmoid

    def backpropogation(self, x, y, error):
         

        #derivative_loss_binarycrossentropy = -(y[index]/self.output) + ((1-y[index])/(1-self.output))
        #print("derivative_loss_binarycrossentropy: ", derivative_loss_binarycrossentropy)
        #output = derivative_loss_binarycrossentropy 
        #print("index: ", index)
        #print("error backpropogation: ", error)
        #print("x", x)
        #print("x.T", x.T)
        #print("deriv_sigmoid(self.output_sum)", deriv_sigmoid(self.output_sum))
        #print("self.output_sum", self.output_sum)
        #hiddenLayer1 = np.dot(x.T, error * deriv_sigmoid(self.output_sum))
        #hiddenLayer2 = np.dot(x.T, error * deriv_sigmoid(self.output_sum))
        #hiddenLayer3 = np.dot(x.T, error * deriv_sigmoid(self.output_sum))
        #print("hiddenLayer1: ", hiddenLayer1)
        #print("hiddenLayer2: ", hiddenLayer2)
        #print("hiddenLayer3: ", hiddenLayer3)
        #print("len(x)", len(x))
        #hiddenLayer = np.dot(x,(self.output-y).T)
        #print("hiddenLayer", hiddenLayer)        
        #db2 = np.sum(self.output-y)
        #print("db2", db2)
        #print("hiddenLayer: ", hiddenLayer)
        #print("x: ", x)
        #print("self.hidden_sum: ", error * deriv_sigmoid(self.hidden_sum))
        #inputLayer = np.dot(x.T, error * deriv_sigmoid(self.hidden_sum))
        #print("self.hidden: ", self.hidden)https://www.jeremyjordan.me/neural-networks-training/
        #print("y: ", y)
        #print("x: ", x)
        #inputLayer = np.dot(x,(self.hidden-y[index]).T) 
        #db1 = np.sum(self.hidden-y)
        #print("inputLayer: ", inputLayer)
        #self.b1
        #self.w2 -= self.learn_rate * hiddenLayer
        #self.b2 -= self.learn_rate * db2
        #print("inputLayer: ", inputLayer)
        #print("self.learn_rate: ", self.learn_rate)
        #print("self.w1: ", self.w1)
        #self.w1 -= self.learn_rate * inputLayer
        #self.b1 -= self.learn_rate * db1 
        
        #self.output_sum = np.dot(self.w1, error * deriv_sigmoid(self.output_sum)) 
        #print("self.output_sum", self.output_sum)
        
        #print("self.w1: ", self.w1)
        #print("self.learn_rate: ", self.learn_rate)
        #print("derivative_loss: ", derivative_loss)
        #print("y[index]: ", y[index])
        #print("self.output_sigmoid: ", self.output_sigmoid)
        #print("y: ", y)
        #print("input sigmoid: ", self.input_sigmoid)
        #print("predict: ", self.feedforward(self.input_sigmoid))
        derivative_loss = mse_derivative(y, x, amountData)
        #print("derivative_loss: ", derivative_loss)
        #print("derivative_loss len: ", len(derivative_loss))
        '''
        self.w2[0] -= self.learn_rate * derivative_loss[0] * self.hidden_sigmoid
        self.w2[1] -= self.learn_rate * derivative_loss[1] 
        '''
        #print("self.w2[0]: ", self.w2[0], " self.w2[0]: ", self.w2[1])
        '''
        for indexBackProg in range(size_flatten):
            print("num backprog: ", indexBackProg)
            self.w1[indexBackProg][0] -= self.learn_rate * self.w2[0] *  
            self.w1[indexBackProg][1] -= self.learn_rate * self.w2[1] * 
        '''
        #print("probability derive: ", self.feedforward(derivative_loss))
        #derivative_output = self.output_sigmoid * deriv_sigmoid(self.output_sum) 
        #derivative_output_weight = x * deriv_sigmoid(self.output_sum)
        #derivative_bias_output = deriv_sigmoid(self.output_sum)

        #derivative_hidden = self.hidden_sigmoid * deriv_sigmoid(self.hidden_sum)
        #derivative_hidden_weight = x * deriv_sigmoid(self.hidden_sum)
        #derivative_bias_hidden = deriv_sigmoid(self.hidden_sum)

        #derivative_input = x * deriv_sigmoid(self.input_sum)
        #derivative_input_weight = x * deriv_sigmoid(self.input_sum)
        #derivative_bias_input = deriv_sigmoid(self.input_sum)
        """ 
        self.w1 -= self.learn_rate * derivative_input * derivative_input_weight * derivative_loss
        self.b1 -= self.learn_rate * derivative_loss * derivative_bias_output

        self.w2 -= self.learn_rate * derivative_hidden * derivative_hidden_weight * derivative_loss
        self.b2 -= self.learn_rate * derivative_loss * derivative_bias_hidden
        
        self.w3 -= self.learn_rate * derivative_output * derivative_output_weight * derivative_loss
        self.b3 -= self.learn_rate * derivative_loss * derivative_bias_output 
        """

        #
        #if y[index][0] == 0 and y[index][1] == 1: 
            #print("self.w1[0] ", self.w1)
            #print("derivative_loss[index] ", derivative_loss[index])
        #    self.w1[0] -= self.learn_rate * derivative_loss[index][0]
            
        
        #print("y ", y[index])

        #print("y ", y[index][0])
        #print("self.w1[0] ", self.w1[0])
        #print("index: ", index)
        #print("self.w1 error", derivative_loss[index]) 
        
        #if y[index][0] == 1 and y[index][1] == 0:
            #print("self.w1[1] ", self.w1)
            #print("derivative_loss[index] ", derivative_loss[index])
        #    self.w1[1] -= self.learn_rate * derivative_loss[index][1]

        """ 
        print("self.w1 :", self.w1)
        print("self.w1[0] :", self.w1[0])
        print("self.w1[1] :", self.w1[1])

        print("y[index] :", y[index])
        print("y[index][0] :", y[index][0])
        print("y[index][1] :", y[index][1])

        print("derivative_loss[index]", derivative_loss[index])
        print("derivative_loss[index][0]", derivative_loss[index][0])
        print("derivative_loss[index][1]", derivative_loss[index][1])
        
        print("derivative_loss[index] :", derivative_loss[index])
        print("self.w1 :", self.w1)
        """
        # 8004 is examples train
        #print("w1 shape", self.w1.shape)
        #print("w1[0] shape", self.w1[0].shape)
        #print("w1[1] shape", self.w1[1].shape)
        #print("w1[3] shape", self.w1[3].shape)
        #print("w1[4] shape", self.w1[4].shape)
        #print("w1[2000] shape", self.w1[2000].shape)
        #print("w1[index][0] shape", self.w1[index][0])
        #print("w1[index][1] shape", self.w1[index][1])
        #print("w1[index][2] shape", self.w1[index][2])
        #print("w1[index][3] shape", self.w1[index][3])
        
        #print("self.w1[1000][0] :", self.w1[1000][0])
        #print("self.w1[1000][1] :", self.w1[1000][1])

        #print("self.w1[30000][0] :", self.w1[30000][0])
        #print("self.w1[30000][1] :", self.w1[30000][1])


        #for oneNeuron in range(amountData):
        #print("oneNeuron: ", oneNeuron)
            #print("w1: ", self.w1.shape)
            #print("oneNeuron: ", oneNeuron) 
            #print("w1: ", self.w1[oneNeuron])
            #self.w1[oneNeuron] -= self.learn_rate * np.sum(derivative_loss[oneNeuron]) 
            #print("derivative_loss[oneNeuron]: ", derivative_loss[oneNeuron])
            #print("derivative_loss[oneNeuron][0]: ", derivative_loss[oneNeuron][0])
            #print("derivative_loss[oneNeuron][1]: ", derivative_loss[oneNeuron][1])
            #print("self.w2: ", self.w2)
            #print("self.w2.shape: ", self.w2.shape)
            #self.w2[0][0] -= self.learn_rate * derivative_loss[oneNeuron][0]
            #self.w2[0][1] -= self.learn_rate * derivative_loss[oneNeuron][1]

        #print("self.w1[oneNeuron][0] :", self.w1[oneNeuron][0])
            #print("self.w1[oneNeuron][1] :", self.w1[oneNeuron][1])
            #print("derivative_loss[oneNeuron][0] :", derivative_loss[oneNeuron][0])
            #print("derivative_loss[oneNeuron][1] :", derivative_loss[oneNeuron][1])
            #self.w1[oneNeuron][0] -= self.learn_rate * derivative_loss[oneNeuron][0] 
            #self.w1[oneNeuron][1] -= self.learn_rate * derivative_loss[oneNeuron][1]
            
            #self.w2[0][oneNeuron] -= self.learn_rate * derivative_loss[oneNeuron][0] 
            #self.w2[1][oneNeuron] -= self.learn_rate * derivative_loss[oneNeuron][1]
        
        #print("len self.w1: ", len(self.w1))
        #self.w1[0] -= self.learn_rate * derivative_loss[index][0]
        
        #print("self.w1[1]  error", self.w1[0])
        #self.b1 -= self.learn_rate * derivative_loss[index]

        #self.w2 -= self.learn_rate * derivative_loss
        #self.b2 -= self.learn_rate * derivative_loss 

        #self.w3 -= self.learn_rate * derivative_loss
        #self.b3 -= self.learn_rate * derivative_loss

        #print("w1 backprop: ", self.w1[100])
        #print("w2 backprop: ", self.w2)
        #print("w3 backprop: ", self.w3)
        #print("b1 backprop: ", self.b1)
        #print("b2 backprop: ", self.b2)
        #pass
    def update_weights():
        pass
    def train(self, x, y):
        all_error = 0
        epochs = 3
        all_pred = []
        size_batch = 20

        for epoch in range(epochs):
            #for oneElemData in range(len(x)): 
                #print("x: ", x)
                #y_pred = self.feedforward(x)
            #num_teach = random.randint(1, amountData)
            y_pred = self.feedforward(x)
                #print("y_pred: ", y_pred)
                #all_pred.append(y_pred)
                #all_pred = np.apply_along_axis(self.feedforward, 1, x)
                #print("all pred: ", y_pred)
                #print("pred: ", y_pred)
                #print("pred len: ", len(y_pred))
                #print("x: ", x)
                #print("y[oneElemData]: ", y[oneElemData])
                #print("y: ", y)
                #print("y len: ", len(y))           
                # Compute error
                #error = np.sum((y - y_pred) * deriv_sigmoid(y_pred))
                #print("y_pred = ", y_pred, " y = ", y[oneElemData])
                
                

                
                #error = mse_loss(x, y) 
                #total_error += error
                #print("error = ", error)
                #derivative_loss = mse_derivative(y[oneElemData], y_pred) 
                #error = y_pred - y
                #error = binary_cross_entropy(y_pred, y)
                #print("error: ", error)
                #error_binary_cross_entropy = binary_cross_entropy(y_pred, y)
                #print("----------------------")

                #print("Loss binary cross entropy: ", error_binary_cross_entropy)

                #print("----------------------")
            error = mse_loss(y_pred, y[num_teach:size_batch], amountData)
                #all_pred.clear()
                #print("error: ", error)
                #all_error += error
                #print("all error: ", all_error)
             
            self.backpropogation(y_pred, y, error)
            errors.append(error)
            #epochsArr.append(epoch)
            print("----------------------")
            print("Epoch: ", epoch)
            print("Loss: ", error)
            #print("----------------------")        


class NeuralNetworkUpdate:
    def __init__(self, lr, ep, size_input):
        self.learn_rate = lr
        self.epoch = ep
        self.size_input = size_input 

        self.w1 = np.random.randn(28, size_input)
        self.w2 = np.array(np.random.randn(28, 28), dtype=object) 
        self.w3 = np.random.randn(2, 28)       

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        self.input_data = x
        self.input_sum = np.array([np.sum(np.dot(self.w1[index].T, x)) + self.b1 for index in range(len(self.w1))])
        self.input_sigmoid = sigmoid(self.input_sum)
        #if debug_input_layer: 
        #    print("self.input_sum: ", self.input_sum)
        #    print("self.input_sigmoid: ", self.input_sigmoid)

        self.hidden_sum = np.array([np.sum(np.dot(self.w2[index].T, self.input_sigmoid)) + self.b2 for index in range(len(self.w2))])
        self.hidden_sigmoid = sigmoid(self.hidden_sum)
        #if debug_hidden_layer: 
        #    print("self.hidden_sum: ", self.hidden_sum)
        #    print("self.hidden_sigmoid: ", self.hidden_sigmoid)
        
        self.output_sum = np.array([np.sum(np.dot(self.w3[index].T, self.hidden_sum)) + self.b3 for index in range(len(self.w3))])
        self.output_sigmoid = softmax(self.output_sum)
        #if debug_output_layer: 
        #    print("self.output_sum: ", self.output_sum)
        #    print("self.output_sigmoid: ", self.output_sigmoid)
        return self.output_sigmoid  

    def backpropogation(self, x, y, error):
        #print("w1 len", len(self.w1))
        #print("w1[0] len", len(self.w1[0]))
        #print("w2 len", len(self.w2))
        #print("w2[0] len", len(self.w2[0]))
        #print("w3 len", len(self.w3))
        #print("w3[0] len", len(self.w3[0]))
        #print("self.input_sum len: ", len(self.input_sum))
        # mse_derivative(y, x, amount)[0][0]
        for i in range(len(self.w3[0])):
            self.w3[0][i] -= self.learn_rate * mse_derivative_u(y, x)[0][0] * deriv_sigmoid(self.output_sum[0]) * self.hidden_sigmoid[i]
            self.w3[1][i] -= self.learn_rate * mse_derivative_u(y, x)[0][1] * deriv_sigmoid(self.output_sum[1]) * self.hidden_sigmoid[i]
            #self.w3[0][i] -= self.learn_rate * mse_derivative(y, x)[0][0]
            #self.w3[1][i] -= self.learn_rate * mse_derivative(y, x)[0][1]
        
        self.b3 -= error * deriv_sigmoid(self.output_sum[0]) 

        for i in range(len(self.w2)):
            for j in range(len(self.w2[0])):
                label_1_derive = self.learn_rate * mse_derivative_u(y, x)[0][0] * deriv_sigmoid(self.output_sum[0]) * self.w3[0][i] * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i]
                label_2_derive = self.learn_rate * mse_derivative_u(y, x)[0][1] * deriv_sigmoid(self.output_sum[1]) * self.w3[1][i]  * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i]
                self.w2[i][j] -= label_1_derive + label_2_derive 
                #self.w2[i][j] -= self.learn_rate * mse_derivative(y, x)[0][0] * deriv_sigmoid(self.output_sum[0]) * self.w3[0][i] * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i] 
                #self.w2[i][j] -= self.learn_rate * mse_derivative(y, x)[0][1] * deriv_sigmoid(self.output_sum[1]) * self.w3[1][i]  * deriv_sigmoid(self.hidden_sum[i]) * self.input_sigmoid[i]
                #self.w2[i][j] -= self.learn_rate * mse_derivative(y, x)[0][0]
                #self.w2[i][j] -= self.learn_rate * mse_derivative(y, x)[0][1]
        
        for i in range(len(self.w1)):
            for j in range(len(self.w1[0])):
                #for k in range(len(self.w2)):
                label_1_derive = self.learn_rate * mse_derivative_u(y, x)[0][0] * deriv_sigmoid(self.hidden_sum[i]) * self.w2[0][i] * self.input_data[j]
                label_2_derive = self.learn_rate * mse_derivative_u(y, x)[0][1] * deriv_sigmoid(self.hidden_sum[i]) * self.w2[1][i] * self.input_data[j] 
                self.w1[i][j] = label_1_derive + label_2_derive 
                #self.w1[i][j] -= self.learn_rate * mse_derivative(y, x)[0][0] * deriv_sigmoid(self.hidden_sum[i]) * self.w2[0][i] * self.input_data[j]
                #self.w1[i][j] -= self.learn_rate * mse_derivative(y, x)[0][1] * deriv_sigmoid(self.hidden_sum[i]) * self.w2[1][i] * self.input_data[j] 

                #self.w1[i][j] -= self.learn_rate * self.w2[1][i] * mse_derivative(y, x)[0][1] 
                #self.w1[i][j] -= self.learn_rate * mse_derivative(y, x)[0][0]
                #self.w1[i][j] -= self.learn_rate * mse_derivative(y, x)[0][1]

    def train(self, x, y):
        all_pred = [] 
        for epoch in range(self.epoch):
            
            for pred_i in range(len(x)):
                pred_x = np.array(self.feedforward(x[pred_i]))
                all_pred.append(pred_x)
            
            error = mse_loss_u(np.array(all_pred), y)

            print("--------------")
            print("epoch: ", epoch)
            print("error: ", error)
            
            self.backpropogation(np.array(all_pred), y, error)
            all_pred = []

def CheckIsScuf(x):
    if x > 0.5:
        print("^ Femboy")
    else:
        print("^ Skuf")

#network = NeuralNetwork(size_flatten)
network = NeuralNetworkUpdate(100.0, 1000, size_flatten)


# [1, 0] - cat
# [0, 1] - dogs
network.train(x_data, y_data)
print("Cat: [1, 0], Dog: [0, 1]")

print("First 10 cats")
print("Cat probability: ", network.feedforward(load_image_numpy(path_cats_test + "/cat.4001.jpg")))
print("Cat probability: ", network.feedforward(load_image_numpy(path_cats_test + "/cat.4002.jpg")))
print("Cat probability: ", network.feedforward(load_image_numpy(path_cats_test + "/cat.4003.jpg")))
print("Cat probability: ", network.feedforward(load_image_numpy(path_cats_test + "/cat.4004.jpg")))
print("Cat probability: ", network.feedforward(load_image_numpy(path_cats_test + "/cat.4005.jpg")))
print("Cat probability: ", network.feedforward(load_image_numpy(path_cats_test + "/cat.4006.jpg")))
print("Cat probability: ", network.feedforward(load_image_numpy(path_cats_test + "/cat.4007.jpg")))
print("Cat probability: ", network.feedforward(load_image_numpy(path_cats_test + "/cat.4008.jpg")))
print("Cat probability: ", network.feedforward(load_image_numpy(path_cats_test + "/cat.4009.jpg")))
print("Cat probability: ", network.feedforward(load_image_numpy(path_cats_test + "/cat.4010.jpg")))

print("First 10 dogs")
print("Dog probability: ", network.feedforward(load_image_numpy(path_dogs_test + "/dog.4001.jpg")))
print("Dog probability: ", network.feedforward(load_image_numpy(path_dogs_test + "/dog.4002.jpg")))
print("Dog probability: ", network.feedforward(load_image_numpy(path_dogs_test + "/dog.4003.jpg")))
print("Dog probability: ", network.feedforward(load_image_numpy(path_dogs_test + "/dog.4004.jpg")))
print("Dog probability: ", network.feedforward(load_image_numpy(path_dogs_test + "/dog.4005.jpg")))
print("Dog probability: ", network.feedforward(load_image_numpy(path_dogs_test + "/dog.4006.jpg")))
print("Dog probability: ", network.feedforward(load_image_numpy(path_dogs_test + "/dog.4007.jpg")))
print("Dog probability: ", network.feedforward(load_image_numpy(path_dogs_test + "/dog.4008.jpg")))
print("Dog probability: ", network.feedforward(load_image_numpy(path_dogs_test + "/dog.4009.jpg")))
print("Dog probability: ", network.feedforward(load_image_numpy(path_dogs_test + "/dog.4010.jpg")))

ax.set_xlabel('Epoch')
ax.set_ylabel('Error')
ax.plot(epochsArr, errors)

plt.show()



