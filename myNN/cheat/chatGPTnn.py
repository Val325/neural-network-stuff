import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

class NeuralNetwork:
    def __init__(self):
        # Initialize weights and biases randomly
        self.w1 = np.random.randn(2, 3)  # Weights from input to hidden layer
        self.b1 = np.zeros((1, 3))  # Bias for hidden layer
        self.w2 = np.random.randn(3, 1)  # Weights from hidden to output layer
        self.b2 = np.zeros((1, 1))  # Bias for output layer

    def feedforward(self, x):
        # Forward pass
        self.hidden_sum = np.dot(x, self.w1) + self.b1
        self.hidden_output = sigmoid(self.hidden_sum)
        self.output_sum = np.dot(self.hidden_output, self.w2) + self.b2
        self.output = sigmoid(self.output_sum)
        return self.output
    
    def backpropagation(self, x, y, learning_rate):
        # Backward pass
        error = y - self.output
        d_output = error * deriv_sigmoid(self.output_sum)
        d_hidden = np.dot(d_output, self.w2.T) * deriv_sigmoid(self.hidden_sum)
        
        # Update weights and biases
        self.w2 += np.dot(self.hidden_output.T, d_output) * learning_rate
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        self.w1 += np.dot(x.T, d_hidden) * learning_rate
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, x, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            for i in range(len(x)):
                # Forward pass
                output = self.feedforward(x[i:i+1])
                # Backward pass
                self.backpropagation(x[i:i+1], y[i:i+1], learning_rate)

# Define dataset
x_data = np.array([
    [180, 120], [180, 60], [175, 110], [170, 50], [170, 120],
    [170, 60], [175, 130], [165, 50], [180, 100], [180, 63],
    [175, 130], [170, 55], [173, 120], [170, 58], [177, 130],
    [165, 54]
])

y_data = np.array([[1], [0], [1], [0], [1], [0], [1], [0],
                   [1], [0], [1], [0], [1], [0], [1], [0]])

# Normalize data
x_data /= np.max(x_data)

# Initialize and train the neural network
network = NeuralNetwork()
network.train(x_data, y_data)

# Test with new data
my_data = np.array([60, 65]) / np.max(x_data)
my_data_fat = np.array([180, 140]) / np.max(x_data)
my_data_fat_unreal = np.array([800, 1000]) / np.max(x_data)

print("175 cm, 65 kg:", network.feedforward(my_data))
print("175 cm, 140 kg:", network.feedforward(my_data_fat))
print("800 cm, 1000 kg:", network.feedforward(my_data_fat_unreal))

