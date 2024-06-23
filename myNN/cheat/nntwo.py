# imports
import numpy as np
import matplotlib.pyplot as plt
# generate random data
np.random.seed(0)
x = 5 * np.random.rand(100,1)
y = 2 + 3 * x + np.random.randn(100,1)

#Add a column of ones
add_ones=np.ones((len(x), 1))
x_data=np.hstack((add_ones,x))
print("Shape of x_data:", x_data.shape)

def GradientDescent(X,y,theta,lr=0.01,n_iters=100):
    m = len(y)
    costs = [] 
    for _ in range(n_iters):
        y_hat = np.dot(X,theta)
        theta = theta -(1/m) * lr * (np.dot(X.T,(y_hat - y)))
        cost = (1/2*m) * np.sum(np.square(y_hat-y))
        costs.append(cost)
    return theta, costs

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.cost = np.zeros(self.n_iters)
            
    def train(self, x, y):
        self.theta =np.random.randn(x.shape[1],1)
        thetas,costs=GradientDescent(x,y,self.theta,self.lr,self.n_iters)
        self.theta=thetas
        self.cost=costs
        return self
    def predict(self, x):
        return np.dot(x, self.theta)

# Initialize the model
model = LinearRegression(lr=0.01, n_iters=1000)
# Train the data
model.train(x_data, y)
# printing thetas values
print('Thetas:' ,model.theta)
# Predict
y_predicted = model.predict(x_data)
# Plot original data points
plt.scatter(x, y, s=5,color='b')
plt.xlabel("x", fontsize=18)
plt.ylabel("y", rotation=0, fontsize=18)
# Draw predicted line
plt.plot(x, y_predicted, color='r')
_ =plt.axis([0,5.5,0,30])
plt.grid(color = 'k', linestyle = '--', linewidth = 0.2)
plt.show()
