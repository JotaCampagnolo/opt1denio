import numpy as np

# Openning the input file:
f = open('input.txt')
prices = f.readlines()

# Reading the input file:
y = np.array([int(i.split(',')[2]) for i in prices])
X = np.array([i.split(',')[0:2] for i in prices], dtype=int)
X = np.insert(X,0,1,axis=1) # setting the bias. (0 is row, 1 is column).

# Defining the Theta matrix (list):
Theta = np.array([[0.5, 0.3, 0.2]])

# Getting 70% of data size:
train = int(X.shape[0]*0.7)

# Separating the training and testing data:
Xtrain = X[:train]
Xtest = X[train:]
ytrain = y[:train]
ytest = y[train:]

# Defining the cost function:
def costFunction(X, y, Theta):
	m = X.shape[0] # this is the size of dataset.
	y_hat = X.dot(Theta.T) # if Theta is a list, you have to .T.
	cost = np.sum((y_hat - y)**2)
	return cost/(2*m)
	
print(costFunction(Xtrain, ytrain, Theta))
