import numpy as np
import matplotlib.pyplot as pl

# Openning the input file:
f = open('input2.txt')
prices = f.readlines()

# Reading the input file:
y = np.array([[i.split(',')[2] for i in prices]], dtype=float).T
X = np.array([i.split(',')[0:2] for i in prices], dtype=float)
X = X / np.max(X,0) # this resizes de values to values between 0 and 1. Scales the data set.
X = np.insert(X,0,1,axis=1) # setting the bias. (0 is row, 1 is column).
print("X matrix shape:", X.shape[0], "x", X.shape[1])

# Defining the Theta matrix (list):
Theta = np.array([[-175.0, 1.0, 0.0]])
print("Theta matrix shape:", Theta.shape[0], "x", Theta.shape[1])
print("Theta.T matrix shape:", Theta.T.shape[0], "x", Theta.T.shape[1])

# Getting 70% of data size:
train = int(X.shape[0]*0.7)
print("Train samples amount:", train)

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

# Defining the Gradient Descendent function:
def gradient(X, y, Theta, alpha, iters):
	J = []
	m = X.shape[0]
	for i in range(1,iters):
		y_hat = X.dot(Theta.T) # 67x1
		error = (y_hat - y) # 67x1
		error = error * X
		error = np.sum(error,0)/m
		Theta = Theta -(alpha*error)
		J.append(costFunction(X, y, Theta))
	#print(J)
	print(costFunction(Xtest, ytest, Theta))
	return Theta, J[1:]

print(ytrain)
Theta, J = gradient(Xtrain, ytrain, Theta, 0.01, 2000)
y_hat = Xtest.dot(Theta.T) # this is the prediction

pl.title("Gradient Descendent")
pl.ylabel("Profit")
pl.xlabel("Feature")
#pl.plot(Xtrain[:,1], '*') # this ignores the first column
#pl.plot(ytrain, '*', color='red')
#pl.plot(y_hat)
#pl.plot(J)
pl.plot(Xtest[:,1], ytest, '*')
pl.plot(Xtest[:,1], y_hat)
pl.show()
