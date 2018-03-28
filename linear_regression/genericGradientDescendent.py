# Imports:
import numpy as np
import matplotlib.pyplot as pl

# Program Start PRINT:
print("[0] : The LINEAR REGRESSION using GRADIENT DESCENDENT has started!\n")

# Program Definitions:
LABELS = 1  # Labels Amount.
TRAIN_SIZE = 0.7 # Percentage of DATA SAMPLES that will be used to train.
print("[1] : Program Definitions:")
print("      * Labels Amount:", LABELS)
print("      * Train Size:", TRAIN_SIZE*100, "%\n")

# Openning INPUT FILE:
inputFILE = open('input.txt')
inputDATA = inputFILE.readlines()
print("[2] : Input FILE was successfully oppened!\n")

# Collecting INPUT DATA dimensions:
m = len(inputDATA) # Samples Amount.
d = (np.array([line.split(',')[:] for line in inputDATA]).shape[1]) - LABELS # Features Amount.
print("[3] : Data Dimensions:")
print("      *", m, "samples.")
print("      *", d, "features.\n")

# Calculating Train Samples Size:
TRAIN_SAMPLES = int(TRAIN_SIZE*m)
print("[4] : Definition of Train Samples Size:")
print("      *", TRAIN_SAMPLES, "samples\n")

# Instanciation of DATA matrices:
X = np.array([line.split(',')[0:d] for line in inputDATA], dtype = float)
y = np.array([line.split(',')[d:(d+1)] for line in inputDATA], dtype = float)
print("[5] : Data Matrices:")
print("      * X shape:", X.shape, "rows/cols")
print("      * y shape:", y.shape, "rows/cols\n")

# Instanciation of Theta Matrix:
Theta = np.array([[0.0, 0.0]])
print("[6] : Theta Matrix Values:")
for i in range(0, len(Theta[0])):
    print("      * Theta", i, "inicialized as", Theta[0][i])
print()

# Discretizing DATA features:
DX = X / (np.max(X, 0))
print("[7] : The X matrix was discretized with:")
print("      * MAX value:", np.max(DX))
print("      * MIN value:", np.min(DX))
print()

# Inserting the 1 column in X matrix:
DX = np.insert(DX,0,1,axis=1)
print("[8] : Data Matrices after Theta 0 column:")
print("      * X shape:", DX.shape, "rows/cols")
print("      * y shape:", y.shape, "rows/cols\n")

# Separating Train and Test Data:
Xtrain = DX[:TRAIN_SAMPLES]
Xtest = DX[TRAIN_SAMPLES:]
ytrain = y[:TRAIN_SAMPLES]
ytest = y[TRAIN_SAMPLES:]
print("[9] : Train and Test Data Matrices:")
print("      * Xtrain shape:", Xtrain.shape, "rows/cols")
print("      * Xtest shape:", Xtest.shape, "rows/cols")
print("      * ytrain shape:", ytrain.shape, "rows/cols")
print("      * ytest shape:", ytest.shape, "rows/cols\n")

# Defining the Cost Function:
def costFunction(X, y, Theta):
	m = X.shape[0] # Amount of Samples in this Dataset.
	y_hat = X.dot(Theta.T) # Once Theta is a list, you have to translate the Theta matrix.
	cost = np.sum((y_hat - y)**2)
	return cost/(2*m)

# Defining the Gradient Descendent function:
def gradientDescendent(X, y, Theta, alpha, iters):
    print("[G] : Starting Gradient Descendent Function:")
    J = [] # List of all calculated costs
    m = X.shape[0] # Amount of Samples in this Dataset
    for i in range(1,iters):
        y_hat = X.dot(Theta.T)
        error = (y_hat - y)
        error = error * X
        error = np.sum(error,0)/m
        Theta = Theta -(alpha*error)
        J.append(costFunction(X, y, Theta))
        print("      * Iteracao", i, "-> Theta:", Theta, "and Cost:", J[-1])
    return Theta, J[1:]

# MAIN:
finalTheta, finalCost = gradientDescendent(Xtrain, ytrain, Theta, 0.01, 10000)
y_hat = Xtest.dot(finalTheta.T)

pl.title("Gradient Descendent")
pl.ylabel("Profit")
pl.xlabel("Feature")
pl.plot(Xtest[:,1], ytest, '*')
pl.plot(Xtest[:,1], y_hat)
pl.show()
