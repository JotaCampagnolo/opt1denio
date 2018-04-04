# Imports:
import numpy as np
import matplotlib.pyplot as pl

# Program Start PRINT:
print("[0] : The LINEAR REGRESSION using GRADIENT DESCENDENT has started!\n")

# Program Definitions:
LABELS = 1  # Labels Amount.
TRAIN_SIZE = 0.7 # Percentage of DATA SAMPLES that will be used to train.
PRINT_GD = True # Print the Gradient Descendent steps?
print("[1] : Program Definitions:")
print("      * Labels Amount:", LABELS)
print("      * Train Size:", TRAIN_SIZE*100, "%")
print("      * Print Gradient Descendent:", PRINT_GD, "%\n")

# Openning INPUT FILE:
inputFILE = open('admission_input.txt')
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

# Adding and Removing Features:
# X = np.delete(X, [0, 3, 9], axis=1)

# Recollecting INPUT DATA dimensions:
m = X.shape[0] # Samples Amount.
d = X.shape[1] # Features Amount.
print("[6] : Data New Dimensions:")
print("      *", m, "samples.")
print("      *", d, "features.\n")

# Instanciation of Theta Matrix:
Theta = np.zeros((1, d+1))
print("[7] : Theta Matrix Values:")
for i in range(0, len(Theta[0])):
    print("      * Theta", i, "inicialized as", Theta[0][i])
print()

# Discretizing DATA features:
den = np.max(X,0) - np.min(X,0)
X = X - np.min(X,0)
DX = X / den
print("[8] : The X matrix was discretized with:")
print("      * MAX value:", np.max(DX))
print("      * MIN value:", np.min(DX))
print()

# Inserting the 1 column in X matrix:
DX = np.insert(DX,0,1,axis=1)
print("[9] : Data Matrices after Theta 0 column:")
print("      * X shape:", DX.shape, "rows/cols")
print("      * y shape:", y.shape, "rows/cols\n")

# Separating Train and Test Data:
Xtrain = DX[:TRAIN_SAMPLES]
Xtest = DX[TRAIN_SAMPLES:]
ytrain = y[:TRAIN_SAMPLES]
ytest = y[TRAIN_SAMPLES:]
print("[10] : Train and Test Data Matrices:")
print("      * Xtrain shape:", Xtrain.shape, "rows/cols")
print("      * Xtest shape:", Xtest.shape, "rows/cols")
print("      * ytrain shape:", ytrain.shape, "rows/cols")
print("      * ytest shape:", ytest.shape, "rows/cols\n")

# Defining the Y_HAT function:
def get_y_hat(X, Theta):
    return X.dot(Theta.T) # Once Theta is a list, you have to translate the Theta matrix.

# Defining Signoid function:
def signoid(X, Theta):
    return np.round(1 / (1 + (np.e**(-X.dot(Theta.T)))))

# Defining the Cost Function:
def costFunction(X, y, Theta):
    m = X.shape[0] # Amount of Samples in this Dataset.
    y_hat = signoid(X, Theta)
    cost = np.sum(y * np.log(y_hat) + (1 - y) * np.log(y_hat))
    return -cost/m

# Defining the Gradient Descendent function:
def gradientDescendent(X, y, Theta, alpha, iters):
    print("[G] : Starting Gradient Descendent Function:")
    J = [] # List of all calculated costs
    m = X.shape[0] # Amount of Samples in this Dataset
    for i in range(1,iters):
        y_hat = get_y_hat(X, Theta)
        error = (y_hat - y)
        error = error * X
        error = np.sum(error,0)/m
        Theta = Theta -(alpha*error)
        J.append(costFunction(X, y, Theta))
        if PRINT_GD:
            print("      * Iteracao", i, "-> Theta:", Theta, "and Cost:", J[-1])
    print()
    return Theta, J[1:]

# Defining MSE function:
def MSE(y, y_hat):
    m = y.shape[0]
    return np.sum((y_hat - y)**2) / m

# Defining RMSE function:
def RMSE(y, y_hat):
    return np.sqrt(MSE(y, y_hat))

# MAIN is from here to the end:
finalTheta, finalCost = gradientDescendent(Xtrain, ytrain, Theta, 0.001, 10000)
y_hat_train = get_y_hat(Xtrain, finalTheta)
y_hat_test = get_y_hat(Xtest, finalTheta)

# Print the Final Theta Values:
print("[T] : Final Theta Values:")
for i in range(0, len(finalTheta[0])):
    print("      * The Theta", i, "found is", finalTheta[0][i])
print()

# Print the Final Cost:
print("[C] : Final Cost:")
print("      * Train Dataset", costFunction(Xtrain, ytrain, finalTheta))
print("      * Test Dataset", costFunction(Xtest, ytest, finalTheta))
print()

# Print the RMSE Cost:
print("[C] : RMSE Final Cost:")
print("      *", RMSE(ytest, y_hat_test))

# Ploting the Classification:
pos = (np.round(y_hat_train)==1).ravel()
neg = (np.round(y_hat_train)==0).ravel()
pl.plot(Xtrain[pos,1], Xtrain[pos,2], 'x', color='red')
pl.plot(Xtrain[neg,1], Xtrain[neg,2], 'o', color='blue')
pl.show()
