# Imports:
import numpy as np
import matplotlib.pyplot as pl
import random

# Program Start PRINT:
print("[0] : The LOGISTIC REGRESSION using GRADIENT DESCENDENT has started!\n")

# Program Definitions:
LABELS = 1  # Labels Amount.
TRAIN_SIZE = 0.7 # Percentage of DATA SAMPLES that will be used to train.
SHUFFLE_DATA = True # Enable to shuffle all the DATASET.
print("[1] : Program Definitions:")
print("      * Labels Amount:", LABELS)
print("      * Train Size:", TRAIN_SIZE*100, "%")
print("      * Print Gradient Descendent:", PRINT_GD)
print("      * Print Predictions Samples Comparation:", PRINT_PS)
print("      * Shuffle DATASET:", SHUFFLE_DATA, "\n")

# Openning INPUT DATA FILE:
inputFILE = open('digits.txt')
inputDATA = inputFILE.readlines()
if SHUFFLE_DATA:
    np.random.shuffle(inputDATA) # This shuffles the DATASET.
print("[2] : Input FILE was successfully oppened!")
print("      * The inputDATA was successfully shuffled.\n")

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
y = np.array([line.split(',')[d:(d+1)] for line in inputDATA], dtype = np.int64)
y[y==10] = 0 # Changes Class 10 to Class 0.
Classes = np.unique(y)
print("[5] : Data Matrices:")
print("      * X shape:", X.shape, "rows/cols")
print("      * y shape:", y.shape, "rows/cols")
print("      * Unique Classes:", Classes, "\n")

# Adding and Removing Features:
# X = np.delete(X, [0, 3, 9], axis=1)

# Recollecting INPUT DATA dimensions:
m = X.shape[0] # Samples Amount.
d = X.shape[1] # Features Amount.
print("[6] : Data New Dimensions (changes in cases we remove some feature):")
print("      *", m, "samples.")
print("      *", d, "features.\n")

# Instanciation of Theta Matrix:
Theta = np.zeros((len(Classes), d+1))
print("[7] : Theta Matrix Values:")
print("      * Theta Matrix inicialized with zeros.\n")

# Normalizing DATA features:
DX = X #(X - np.mean(X,0)) / np.std(X,0)
print("[8] : The X matrix was normalized with:")
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
ytrain = y[:TRAIN_SAMPLES]
Xtest = DX[TRAIN_SAMPLES:]
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
    return 1 / (1 + (np.e**(-get_y_hat(X, Theta))))

# Defining the Cost Function:
def costFunction(X, y, Theta):
    m = X.shape[0] # Amount of Samples in this Dataset.
    cost = np.sum((-y * np.log(signoid(X, Theta))) - ((1 - y) * np.log(1 - signoid(X, Theta))))
    return cost/m

# Defining the Gradient Descendent function:
def gradientDescendent(X, y, Theta, alpha, iters, c):
    print("[G] : Starting Gradient Descendent Function for Class " + str(c) +":")
    J = [] # List of all calculated costs
    m = X.shape[0] # Amount of Samples in this Dataset
    for i in range(1,iters):
        y_hat = get_y_hat(X, Theta)
        error = (y_hat - y)
        error = error * X
        error = np.sum(error,0)/m
        Theta = Theta -(alpha*error)
        J.append(costFunction(X, y, Theta))
    print()
    return Theta

# MAIN is from here to the end:
for c in Classes:
    Theta[c] = gradientDescendent(Xtrain, (ytrain==c)*1, np.array([Theta[c]]), 0.001, 100, c)

train_y_hat = np.argmax(get_y_hat(Xtrain, Theta), axis=1)
acc = np.mean(train_y_hat == ytrain.ravel())
print(acc*100)
