# Imports:
import numpy as np
import matplotlib.pyplot as pl
import random

# Program Start PRINT:
print("[0] : The LOGISTIC REGRESSION using GRADIENT DESCENDENT has started!\n")

# Program Definitions:
LABELS = 1  # Labels Amount.
TRAIN_SIZE = 0.7 # Percentage of DATA SAMPLES that will be used to train.
PRINT_GD = False # Print the Gradient Descendent steps?
PRINT_PS = False # Print the Predictions Samples comparation?
SHUFFLE_DATA = True # Enable to shuffle all the DATASET.
print("[1] : Program Definitions:")
print("      * Labels Amount:", LABELS)
print("      * Train Size:", TRAIN_SIZE*100, "%")
print("      * Print Gradient Descendent:", PRINT_GD)
print("      * Print Predictions Samples Comparation:", PRINT_PS)
print("      * Shuffle DATASET:", SHUFFLE_DATA, "\n")

# Openning INPUT DATA FILE:
inputFILE = open('skin.txt')
inputDATA = inputFILE.readlines()
if SHUFFLE_DATA:
    np.random.shuffle(inputDATA) # This shuffles the DATASET.
print("[2] : Input FILE was successfully oppened!")
print("      * The inputDATA was successfully shuffled.\n")

# Collecting INPUT DATA dimensions:
m = len(inputDATA) # Samples Amount.
d = (np.array([line.split('\t')[:] for line in inputDATA]).shape[1]) - LABELS # Features Amount.
print("[3] : Data Dimensions:")
print("      *", m, "samples.")
print("      *", d, "features.\n")

# Calculating Train Samples Size:
TRAIN_SAMPLES = int(TRAIN_SIZE*m)
print("[4] : Definition of Train Samples Size:")
print("      *", TRAIN_SAMPLES, "samples\n")

# Instanciation of DATA matrices:
X = np.array([line.split('\t')[0:d] for line in inputDATA], dtype = float)
y = np.array([line.split('\t')[d:(d+1)] for line in inputDATA], dtype = float)
y[y==2] = 0 # Changes Class 2 to Class 0.
print("[5] : Data Matrices:")
print("      * X shape:", X.shape, "rows/cols")
print("      * y shape:", y.shape, "rows/cols\n")

# Adding and Removing Features:
# X = np.delete(X, [0, 3, 9], axis=1)

# Recollecting INPUT DATA dimensions:
m = X.shape[0] # Samples Amount.
d = X.shape[1] # Features Amount.
print("[6] : Data New Dimensions (changes in cases we remove some feature):")
print("      *", m, "samples.")
print("      *", d, "features.\n")

# Instanciation of Theta Matrix:
Theta = np.zeros((1, d+1))
print("[7] : Theta Matrix Values:")
for i in range(0, len(Theta[0])):
    print("      * Theta", i, "inicialized as", Theta[0][i])
print()

# Normalizing DATA features:
DX = (X - np.mean(X,0)) / np.std(X,0)
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

# Definition of Confusion Matrix Metrics:
def classMetrics(y_hat, y):
    CF = [] # TP, FP, TN, FN
    y_P = set(np.where(y==1)[0])
    y_N = set(np.where(y==0)[0])
    y_hat_P = set(np.where(y_hat==1)[0])
    y_hat_N = set(np.where(y_hat==0)[0])
    CF.append(len(y_P.intersection(y_hat_P)))
    CF.append(len(y_hat_P) - CF[0])
    CF.append(len(y_N.intersection(y_hat_N)))
    CF.append(len(y_hat_N) - CF[2])
    precision = CF[0] / (CF[0]+CF[1])
    recall = CF[0] / (CF[0]+CF[2])
    accuracy = (CF[0]+CF[3]) / (CF[0]+CF[1]+CF[2]+CF[3])
    f1 = (2*recall*precision) / (recall+precision)
    return CF, precision, recall, accuracy, f1

# Definition of Print Metrics function:
def printMetrics(dataName, y, y_hat, CF, recall, precision, accuracy, f1):
    notequal = (y_hat != y).ravel() # This is the position of wrong predictions.
    print("[P] :", dataName, "Predictions:")
    print("      * Confusion Matrix:")
    print("             +----------------+----------------+")
    print("             | TP: " + "{0: >#010d}".format(int(CF[0])) + " | FP: " + "{0: >#010d}".format(int(CF[1])) + " |")
    print("             +----------------+----------------+")
    print("             | TN: " + "{0: >#010d}".format(int(CF[2])) + " | FN: " + "{0: >#010d}".format(int(CF[3])) + " |")
    print("             +----------------+----------------+")
    print("      * Recall: " + str(recall*100) + " %")
    print("      * Precision: " + str(precision*100) + " %")
    print("      * Accuracy: " + str(accuracy*100) + " %")
    print("      * F1-Score: " + str(f1*100) + " %")
    if PRINT_PS:
        parrays = np.column_stack((y_hat, y))
        for i in range(0, len(parrays)):
            if notequal[i]:
                wrongstr = "*"
            else:
                wrongstr = ""
            print("      *", i+1, "samples:", parrays[i], wrongstr)
        print()
    print()
    return

# MAIN is from here to the end:
finalTheta, finalCost = gradientDescendent(Xtrain, ytrain, Theta, 0.001, 10000)
y_hat_train = get_y_hat(Xtrain, finalTheta) # This is the TRAIN prediction real values.
y_hat_test = get_y_hat(Xtest, finalTheta) # This is the TEST prediction real values.
y_hat_train_r = np.round(y_hat_train) # This is the TRAIN prediction found classes.
y_hat_test_r = np.round(y_hat_test) # This is the TEST prediction found classes.
train_CF, train_precision, train_recall, train_accuracy, train_f1 = classMetrics(y_hat_train_r, ytrain)
test_CF, test_precision, test_recall, test_accuracy, test_f1 = classMetrics(y_hat_test_r, ytest)

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

# Print the Predictions and Metrics of model:
printMetrics("TRAIN", ytrain, y_hat_train_r, train_CF, train_recall, train_precision, train_accuracy, train_f1)
printMetrics("TRAIN", ytest, y_hat_test_r, test_CF, test_recall, test_precision, test_accuracy, test_f1)
