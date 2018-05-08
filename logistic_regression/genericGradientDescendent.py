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
PLOT_GR = False # Plot the graphs of model results.
print("[1] : Program Definitions:")
print("      * Labels Amount:", LABELS)
print("      * Train Size:", TRAIN_SIZE*100, "%")
print("      * Print Gradient Descendent:", PRINT_GD)
print("      * Print Predictions Samples Comparation:", PRINT_PS)
print("      * Plot the Graphics:", PLOT_GR, "\n")

# Openning INPUT FILE:
inputFILE = open('skin.txt')
inputDATA = inputFILE.readlines()
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
y[y==2] = 0
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
    return precision, recall, accuracy, f1


# MAIN is from here to the end:
finalTheta, finalCost = gradientDescendent(Xtrain, ytrain, Theta, 0.001, 10000)
y_hat_train = get_y_hat(Xtrain, finalTheta)
y_hat_test = get_y_hat(Xtest, finalTheta)
train_precision, train_recall, train_accuracy, train_f1 = classMetrics(np.round(y_hat_train), ytrain)
test_precision, test_recall, test_accuracy, test_f1 = classMetrics(np.round(y_hat_test), ytest)
notequaltrain = (np.round(y_hat_train) != ytrain).ravel()
notequaltest = (np.round(y_hat_test) != ytest).ravel()

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

# Print the Predictions of TRAIN dataset:
print("[P] : Train Predictions:")
print("      * Prediction Accuracy: " + str(np.mean(np.round(y_hat_train)==ytrain)*100) + "%")
if PRINT_PS:
    parrays = np.column_stack((np.round(y_hat_train),ytrain))
    for i in range(0, len(parrays)):
        if notequaltrain[i]:
            wrongstr = "<-- Wrong Prediction"
        else:
            wrongstr = ""
        print("      *", i+1, "samples:", parrays[i], wrongstr)
    print()

# Print the Predictions of TEST dataset:
print("[P] : Test Predictions:")
print("      * Prediction Accuracy: " + str(np.mean(np.round(y_hat_test)==ytest)*100) + "%")
if PRINT_PS:
    parrays = np.column_stack((np.round(y_hat_test),ytest))
    for i in range(0, len(parrays)):
        if notequaltest[i]:
            wrongstr = "<-- Wrong Prediction"
        else:
            wrongstr = ""
        print("      *", i+1, "samples:", parrays[i], wrongstr)
print()

# Print zeros and ones amount:
print("[Z] : Train Zeros and Ones:")
print("      * Zeros:", np.count_nonzero(np.round(y_hat_train)==0))
print("      * Ones:", np.count_nonzero(np.round(y_hat_train)==1))
print("[Z] : Test Zeros and Ones:")
print("      * Zeros:", np.count_nonzero(np.round(y_hat_test)==0))
print("      * Ones:", np.count_nonzero(np.round(y_hat_test)==1))
print()

# Print the Classification Metrics:
print("[M] : Classification Metrics Results for Train DATASET:")
print("      * Precision:", train_precision)
print("      * Recall:", train_recall)
print("      * Accuracy:", train_accuracy)
print("      * F1-Score:", train_f1, "\n")
print("[M] : Classification Metrics Results for Test DATASET:")
print("      * Precision:", test_precision)
print("      * Recall:", test_recall)
print("      * Accuracy:", test_accuracy)
print("      * F1-Score:", test_f1, "\n")

# Ploting the Classification of TRAIN dataset:
pos = (np.round(y_hat_train)==1).ravel()
neg = (np.round(y_hat_train)==0).ravel()
pl.figure(num=1, figsize=(7,7))
pl.get_current_fig_manager().window.wm_geometry("+50+50")
pl.title("Logistic Regression with Gradient Descendent:\nPredicted Classification for TRAIN")
pl.ylabel("Grade 1")
pl.xlabel("Grade 2")
pl.plot(Xtrain[pos,1], Xtrain[pos,2], 'o', color='red')
pl.plot(Xtrain[neg,1], Xtrain[neg,2], 'o', color='green')
pl.plot(Xtrain[notequaltrain,1], Xtrain[notequaltrain,2], 'x', color='yellow')

# Ploting the Classification of TEST dataset:
pos = (np.round(y_hat_test)==1).ravel()
neg = (np.round(y_hat_test)==0).ravel()
pl.figure(num=2, figsize=(7,7))
pl.get_current_fig_manager().window.wm_geometry("+850+50")
pl.title("Logistic Regression with Gradient Descendent:\nPredicted Classification for TEST")
pl.ylabel("Grade 1")
pl.xlabel("Grade 2")
pl.plot(Xtest[pos,1], Xtest[pos,2], 'o', color='red')
pl.plot(Xtest[neg,1], Xtest[neg,2], 'o', color='green')
pl.plot(Xtest[notequaltest,1], Xtest[notequaltest,2], 'x', color='yellow')
pl.show()
