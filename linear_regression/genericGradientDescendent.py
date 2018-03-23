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

# Instanciation of DATA matrices:
X = np.array([line.split(',')[0:d] for line in inputDATA], dtype = float)
y = np.array([line.split(',')[d:(d+1)] for line in inputDATA], dtype = float)
print("[4] : Data Matrices:")
print("      * X shape:", X.shape, "rows/cols")
print("      * y shape:", y.shape, "rows/cols\n")

# Instanciation of Theta Matrix:
Theta = np.array([[-175.0, 1.0]])
print("[5] : Theta Matrix Values:")
for i in range(0, len(Theta[0])):
    print("      * Theta", i, "inicialized as", Theta[0][i])
print()

# Discretizing DATA features:
DX = X / (np.max(X, 0))
print("[6] : The X matrix was discretized with:")
print("      * MAX value:", np.max(X))
print("      * MIN value:", np.min(X))

# Separating Train and Test Data:
