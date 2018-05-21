# Imports:
import numpy as np
import matplotlib.pyplot as pl

# Program Start PRINT:
print("[0] : The CLUSTERING using K-MEANS has started!\n")

# Program Definitions:
NUM_CLUST = 2 # Amount of Clusters.
print("[1] : Program Definitions:")
print("      * Nothing Yet\n")

# Openning INPUT FILE:
inputFILE = open('admission.txt')
inputDATA = inputFILE.readlines()
print("[2] : Input FILE was successfully oppened!\n")

# Collecting INPUT DATA dimensions:
m = len(inputDATA) # Samples Amount.
d = (np.array([line.split(',')[:] for line in inputDATA]).shape[1]) # Features Amount.
print("[3] : Data Dimensions:")
print("      *", m, "samples.")
print("      *", d, "features.\n")

# Instanciation of DATA matrices:
X = np.array([line.split(',')[0:d+1] for line in inputDATA], dtype = float)
print("[4] : Data Matrices:")
print("      * X shape:", X.shape, "rows/cols\n")

# Adding and Removing Features:
# X = np.delete(X, [0, 3, 9], axis=1)

# Recollecting INPUT DATA dimensions:
m = X.shape[0] # Samples Amount.
d = X.shape[1] # Features Amount.
print("[5] : Data New Dimensions (changes in cases we remove some feature):")
print("      *", m, "samples.")
print("      *", d, "features.\n")

# Defining the Clusters Centroids:
C = np.zeros((NUM_CLUST,d))
print(C)
for i in range(NUM_CLUST):
    rdm = np.random.randint(m)
    C[i] = X[rdm]
print(C)
