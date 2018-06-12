# Imports:
import numpy as np
import matplotlib.pyplot as pl

# Program Start PRINT:
print("[0] : The CLUSTERING using K-MEANS has started!\n")

# Program Definitions:
NUM_CLUST = 2 # Amount of Clusters.
print("[1] : Program Definitions:")
print("      * Number of Clusters:", NUM_CLUST, "\n")

# Openning INPUT FILE:
inputFILE = open('admission.txt')
inputDATA = inputFILE.readlines()
print("[2] : Input FILE was successfully oppened!\n")

# Collecting INPUT DATA dimensions:
m = len(inputDATA) # Samples Amount.
d = (np.array([line.split(',')[:] for line in inputDATA]).shape[1] - 1) # Features Amount.
print("[3] : Data Dimensions:")
print("      *", m, "samples.")
print("      *", d, "features.\n")

# Instanciation of DATA matrices:
X = np.array([line.split(',')[0:d] for line in inputDATA], dtype = float)
y = np.array([line.split(',')[d:(d+1)] for line in inputDATA], dtype = np.int64)
print("[4] : Data Matrices:")
print("      * X shape:", X.shape, "rows/cols")
print("      * y shape:", y.shape, "rows/cols\n")

# Adding and Removing Features:
# X = np.delete(X, [0, 3, 9], axis=1)

# Recollecting INPUT DATA dimensions:
m = X.shape[0] # Samples Amount.
d = X.shape[1] # Features Amount.
print("[5] : Data New Dimensions (changes in cases we remove some feature):")
print("      *", m, "samples.")
print("      *", d, "features.\n")

# Ploting the original DATA:
pl.scatter(X[:,0], X[:,1], c=y[:,0])
pl.show()

# Defining the Clusters Centroids:
C = np.zeros((NUM_CLUST,d))
for i in range(NUM_CLUST):
    print(i)
    rdm = np.random.randint(m)
    C[i] = X[rdm]
print(C)

# Defining the Centroids Association Vector:
centr = np.zeros((m,1))
print(centr)

# Clustering Routine:
for j in range(10):
    for i in range(m):
        di = np.sum(np.sqrt((X[i,:] - C)**2), 1)
        centr[i] = np.argmin(di)
    for i in range(NUM_CLUST):
        print(i)
        all_points = X[centr[:,0] == i]
        mean_points = np.mean(all_points, axis=0)
        C[i] = mean_points
    pl.scatter(X[:,0], X[:,1], c=centr[:,0])
    pl.plot(C[:,0], C[:,1], 'x', c='k')
    pl.show()
