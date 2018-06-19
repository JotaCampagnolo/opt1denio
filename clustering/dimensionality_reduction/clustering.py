# Imports:
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as pl

# Program Start PRINT:
print("[0] : The CLUSTERING using K-MEANS has started!\n")

# Program Definitions:
ITERS = 100 # Number of iterations of each Clusterization routine.
NEW_DIM = 2 # The new dimension.
print("[1] : Program Definitions:")
print("      * Number of Clustering Iterations:", ITERS, "\n")

# Normalize function:
def normalize(X):
    return (X - np.mean(X, 0)) / np.std(X, 0, ddof=1)

# Covariance Matrix function:
def covariance(X):
    return (X.T.dot(X)) / (X.shape[0] - 1)

# Defining the Clusters Inititialization function:
def initClusters(amount, X):
    C = np.zeros((amount, X.shape[1]))
    for i in range(amount):
        rdm = np.random.randint(X.shape[0])
        C[i] = X[rdm]
    return C

# Defining the Clusterization function:
def clusterizate(X, C, Xcentr):
    for j in range(ITERS):
        for i in range(X.shape[0]):
            di = np.sum(np.sqrt((X[i,:] - C)**2), 1)
            Xcentr[i] = np.argmin(di)
        for i in range(len(C)):
            all_points = X[Xcentr[:,0] == i]
            mean_points = np.mean(all_points, axis=0)
            C[i] = mean_points
    return C, Xcentr

# Defining the Cost function:
def costFunction(X, C):
    costs = np.zeros((len(C),1))
    for i in range(X.shape[0]):
        cost = np.sum(np.sqrt((X[i,:] - C)**2), 1)
        argmin = np.argmin(cost)
        costs[argmin] = costs[argmin] + cost[argmin]
    return np.sum(costs)

# Importing the DATA SET:
iris_ds = datasets.load_iris()
X = iris_ds.data
y = iris_ds.target

# Collecting INPUT DATA dimensions:
m = len(X) # Samples Amount.
d = X.shape[1] # Features Amount.
print("[2] : Data Dimensions:")
print("      *", m, "samples.")
print("      *", d, "features.\n")

# Normalizing the DATASET:
X = normalize(X)
print("[3] : The X matrix was normalized.\n")

# X Covariance Matrix:
covX = covariance(X)
print("[4] : The X Covariance Matrix was created.\n")

# X Singular Value Composition:
U, S, V = np.linalg.svd(covX)

# Ur and Xr Matrix:
Ur = U[:,0:NEW_DIM]
Xr = X.dot(Ur)

# Recollecting INPUT DATA dimensions:
print("[5] : Xr Matrix Dimensions:")
print("      *", Xr.shape[0], "samples.")
print("      *", Xr.shape[1], "features.\n")

# Ploting the ground truth:
pl.scatter(Xr[:,0], Xr[:,1], c=y)
pl.show()

# Clustering Routine to find the best Configuration of Centroids:
bestCost = np.inf
for i in range(100):
    # Defining the Clusters Centroids:
    C = initClusters(3, Xr)
    # Defining the Centroids Association Vector:
    Xcentr = np.zeros((Xr.shape[0], 1))
    # Clusterizing:
    C, Xcentr = clusterizate(Xr, C, Xcentr)
    if costFunction(Xr, C) < bestCost:
        bestClusters = C

# Running K-menas with the best Configuration:
Xcentr = np.zeros((Xr.shape[0], 1))
C, Xcentr = clusterizate(Xr, bestClusters, Xcentr)

# Ploting DATA after K-menas:
pl.scatter(Xr[:,0], Xr[:,1], c=Xcentr[:,0])
pl.plot(C[:,0], C[:,1], 'x', c='k')
pl.show()
