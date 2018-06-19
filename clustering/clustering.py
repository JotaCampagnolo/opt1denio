# Imports:
import numpy as np
import matplotlib.pyplot as pl

# Program Start PRINT:
print("[0] : The CLUSTERING using K-MEANS has started!\n")

# Program Definitions:
ITERS = 25 # Number of iterations of each Clusterization routine.
RUN_E = False # Run the Elbow find routine.
RUN_C = True # Run the best Configuration routine.
print("[1] : Program Definitions:")
print("      * Number of Clustering Iterations:", ITERS, "\n")

# Openning INPUT FILE:
inputFILE = open('RelationNetwork.csv')
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

# Normalizing the DATASET:
X = (X - np.mean(X, 0)) / np.std(X, 0, ddof=1)
print("[6] : The X matrix was normalized.\n")

# Defining the Clusters Inititialization function:
def initClusters(amount):
    C = np.zeros((amount, d))
    for i in range(amount):
        rdm = np.random.randint(m)
        C[i] = X[rdm]
    return C

# Defining the Cost function:
def costFunction(X, C):
    costs = np.zeros((len(C),1))
    for i in range(X.shape[0]):
        cost = np.sum(np.sqrt((X[i,:] - C)**2), 1)
        argmin = np.argmin(cost)
        costs[argmin] = costs[argmin] + cost[argmin]
    return np.sum(costs)

# Defining the Clusterization function:
def clusterizate(clusters, association):
    for j in range(ITERS):
        old_clusters = clusters.copy()
        for i in range(m):
            di = np.sum(np.sqrt((X[i,:] - clusters)**2), 1)
            association[i] = np.argmin(di)
        for i in range(len(clusters)):
            all_points = X[association[:,0] == i]
            mean_points = np.mean(all_points, axis=0)
            clusters[i] = mean_points
    return clusters, association

if RUN_E:
    # Clustering Routine to find the Elbow:
    ECosts = np.zeros((15,1))
    for i in range(1, 16):
        # Defining the Clusters Centroids:
        C = initClusters(i)
        # Defining the Centroids Association Vector:
        A = np.zeros((m,1))
        # Clusterizing:
        C, A = clusterizate(C, A)
        ECosts[i-1] = costFunction(C)
    pl.plot(range(1,16), ECosts, 'o', c='b')
    pl.show()

if RUN_C:
    # Clustering Routine to find the best Configuration of Centroids:
    bestCost = np.inf
    for i in range(50):
        # Defining the Clusters Centroids:
        C = initClusters(5)
        # Defining the Centroids Association Vector:
        A = np.zeros((m,1))
        # Clusterizing:
        C, A = clusterizate(C, A)
        if costFunction(C) < bestCost:
            bestClusters = C
    print(bestClusters)
