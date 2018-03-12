import numpy as np

X = np.array([[100,2],[50,42],[45,31]])
y = np.array([[5],[25],[22]])
Theta = np.array([[-0.18, 0.041, 0.551]])

X1 = np.insert(X,0,1,axis=1)
X_new = np.array([[1,60,35]])

y_hat = X_new.dot(Theta.T)

residual = np.sum(np.sqrt((y - y_hat)**2)) # could be np.abs(y_matrix - y_hat) instead of actual one.

print("X:\n", X)
print("y:\n", y)
print("Theta:\n", Theta)
print("X1:\n", X1)
print("y_hat:\n", y_hat)
print("Residual:\n", residual)




