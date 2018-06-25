import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

param_grid = {
    "criterion": ['gini', 'entropy'],
    "splitter": ['best', 'random'],
    "max_features": ['auto', 'sqrt', 'log2', None],
    "random_state" : [2015412]
}

iris_ds = datasets.load_iris()
X = iris_ds.data
y = iris_ds.target

# Make a train/test split using 30% test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Hyperparameter Tuning
dt = DecisionTreeClassifier()
grid_search = GridSearchCV(dt, param_grid)
grid_search.fit(X_train,y_train)
best = grid_search.best_params_

y_hat = grid_search.predict(X_test)

print(best)
print(classification_report(y_hat, y_test))
