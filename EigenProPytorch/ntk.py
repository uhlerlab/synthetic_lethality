from sklearn.linear_model import LinearRegression,LogisticRegression
import pandas as pd
import numpy as np
import math
import torch
from EigenProPytorch import eigenpro
from EigenProPytorch import kernel
import copy
from sklearn.model_selection import train_test_split
import copy
import random
from sklearn import svm

def model(X_train,y_train,X_val,y_val,X_test=None,y_test=None,bandwidth=5,num_epochs=20,cross_validate=True):

    def remove_duplicates(X,y):
        u, indices = np.unique(X, return_index=True, axis=0)
        X = X[indices]
        y = y[indices]
        return X,y

    X_train,y_train = remove_duplicates(X_train,y_train)
    X_val, y_val = remove_duplicates(X_val, y_val)
    X_test, y_test = remove_duplicates(X_test, y_test)

