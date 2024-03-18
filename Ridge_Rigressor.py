import pandas as pd
from sklearn.base import BaseEstimator
import numpy as np


class RidgeRegressor(BaseEstimator):
    def __init__(self, kernel='linear', l=1, c=0, d=2, r=1):
        self.kernel = kernel
        self.l = l
        self.c = c
        self.d = d
        self.r = r

    def linearKernel(self, X1: pd.Series, X2: pd.Series):
        return (X1 @ X2) + self.c

    def polynomialKernel(self, X1: pd.Series, X2: pd.Series):
        return (X1 @ X2 + self.r) ** self.d

    def calcKernelMatrix(self, X1: pd.DataFrame, X2: pd.DataFrame):
        kdict = {
            'linear': self.linearKernel,
            'polynomial': self.polynomialKernel,
        }
        kernelFunc = kdict[self.kernel]
        matrix = [[kernelFunc(X1.iloc[i], X2.iloc[j])
                   for j in range(len(X2))] for i in range(len(X1))]
        return np.array(matrix)

    def fit(self, X, Y):
        kernel_matrix = self.calcKernelMatrix(X, X)
        self.temp = Y.T @ np.linalg.inv(self.l *
                                        np.eye(kernel_matrix.shape[0]) + kernel_matrix)
        self.X = X
        return self

    def predict(self, X_test):
        return self.temp @ self.calcKernelMatrix(self.X, X_test)

    def score(self, X_test, y_test):
        # Calculates MSE
        predict = self.predict(X_test)
        return np.mean([(y - y_p)**2 for y, y_p in zip(y_test, predict)])
