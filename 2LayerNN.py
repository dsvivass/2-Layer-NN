#!/usr/bin/env python3

import numpy as np
import pandas as pd

def X_Y(fileName):

    df = pd.read_csv(fileName, header=None)
    df.columns = ['X', 'Y', 'Class']
    X = np.append(df['X'].to_numpy().reshape(1,-1), df['Y'].to_numpy().reshape(1,-1), axis=0)
    Y = df['Class'].to_numpy().reshape(1, -1)

    return X,Y

def tamanoCapas(X, Y):

    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return n_x, n_h, n_y

def parametros(n_x, n_h, n_y):

    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    return W1, b1, W2, b2

def propagacionHaciaAdelante(W1, b1, W2, b2, X, Y):

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    return Z1, A1, Z2, A2

def costo(A2, Y):

    cost = np.sum(-1/Y.shape[1]*(Y*np.log(A2) + (1-Y)*np.log(1 - A2)))

    return cost

def main_nn():

    X, Y = X_Y('x_y_class.csv')
    n_0, n_1, n_2 = tamanoCapas(X, Y)
    W1, b1, W2, b2 = parametros(n_0, n_1, n_2)
    Z1, A1, Z2, A2 = propagacionHaciaAdelante(W1, b1, W2, b2, X, Y)
    cost = costo(A2, Y)

if __name__ == '__main__':
    main_nn()