#!/usr/bin/env python3

# Para practicar: http://cs231n.github.io/neural-networks-case-study/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

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
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    return W1, b1, W2, b2


def propagacionHaciaAdelante(W1, b1, W2, b2, X):

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    return Z1, A1, Z2, A2


def costo(A2, Y):

    cost = np.sum(-1/Y.shape[1]*(Y*np.log(A2) + (1-Y)*np.log(1 - A2)))

    return cost


def propagacionHaciaAtras(A1, A2, X, Y, W2):

    dZ2 = A2 - Y
    dW2 = 1/Y.shape[1]*(np.dot(dZ2, A1.T))
    db2 = 1/Y.shape[1]*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    # dZ1 = W2.T * dZ2 * (1 - np.power(A1, 2))
    dW1 = 1/Y.shape[1]*np.dot(dZ1, X.T)
    db1 = 1/Y.shape[1]*np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def actualizacionParametros(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    return W1, b1, W2, b2


def prediccion(A2):

    predict = (A2 > 0.5)

    return predict


def main_nn():
    np.random.seed(3)
    X, Y = X_Y('x_y_class.csv')
    n_0, n_1, n_2 = tamanoCapas(X, Y)
    W1, b1, W2, b2 = parametros(n_0, n_1, n_2)

    for i in tqdm.tqdm(range(10000)):
        Z1, A1, Z2, A2 = propagacionHaciaAdelante(W1, b1, W2, b2, X)
        cost = costo(A2, Y)
        dW1, db1, dW2, db2 = propagacionHaciaAtras(A1, A2, X, Y, W2)
        W1, b1, W2, b2 = actualizacionParametros(W1, b1, W2, b2, dW1, db1, dW2, db2, 1.2)
        plt.plot(i, cost, 'r.')

    predict = prediccion(A2)
    plt.show()

    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    X_Apred = np.append(xx.reshape(1,-1), yy.reshape(1,-1), axis=0)

    Z1, A1, Z2, A2 = propagacionHaciaAdelante(W1, b1, W2, b2, X_Apred)

    Z = prediccion(A2)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z.reshape(xx.shape), cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=Y[0], cmap=plt.cm.Spectral)
    plt.show()

if __name__ == '__main__':
    main_nn()