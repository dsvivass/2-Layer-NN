#!/usr/bin/env python3

import numpy as np
import pandas as pd

def X_Y(fileName):
    df = pd.read_csv(fileName, header=None)
    df.columns = ['X', 'Y', 'Class']
    X = df[['X', 'Y']].to_numpy()
    Y = df['Class'].to_numpy().reshape(-1, 1)
    return X,Y

def main_nn():
    X, Y = X_Y('x_y_class.csv')

if __name__ == '__main__':
    main_nn()