#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import csv
import sys

def rose(a=1, n=3, disp=2, paso_rads = 0.01, plot=True):
    '''Funcion para definir las rosas,
    disp = dispersion (mayor --> datos mas ajustados)'''
    # containing the radian values
    rads = np.arange(0, 2 * np.pi, paso_rads)

    color1 = ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'r', 'r']
    color2 = ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'b', 'b']

    b = np.arange(0, rads.shape[0], rads.shape[0]/n)

    l_2_export = []
    for i,rad in enumerate(tqdm.tqdm(rads, desc = 'GENERANDO DATOS')):
        r = a * np.cos(n * rad) + np.random.randn(1) / disp
        x = np.squeeze(r*np.cos(rad))
        y = np.squeeze(r*np.sin(rad))

        b_prueba = b <= i
        if np.where(b_prueba == True)[0][-1] % 2 == 0:
            plt.plot(x, y, np.random.choice(color1, 1)[0] + '.')
            l_2_export.append([x, y, 1])
        else:
            plt.plot(x, y, np.random.choice(color2, 1)[0] + '.')
            l_2_export.append([x, y, 0])

    np.savetxt("x_y_class.csv",  l_2_export, fmt='%01.3f', delimiter=",")
    print('{}ARCHIVO "x_y_class.csv" GENERADO SATISFACTORIAMENTE.{}'.format('\033[92m', '\033[0m'))
    # display the polar plot
    if plot == True: plt.show()

def main():
    n, a, disp, paso_rads, plot = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    plot = plot == 'True' # True si se cumple la condicion, asi pasa de str a bool
    rose(n=n, a=a, disp=disp, paso_rads=paso_rads, plot=plot)


if __name__ == '__main__':
    main()