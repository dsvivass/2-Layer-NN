#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tqdm

def rose(a=1, n=3, disp=2):
    '''Funcion para definir las rosas,
    disp = dispersion (mayor --> datos mas ajustados)'''
    # containing the radian values
    rads = np.arange(0, 2 * np.pi, 0.01)

    color1 = ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'r', 'r']
    color2 = ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'b', 'b']

    b = np.arange(0, rads.shape[0], rads.shape[0]/n)

    for i,rad in enumerate(tqdm.tqdm(rads)):
        r = a * np.cos(n * rad) + np.random.randn(1)/disp
        b_prueba = b <= i
        if np.where(b_prueba == True)[0][-1] % 2 == 0:
            plt.polar(rad, r, np.random.choice(color1, 1)[0]+'.')
        else:
            plt.polar(rad, r, np.random.choice(color2, 1)[0]+'.')

    # display the polar plot
    plt.show()


def main_nn():
    rose(n=10, a=1, disp=2)


if __name__ == '__main__':
    main_nn()