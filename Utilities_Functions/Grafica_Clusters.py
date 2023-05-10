# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:06:04 2023

@author: Antonio
"""
import random
import matplotlib.pyplot as plt
import numpy as np

def grafica_colores_cluster(lista_trazas: np.array(3),                        \
                            etiquetas: np.array(int), algoritmo: str):
    """
    It plots the clusters

    Parameters
    ----------
    lista_trazas: np.array([V, z, t])
        Lista con las posiciones de las trazas y el vértice al que pertenecen
    etiquetas : np.array(int)
        Array que indica a que cluster pertenece cada traza.
    algoritmo : str
        Nombre del algoritmo con el que se obtienen los datos.

    """
    markers = ['.', 'o', 'v', 's', 'p', 'P', '*', '+', 'd', 'D']
    colours = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink',   \
               'gray', 'olive', 'cyan']

    etiquetas_unique = set(etiquetas)
    for i_etiqueta in etiquetas_unique:
        c = random.choice(colours)
        m = random.choice(markers)
        z = []
        t = []
        for i_traza in range(len(lista_trazas)):
            if etiquetas[i_traza] == i_etiqueta:
                z.append(lista_trazas[i_traza, 1])
                t.append(lista_trazas[i_traza, 2])
        plt.plot(z, t, m, c = c, label = str(i_etiqueta))
    plt.xlabel(r"$z$")
    plt.ylabel(r"$t$")
    # plt.legend(loc = 'best')
    plt.title(algoritmo)
    plt.show()
