# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:55:08 2023

@author: Antonio
"""
import numpy as  np
import random
import matplotlib.pyplot as plt


def encontrar_centroides(etiquetas: np.array(1), lista_trazas: np.array(3),   \
               num_clusters: int):
    """
    Parameters
    ----------
    etiquetas : np.array
        Indica a que cluster pertenece cada traza.
    lista_trazas : np.array(num_trazas, 3)
        Lista con: [vertice, z, t] de cada traza.
    num_clusters : int
        Indica el número total de clusters.

    Returns
    -------
    np.array(num_clusters, 2)
        Posición de los centroides calculado como la media aritmética
        de las trazas pertenecientes a cada cluster.
    """
    icluster = 0
    centroides = []
    for icluster in range(num_clusters):
        z = []
        t = []
        for i in range(len(lista_trazas)):
            if etiquetas[i] == icluster:
                z.append(lista_trazas[i,1])
                t.append(lista_trazas[i,2])
        if z != []:
            zmedio = np.mean(z)
            tmedio = np.mean(t)

        else:
            zmedio = np.inf
            tmedio = np.inf
        centroidei = [zmedio, tmedio]
        centroides.append(centroidei)
    return np.array(centroides)

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
    plt.xlabel(r"$z/\sigma_z$")
    plt.ylabel(r"$t/\sigma_t$")
    # plt.legend(loc = 'best')
    plt.title(algoritmo)
    plt.savefig(f'Centroides vs Vertices-{algoritmo}')
    plt.show()

def errores_to_sample_weight(errores: np.array(2)):
    """
    Parameters
    ----------
    errores : np.array(2)
        Error in track measurement.

    Returns
    -------
    errores : np.array(1)
        Sample weight.

    """
    errores = np.sqrt(np.sum(np.square(errores), axis=1))

    error_max = max(errores)
    errores = 1-(errores/error_max)
    return errores

def momentum_to_sample_weight(momentum: np.array(1)):
    """
    Parameters
    ----------
    momentum : np.array(1)
        pt.

    Returns
    -------
    errores : np.array(1)
        Sample weight.

    """
    errores = momentum
    error_max = max(errores)
    errores = errores/error_max
    return errores
