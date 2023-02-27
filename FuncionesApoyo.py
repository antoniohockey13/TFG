# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:55:08 2023

@author: Antonio
"""
import numpy as  np

def centroides(etiquetas: np.array(1), lista_trazas: np.array(3),      \
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
            centroidei = [zmedio, tmedio]
            centroides.append(centroidei)
        icluster += 1
    return(np.array(centroides))

def encontrar_num_clusters():
    a = 5
