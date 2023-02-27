# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:45:28 2023

@author: Antonio
"""
import sklearn.cluster as skc
import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt


def num_clusters(X: list[np.array(2)], minclusters: int, maxclusters: int):
    """
    Parameters
    ----------
    X : list[np.array(2)]
        Conjunto de trazas.
    minclusters : int
        Número mínimo de clusters.
    maxclusters : int
        Número máximo de clusters.

    Returns
    -------
    None.

    """

    # Elegir número de clusters a partir gráfica ver codo
    inercia = []
    for i in range(minclusters, maxclusters):
        algoritmo = skc.KMeans(n_clusters = i, init = 'k-means++', n_init = 5)
        algoritmo.fit(X)
        inercia.append(algoritmo.inertia_)

    plt.plot(list(range(minclusters, maxclusters)), inercia, 'x')
    plt.xlabel('Nº clusters')
    plt.ylabel('Inercia')
    plt.show()
    # Buscar como elegir codo con regla matemática segunda derivada(?)


def Evaluar(lista_vertices: np.array(np.array(3)),                     \
                   lista_trazas: np.array(np.array(3)), num_vertices: int,    \
                   num_trazas: int, etiquetas: np.array(int),                 \
                   centroides: np.array(float)):
    """
    Parameters
    ----------
    lista_vertices : np.array(np.array(3))
        Lista con los vértices de la simulación.
    lista_trazas : np.array(np.array(3))
        Lista con las trazas de la simulación.
    num_vertices : int
        Número se vértices en la simulacion.
    num_trazas : int
        Número de trazas en la simulación.
    etiquetas : np.array(int)
        Array que indica a que cluster pertenece cada traza.
    centroides : np.array(float)
        Posición de los centroides de cada cluster.

    Returns
    -------
    (float, float)
        Puntos (sobre 10), distancia vertice-centroide.

    """

    clustertovertex = [] # Posición marca cluster, número vertice [2 , 1...]
    # Cluster 0--> Vertice 2, cluster 1--> Vertice 1

    distanciatot = 0
    for icentro in range(len(centroides)):
        centroi = centroides[icentro]
        # Distancia se establece como infinito para así escoger la menor y elegir
        # que vértice esta más cerca de que centroide
        distancia_minima = np.inf
        verticeseleccionado = -1

        for ivertice in range(len(lista_vertices)):
            posverticei = [lista_vertices[ivertice,1], lista_vertices[ivertice,2]]
            verticei = lista_vertices[ivertice,0]
            distancia = np.sqrt((centroi[0]-posverticei[0])**2+\
                                (centroi[1]-posverticei[1])**2)
            if distancia < distancia_minima:
                distancia_minima = distancia
                verticeseleccionado = verticei

        distanciatot += distancia_minima
        clustertovertex.append(verticeseleccionado)

    puntos = 0
    # Comprobar si trazas bien asignadas
    for itraza in range(len(lista_trazas)):
        # A que cluster se corresponde la traza
        cluster = etiquetas[itraza]
        # A que vertice se corresponde la traza
        vertice = lista_trazas[itraza,0]
        clustersmal = []
        if vertice == clustertovertex[cluster]:
            # print('Traza bien asignada')
            puntos += 1
        else:
            # print(f'La traza {itraza} esta mal asignada')
            puntos -= 1
            clustersmal.append(cluster)

        # Penalizar si falla mucho en el mismo vértice
        for i in range(len(clustersmal)):
            clusteri = clustersmal[i]
            for j in range(len(clustersmal)):
                clusterj = clustersmal[j]

                if clusteri == clusterj:
                    puntos -=1/2 # Se divide entre dos por que se encontrara el
                                 # elemento dos veces
                    # print(f'1 traza mal en el cluster {clusteri}')

    puntosnorm =puntos/(num_vertices*num_trazas)*10
    distancianorm = distanciatot/num_vertices
    return(puntosnorm, distancianorm)


def evaluacion(lista_trazas: np.array(np.array(3)), etiquetas:  np.array(int)):
    """
    Se basa en:
    https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

    Parameters
    ----------
    lista_trazas : np.array(np.array(3))
        Lista con las trazas de la simulación.
    etiquetas : np.array(int)
        Array que indica a que cluster pertenece cada traza.

    Returns
    -------
    (float, float)
        Nota generada con adjusted_rand_score, nota generada con rand_score.

    """

    verticepertenecentrazas = lista_trazas[:,0]

    notaajustada = skm.adjusted_rand_score(verticepertenecentrazas, etiquetas)
    notanorm = skm.rand_score(verticepertenecentrazas, etiquetas)
    return notaajustada, notanorm
