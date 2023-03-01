# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:19:29 2023

@author: Antonio
"""
import numpy as np

def tabla_trazas(lista_vertices: np.array(np.array(3)),                       \
                 lista_trazas: np.array(np.array(3)), num_trazas: int,        \
                 etiquetas: np.array(int), centroides: np.array(float),       \
                 num_trazas_en_v: list[int]):
    """
    Función para obtener número absoluto de trazas y clusters bien/mal
    identificados.

    Parameters
    ----------
    lista_vertices : np.array(np.array(3))
        Lista con los vértices de la simulación.
    lista_trazas : np.array(np.array(3))
        Lista con las trazas de la simulación.
    num_trazas : int
        Número de trazas en la simulación.
    etiquetas : np.array(int)
        Array que indica a que cluster pertenece cada traza.
    centroides : np.array(float)
        Posición de los centroides de cada cluster.

    Returns
    -------
    (int, int, int, int)
        trazas_bien, trazas_mal, clusters_bien, clusters_mal.

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

    lista_trazas_bien = []
    lista_trazas_mal = []
    trazas_bien = 0
    trazas_mal = 0

    # Comprobar si trazas bien asignadas
    for itraza in range(len(lista_trazas)):
        # A que cluster se corresponde la traza
        cluster = etiquetas[itraza]
        # A que vertice se corresponde la traza
        vertice = lista_trazas[itraza,0]

        if vertice == clustertovertex[cluster]:
            # print('Traza bien asignada')
            trazas_bien += 1
            lista_trazas_bien.append(vertice)
        else:
            # print(f'La traza {itraza} esta mal asignada')
            trazas_mal += 1
            lista_trazas_mal.append(vertice)

    clusters_bien = 0
    contador_trazas_bien = 1

    itraza = 0
    while itraza < len(lista_trazas_bien)-1:
        ivertice = lista_trazas_bien[itraza]
        if ivertice == lista_trazas_bien[itraza+1]:
            contador_trazas_bien +=1
            if contador_trazas_bien == num_trazas_en_v[int(ivertice)]:
                clusters_bien +=1

        else:
            contador_trazas_bien = 1
        itraza += 1


    clusters_mal = 0
    contador_trazas_mal = {}
    for itraza in lista_trazas_mal:
        if itraza not in contador_trazas_mal:
            contador_trazas_mal[itraza] = 1
        else:
            contador_trazas_mal[itraza] += 1
        if contador_trazas_mal[itraza] == 1:
            clusters_mal += 1
    return(trazas_bien, trazas_mal, clusters_bien, clusters_mal)


def evaluar(lista_vertices: np.array(np.array(3)),                            \
            lista_trazas: np.array(np.array(3)), num_vertices: int,           \
            num_trazas: int, etiquetas: np.array(int),                        \
            centroides: np.array(float), clustertovertex: np.array(int)):
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
    clustersmal = []
    # Comprobar si trazas bien asignadas
    for itraza in range(len(lista_trazas)):
        # A que cluster se corresponde la traza
        cluster = etiquetas[itraza]
        # A que vertice se corresponde la traza
        vertice = lista_trazas[itraza,0]
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

    puntosnorm =puntos/(num_trazas)*10
    distancianorm = distanciatot/num_vertices
    return(puntosnorm, distancianorm)
