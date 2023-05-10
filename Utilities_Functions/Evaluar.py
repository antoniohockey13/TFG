# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 19:32:54 2023

@author: Antonio
"""
import numpy as np
import sklearn.metrics as skm

def evaluacion(lista_trazas: np.array(np.array(3)), etiquetas: np.array(int)):
    """
    Se basa en:
    https://scikit-learn.org/stable/modules/clustering.html#clustering-
    performance-evaluation

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

def cluster_to_vertex(centroides: np.array(float),                            \
                      lista_vertices: np.array(3)):
    """
    Genera un array que relaciona cada cluster con su vértice más parecido

    Parameters
    ----------
    centroides : np.array(float)
        Posición de los centroides de cada cluster.
    lista_vertices : np.array(np.array(3))
        Lista con los vértices de la simulación.

    Returns
    -------
   clustertovertex  :  np.array(int)
       Lista que relaciona el número de cada cluster con cada vértice
           # Posición marca cluster, número vertice [2 , 1...]
           # Cluster 0--> Vertice 2, cluster 1--> Vertice 1.

    """
    clustertovertex = [] # Posición marca cluster, número vertice [2 , 1...]
    # Cluster 0--> Vertice 2, cluster 1--> Vertice 1

    for icentro in range(len(centroides)):
        centroi = centroides[icentro]
        # Distancia se establece como infinito para así escoger la menor y elegir
        # que vértice esta más cerca de que centroide
        distancia_minima = np.inf
        verticeseleccionado = -1

        for ivertice in range(len(lista_vertices)):
            posverticei=[lista_vertices[ivertice,1],lista_vertices[ivertice,2]]
            verticei = lista_vertices[ivertice,0]
            distancia = np.sqrt((centroi[0]-posverticei[0])**2+               \
                                (centroi[1]-posverticei[1])**2)

            if distancia < distancia_minima:
                distancia_minima = distancia
                verticeseleccionado = verticei
        clustertovertex.append(verticeseleccionado)
    return clustertovertex

def distancia_media(centroides: np.array(float),                              \
                    lista_vertices: np.array(3),                              \
                    clustertovertex: np.array(int)):
    """
    Cálcula la distancia media entre los vértices y los centroides de los
    clusters
    Parameters
    ----------
    centroides : np.array(float)
        Posición de los centroides de cada cluster.
    lista_vertices : np.array(np.array(3))
        Lista con los vértices de la simulación.
    clustertovertex  :  np.array(int)
        Lista que relaciona el número de cada cluster con cada vértice
            # Posición marca cluster, número vertice [2 , 1...]
            # Cluster 0--> Vertice 2, cluster 1--> Vertice 1.

    Returns
    -------
    float
        Distancia media.

    """

    distanciatot = 0
    for icentroide in range(len(centroides)):
        centroide = centroides[icentroide]
        numvertice = clustertovertex[icentroide]
        vertice = lista_vertices[int(numvertice)]
        if centroide[0] == np.inf or centroide[1] == np.inf:
            print('Un cluster esta vacío')
        else:
            distancia =np.sqrt((centroide[0]-vertice[1])**2+                  \
                               (centroide[1]-vertice[2])**2)
        distanciatot += distancia
    return distanciatot/len(lista_vertices)



def tabla_trazas(lista_trazas: np.array(np.array(3)),                         \
                 etiquetas: np.array(int), num_trazas_en_v: list[int],        \
                 clustertovertex: np.array(int)):
    """
    Función para obtener número absoluto de trazas y clusters bien/mal
    identificados.

    Parameters
    ----------
    lista_trazas : np.array(np.array(3))
        Lista con las trazas de la simulación.
    etiquetas : np.array(int)
        Array que indica a que cluster pertenece cada traza.
    num_trazas_en_v  :  list[int]
        Lista que indica el número de trazas en cada vértice
            # Posición marca vértice, número número de trazas [10 , 15...]
            # Vértice 0--> 10 trazas, Vértice  1--> 15 trazas
    clustertovertex  :  np.array(int)
        Lista que relaciona el número de cada cluster con cada vértice
            # Posición marca cluster, número vertice [2 , 1...]
            # Cluster 0--> Vertice 2, cluster 1--> Vertice 1

    Returns
    -------
    (int, int, int, int)
        trazas_bien, trazas_mal, clusters_bien, clusters_mal.

    """
    lista_trazas_bien = []
    lista_trazas_mal = []
    trazas_bien = 0
    trazas_mal = 0
    trazas_en_vertices= {} # Comprobar cuantas trazas en cada vértice
    for i in range(len(num_trazas_en_v)):
        trazas_en_vertices[i] = 0

    # Comprobar si trazas bien asignadas
    for itraza in range(len(lista_trazas)):
        # A que cluster se corresponde la traza
        cluster = etiquetas[itraza]
        # A que vertice se corresponde la traza
        vertice = lista_trazas[itraza,0]
        if vertice not in trazas_en_vertices:
            trazas_en_vertices[vertice] = 1
        else:
            trazas_en_vertices[vertice] += 1
        if vertice == clustertovertex[cluster]:
            # print('Traza bien asignada')
            trazas_bien += 1
            lista_trazas_bien.append(vertice)
        else:
            # print(f'La traza {itraza} esta mal asignada')
            trazas_mal += 1
            lista_trazas_mal.append(vertice)


    vertices_faltan = 0
    for ivertice in range(len(num_trazas_en_v)):
        if trazas_en_vertices[ivertice] == 0:
            vertices_faltan += 1

    clusters_bien = 0
    contador_trazas_bien = {}

    for itraza in lista_trazas_bien:
        if itraza not in contador_trazas_bien:
            contador_trazas_bien[itraza] = 1
        else:
            contador_trazas_bien[itraza]+=1
        if contador_trazas_bien[itraza] == num_trazas_en_v[int(itraza)]:
            clusters_bien +=1

    clusters_mal = 0
    contador_trazas_mal = {}
    for itraza in lista_trazas_mal:
        if itraza not in contador_trazas_mal:
            contador_trazas_mal[itraza] = 1
        else:
            contador_trazas_mal[itraza] += 1
        if contador_trazas_mal[itraza] == 1:
            clusters_mal += 1
    return(trazas_bien, trazas_mal, clusters_bien, clusters_mal,              \
           vertices_faltan)


def num_trazas_en_vertices_con_lista_trazas(lista_trazas: np.array(3)):
    num_trazas_en_v = {}
    for itraza in lista_trazas:
        if itraza[0] not in num_trazas_en_v:
            num_trazas_en_v[itraza[0]] = 1
        else:
            num_trazas_en_v[itraza[0]] += 1
    return num_trazas_en_v

def evaluacion_total(lista_trazas: np.array(np.array(3)),                     \
                     etiquetas: np.array(int), centroides: np.array(float),   \
                     lista_vertices: np.array(3), num_trazas_en_v: list[int]):
    """
    Llama al resto de funciones de evaluación para devolver todos los
    resultados juntos.

    Parameters
    ----------
    lista_trazas : np.array(np.array(3))
        Lista con las trazas de la simulación.
    etiquetas : np.array(int)
        Array que indica a que cluster pertenece cada traza.
    centroides : np.array(float)
        Posición de los centroides de cada cluster.
    lista_vertices : np.array(np.array(3))
        Lista con los vértices de la simulación.
    num_trazas_en_v  :  list[int]
        Lista que indica el número de trazas en cada vértice
            # Posición marca vértice, número número de trazas [10 , 15...]
            # Vértice 0--> 10 trazas, Vértice  1--> 15 trazas

    Returns
    -------
    (notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,
     clusters_mal)

    """
    num_trazas_en_v = num_trazas_en_vertices_con_lista_trazas(lista_trazas)
    notaajustada, notanorm = evaluacion(lista_trazas, etiquetas)
    ctv = cluster_to_vertex(centroides, lista_vertices)
    distancia = distancia_media(centroides, lista_vertices, ctv)
    trazas_bien, trazas_mal, clusters_bien, clusters_mal, vertices_faltan     \
        = tabla_trazas(lista_trazas, etiquetas, num_trazas_en_v, ctv)

    return(notaajustada, notanorm, distancia, trazas_bien, trazas_mal,        \
           clusters_bien, clusters_mal, vertices_faltan)
