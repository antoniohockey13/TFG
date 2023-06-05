# -*- coding: utf-8 -*-
"""
Created on Wed May 17 19:17:08 2023

@author: Antonio
"""
import numpy as np
from . import Algoritmos
from . import Evaluar
from . import Grafica_Clusters

def KMeans(lista_trazas: np.array(3), lista_vertices: np.array(3),            \
           fit_trazas: np.array(3) or None, num_clusters: int,                \
           sample_weight: np.array(1) or None = None,                         \
           error_predict: np.array(1) or None = None,  n_init: int = 10,      \
           tol: float = 1e-6, graficas: bool = True):
    """
    Parameters
    ----------
    lista_trazas : np.array(3)
        Lista con las posiciones de las trazas y el vértice al que pertenecen
    lista_vertices : np.array(3)
        Lista con los vértices de la simulación.
    fit_trazas : np.array(3) or None
        Lista con las trazas y el vértice al que pertenecen que será predichas
    num_clusters : int
        Numero de cluster total
    sample_weight : np.array(1) or None, optional
        Errores. The default is None.
    error_predict : np.array(1) or None, optional
        Errores en trazas ha predecir. The default is None.
    n_init : int, optional
        Numero de veces que se corre el algoritmo con distintos centroides.
        The default is 10.
    tol : float, optional
        Tolerancia para asegurar convergencia. Default 1e-6
    graficas : bool, optional
        Se dibujan o no las graficas. Default is true

    Returns
    -------
    notaajustada : float
        Nota generada con adjusted_rand_score.
    notanorm : float
        nota generada con rand_score.
    distancia : float
        Distancia media.
    trazas_bien : float
        trazas Ok.
    trazas_mal : float
        trazas mal.
    clusters_bien : int
        Clusters completos bien.
    clusters_mal : int
        Clusters no completamente bien.
    vertices_faltan : int
        Vertices sin trazas.
    total_time : float
        Tiempo ejecutar algoritmo.
    num_clusters : int
        Número de clusters calculados por el algoritmo

    """

    X = np.column_stack((lista_trazas[:,1], lista_trazas[:,2]))

    num_clusters, centroides, etiquetas, total_time =                         \
        Algoritmos.KMeans(X = X, lista_trazas = lista_trazas,                 \
                          fit_trazas = fit_trazas,                            \
                          sample_weight = sample_weight,                      \
                          error_predict = error_predict,                      \
                          numcluster_manual = num_clusters, n_init = n_init,  \
                          tol = tol)

    if isinstance(fit_trazas, np.ndarray):
        lista_trazas_tot = np.vstack((lista_trazas, fit_trazas))
    else:
        lista_trazas_tot = lista_trazas

    notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,\
        clusters_mal, vertices_faltan = Evaluar.evaluacion_total(             \
                     lista_trazas_tot, etiquetas, centroides, lista_vertices)

    if graficas:
        Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas,     \
                                                 'K-Means')
        Grafica_Clusters.grafica_centroides_vertices(lista_vertices,          \
                                                    centroides, 'K-Means')

    return  notaajustada, notanorm, distancia, trazas_bien, trazas_mal,       \
            clusters_bien, clusters_mal, vertices_faltan, total_time,         \
            num_clusters

def MeanShift(lista_trazas: np.array(3), lista_vertices: np.array(3),         \
              fit_trazas: np.array(3) or None, quantile: float = 1e-2,        \
              n_samples: int = 357, min_bin_freq: int = 1,                   \
              graficas: bool = True):
    """
    Parameters
    ----------
    lista_trazas : np.array(3)
        Lista con las posiciones de las trazas y el vértice al que pertenecen
    lista_vertices : np.array(3)
        Lista con los vértices de la simulación.
    fit_trazas : np.array(3) or None
        Lista con las trazas y el vértice al que pertenecen que será predichas
    quantile : float, OPTIONAL
        Should be between [0, 1] 0.5 means that the median of all pairwise
        distances is used. For bandwith. Default 1e-2.
     n_samples : int, OPTIONAL
         The number of samples to use. If not given, all samples are used. For
         bandwith. Default = 299.
     min_bin_freq : int, OPTIONAL
         To speed up the algorithm, accept only those bins with at
         least min_bin_freq points as seeds. The default is 31.
    graficas : bool, optional
         Se dibujan o no las graficas. Default is true

    Returns
    -------
    notaajustada : float
        Nota generada con adjusted_rand_score.
    notanorm : float
        nota generada con rand_score.
    distancia : float
        Distancia media.
    trazas_bien : float
        trazas Ok.
    trazas_mal : float
        trazas mal.
    clusters_bien : int
        Clusters completos bien.
    clusters_mal : int
        Clusters no completamente bien.
    vertices_faltan : int
        Vertices sin trazas.
    total_time : float
        Tiempo ejecutar algoritmo.
    num_clusters : int
        Número de clusters calculados por el algoritmo

    """
    X = np.column_stack((lista_trazas[:,1], lista_trazas[:,2]))

    num_clusters, centroides, etiquetas, total_time = Algoritmos.MeanShift(   \
                                           X = X, fit_trazas = fit_trazas,    \
                                           quantile = quantile,               \
                                           n_samples =  n_samples,            \
                                           min_bin_freq = min_bin_freq)
    if isinstance(fit_trazas, np.ndarray):
        lista_trazas_tot = np.vstack((lista_trazas, fit_trazas))
    else:
        lista_trazas_tot = lista_trazas

    notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,\
        clusters_mal, vertices_faltan = Evaluar.evaluacion_total(             \
                     lista_trazas_tot, etiquetas, centroides, lista_vertices)
    if graficas:
        Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas,     \
                                                 'MeanShift')
        Grafica_Clusters.grafica_centroides_vertices(lista_vertices,          \
                                                    centroides, 'MeanShift')

    return  notaajustada, notanorm, distancia, trazas_bien, trazas_mal,       \
            clusters_bien, clusters_mal, vertices_faltan, total_time,         \
            num_clusters

def DBSCAN(lista_trazas: np.array(3), lista_vertices: np.array(3),            \
           fit_trazas: np.array(3) or None,                 \
           sample_weight: np.array(1) or None = None,                         \
           error_predict: np.array(1) or None = None, epsilon: float = 0.2,   \
           min_samples: int = 20, leaf_size: int = 12,  graficas: bool = True):
    """


    Parameters
    ----------
    lista_trazas : np.array(3)
        Lista con las posiciones de las trazas y el vértice al que pertenecen
    lista_vertices : np.array(3)
        Lista con los vértices de la simulación.
    fit_trazas : np.array(3) or None
        Lista con las trazas y el vértice al que pertenecen que será predichas
    sample_weight : np.array(1) or None, optional
        Errores. The default is None.
    error_predict : np.array(1) or None, optional
        Errores en trazas ha predecir. The default is None.
    epsilon : float. OPTIONAL
        Valor de epsilon para el ajuste de DBSCAN. The default is 0.2
    min_samples : int, OPTIONAL
        The number of samples in a neighborhood for a point to be considered
        as a core point. This includes the point itself. Deafault is 20
    leaf_size : in, OPTIONAL
        Leaf size passed to BallTree or cKDTree. This can affect the speed of
        the construction and query, as well as the memory required to store
        the tree. Default is 12
    graficas : bool, optional
         Se dibujan o no las graficas. Default is true

    Returns
    -------
    notaajustada : float
        Nota generada con adjusted_rand_score.
    notanorm : float
        nota generada con rand_score.
    distancia : float
        Distancia media.
    trazas_bien : float
        trazas Ok.
    trazas_mal : float
        trazas mal.
    clusters_bien : int
        Clusters completos bien.
    clusters_mal : int
        Clusters no completamente bien.
    vertices_faltan : int
        Vertices sin trazas.
    total_time : float
        Tiempo ejecutar algoritmo.
    num_clusters : int
        Número de clusters calculados por el algoritmo
    """

    X = np.column_stack((lista_trazas[:,1], lista_trazas[:,2]))

    num_clusters, centroides, etiquetas, total_time, num_noise =              \
        Algoritmos.DBSCAN(X = X, fit_trazas = fit_trazas,                     \
                          lista_trazas = lista_trazas,                        \
                          sample_weight = sample_weight,                      \
                          error_predict = error_predict, epsilon = epsilon,   \
                          min_samples = min_samples, leaf_size = leaf_size)

    if isinstance(fit_trazas, np.ndarray):
        lista_trazas_tot = np.vstack((lista_trazas, fit_trazas))
    else:
        lista_trazas_tot = lista_trazas

    notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,\
        clusters_mal, vertices_faltan = Evaluar.evaluacion_total(             \
                     lista_trazas_tot, etiquetas, centroides, lista_vertices)
    if graficas:
        Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas,     \
                                                 'DBSCAN')
        Grafica_Clusters.grafica_centroides_vertices(lista_vertices,          \
                                                    centroides, 'DBSCAN')
    return  notaajustada, notanorm, distancia, trazas_bien, trazas_mal,       \
            clusters_bien, clusters_mal, vertices_faltan, total_time,         \
            num_clusters

def EM_GMM(lista_trazas: np.array(3), lista_vertices: np.array(3),            \
           fit_trazas: np.array(3) or None, num_clusters: int,                \
           sample_weight: np.array(1) or None = None,                         \
           error_predict: np.array(1) or None = None, graficas: bool = True):
    """
    Parameters
    ----------
    lista_trazas : np.array(3)
        Lista con las posiciones de las trazas y el vértice al que pertenecen
    lista_vertices : np.array(3)
        Lista con los vértices de la simulación.
    fit_trazas : np.array(3) or None
        Lista con las trazas y el vértice al que pertenecen que será predichas
    num_clusters : int
        Numero de cluster total
    sample_weight : np.array(1) or None, optional
        Errores. The default is None.
    error_predict : np.array(1) or None, optional
        Errores en trazas ha predecir. The default is None.
    graficas : bool, optional
         Se dibujan o no las graficas. Default is true

    Returns
    -------
    notaajustada : float
        Nota generada con adjusted_rand_score.
    notanorm : float
        nota generada con rand_score.
    distancia : float
        Distancia media.
    trazas_bien : float
        trazas Ok.
    trazas_mal : float
        trazas mal.
    clusters_bien : int
        Clusters completos bien.
    clusters_mal : int
        Clusters no completamente bien.
    vertices_faltan : int
        Vertices sin trazas.
    total_time : float
        Tiempo ejecutar algoritmo.
    num_clusters : int
        Número de clusters calculados por el algoritmo
    """

    X = np.column_stack((lista_trazas[:,1], lista_trazas[:,2]))

    num_clusters, centroides, etiquetas, total_time =                         \
        Algoritmos.EM_GMM(X = X, lista_trazas = lista_trazas,                 \
                          fit_trazas = fit_trazas,                            \
                          sample_weight = sample_weight,                      \
                          numcluster_manual = num_clusters)
    if isinstance(fit_trazas, np.ndarray):
        lista_trazas_tot = np.vstack((lista_trazas, fit_trazas))
    else:
        lista_trazas_tot = lista_trazas

    notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,\
        clusters_mal, vertices_faltan = Evaluar.evaluacion_total(             \
                     lista_trazas_tot, etiquetas, centroides, lista_vertices)
    if graficas:
        Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas,     \
                                                 'EM-GMM')
        Grafica_Clusters.grafica_centroides_vertices(lista_vertices,          \
                                                    centroides, 'EM-GMM')
    return  notaajustada, notanorm, distancia, trazas_bien, trazas_mal,       \
            clusters_bien, clusters_mal, vertices_faltan, total_time,         \
            num_clusters


def AHC(lista_trazas: np.array(3), lista_vertices: np.array(3),               \
           fit_trazas: np.array(3) or None, distance_threshold: float = 1,    \
           graficas: bool = True):
    """
    Parameters
    ----------
    lista_trazas : np.array(3)
        Lista con las posiciones de las trazas y el vértice al que pertenecen
    lista_vertices : np.array(3)
        Lista con los vértices de la simulación.
    fit_trazas : np.array(3) or None
        Lista con las trazas y el vértice al que pertenecen que será predichas
    distance_threshold : float or None. DEFAULT 1
        The linkage distance threshold at or above which clusters will
        not be merged. The default is 1.
   graficas : bool, optional
        Se dibujan o no las graficas. Default is true

   Returns
   -------
   notaajustada : float
       Nota generada con adjusted_rand_score.
   notanorm : float
       nota generada con rand_score.
   distancia : float
       Distancia media.
   trazas_bien : float
       trazas Ok.
   trazas_mal : float
       trazas mal.
   clusters_bien : int
       Clusters completos bien.
   clusters_mal : int
       Clusters no completamente bien.
   vertices_faltan : int
       Vertices sin trazas.
   total_time : float
       Tiempo ejecutar algoritmo.
   num_clusters : int
       Número de clusters calculados por el algoritmo
    """

    X = np.column_stack((lista_trazas[:,1], lista_trazas[:,2]))

    num_clusters, centroides, etiquetas, total_time =                         \
        Algoritmos.AHC(X = X, lista_trazas = lista_trazas,                    \
                       fit_trazas = fit_trazas,                               \
                       distance_threshold = distance_threshold)

    if isinstance(fit_trazas, np.ndarray):
        lista_trazas_tot = np.vstack((lista_trazas, fit_trazas))
    else:
        lista_trazas_tot = lista_trazas

    notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,\
        clusters_mal, vertices_faltan = Evaluar.evaluacion_total(             \
                     lista_trazas_tot, etiquetas, centroides, lista_vertices)
    if graficas:
        Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas,     \
                                                 'AHC')
        Grafica_Clusters.grafica_centroides_vertices(lista_vertices,          \
                                                    centroides, 'AHC')
    return  notaajustada, notanorm, distancia, trazas_bien, trazas_mal,       \
            clusters_bien, clusters_mal, vertices_faltan, total_time,         \
            num_clusters

def BIRCH(lista_trazas: np.array(3), lista_vertices: np.array(3),             \
           fit_trazas: np.array(3) or None, threshold: float = 0.2,           \
           branching: int = 70, graficas: bool = True):
    """
    Parameters
    ----------
    lista_trazas : np.array(3)
        Lista con las posiciones de las trazas y el vértice al que pertenecen
    lista_vertices : np.array(3)
        Lista con los vértices de la simulación.
    fit_trazas : np.array(3) or None
        Lista con las trazas y el vértice al que pertenecen que será predichas
    threshold : float
        The linkage distance threshold at or above which clusters will
        not be merged. The default is 0.2.
    branching : int, optional
        Maximum number of CF subclusters in each node. The default is 70.
   graficas : bool, optional
        Se dibujan o no las graficas. Default is true

   Returns
   -------
   notaajustada : float
       Nota generada con adjusted_rand_score.
   notanorm : float
       nota generada con rand_score.
   distancia : float
       Distancia media.
   trazas_bien : float
       trazas Ok.
   trazas_mal : float
       trazas mal.
   clusters_bien : int
       Clusters completos bien.
   clusters_mal : int
       Clusters no completamente bien.
   vertices_faltan : int
       Vertices sin trazas.
   total_time : float
       Tiempo ejecutar algoritmo.
   num_clusters : int
       Número de clusters calculados por el algoritmo
    """

    X = np.column_stack((lista_trazas[:,1], lista_trazas[:,2]))

    num_clusters, centroides, etiquetas, total_time =                         \
        Algoritmos.BIRCH(X = X, fit_trazas = fit_trazas,                      \
                         threshold = threshold, branching = branching)

    if isinstance(fit_trazas, np.ndarray):
        lista_trazas_tot = np.vstack((lista_trazas, fit_trazas))
    else:
        lista_trazas_tot = lista_trazas

    notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,\
        clusters_mal, vertices_faltan = Evaluar.evaluacion_total(             \
                     lista_trazas_tot, etiquetas, centroides, lista_vertices)
    if graficas:
        Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas,     \
                                                 'BIRCH')
        Grafica_Clusters.grafica_centroides_vertices(lista_vertices,          \
                                                    centroides, 'BIRCH')
    return  notaajustada, notanorm, distancia, trazas_bien, trazas_mal,       \
            clusters_bien, clusters_mal, vertices_faltan, total_time,         \
            num_clusters
