# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 18:10:06 2023

@author: Antonio
"""
import time
import numpy as np
import sklearn.cluster as skc
import sklearn.mixture as skm
from . import NumeroClusters
from . import FuncionesApoyo as FA


def KMeans(X: np.array(2), lista_trazas: np.array(3),                         \
           numcluster_manual: int or None,                                    \
           fit_trazas: np.array(3) or None = None,                            \
           sample_weight: np.array(1) or None = None,                         \
           error_predict: np.array(1) or None = None, n_init: int = 10,       \
           tol: float = 1e-6):
    """
    Realiza el ajuste al algoritmo KMeans y devuelve los valores de interes

    Parameters
    ----------
    X : np.array(2)
        Lista de las trazas a clusterizar.
    lista_trazas : np.array(3)
        Lista con todas las trazas y al vertice perteneciente.
    numcluster_manual : int or None
        Numero de cluster total si es un número se toma el número si es
        'None' se calcula el mejor ajuste con la nota ajustada.
    fit_trazas : np.array(3) or None
        Lista con las trazas y el vértice al que pertenecen que será predichas.
        Default None
    sample_weight : np.array(1) or None
        Errores. Default None
    error_predict : np.array(1) or None
        Errores en trazas ha predecir. Default None
    n_init : int, OPTIONAL
        Numero de veces que se corre el algoritmo con distintos centroides.
        Default 10
    tol : float, OPTIONAL
        Tolerancia para asegurar convergencia. Default 1e-6

    Returns
    -------
    num_clusters, centroides, etiquetas, total_time

    """
    t0 = time.time_ns()

    if numcluster_manual is None:
        num_clusters = NumeroClusters.kmeans_num_clusters(X, lista_trazas,    \
                                        num_min = 190, num_max = 210,         \
                                        num_tries = 20)

    else:
        num_clusters = numcluster_manual

    kmeans =skc.KMeans(n_clusters = num_clusters, init = 'k-means++',         \
                       max_iter = 300, n_init = n_init, tol = tol)

    kmeans.fit(X, sample_weight = sample_weight)

    etiquetas = kmeans.labels_

    if isinstance(fit_trazas, np.ndarray):
        etiquetas_predict = kmeans.predict(fit_trazas[:,1:3],                 \
                                           sample_weight = error_predict)
        etiquetas = np.concatenate((etiquetas, etiquetas_predict))

    # print(kmeans.n_iter_)
    centroides = kmeans.cluster_centers_
    total_time = (time.time_ns()-t0)*1e-9

    # print(kmeans.n_iter_) # Muestra numero iteracciones realizado

    return(num_clusters, centroides, etiquetas, total_time)


def MeanShift(X: np.array(2), fit_trazas: np.array(3) or None = None,         \
              quantile: float = 0.01, n_samples: int = 299,                   \
              min_bin_freq: int = 31):
    """
    Realiza el ajuste al algoritmo MeanShift y devuelve los valores de interes

    Parameters
    ----------
    X : np.array(2)
        Lista de las trazas a clusterizar.
    fit_trazas : np.array(3) or None
        Lista con las trazas y el vértice al que pertenecen que será predichas.
        Default None
    quantile : float, OPTIONAL
        Should be between [0, 1] 0.5 means that the median of all pairwise
        distances is used. For bandwith. Default 0.01
    sample_weight : np.array(1)
        Errores
    error_predict : np.array(1)
        Errores en trazas ha predecir
    n_samples : int, OPTIONAL
        The number of samples to use. If not given, all samples are used. For
        bandwith. Default = 299
    min_bin_freq : int, OPTIONAL
        To speed up the algorithm, accept only those bins with at
        least min_bin_freq points as seeds. Default 31

    Returns
    -------
    num_clusters, centroides, etiquetas, total_time
    """

    t0 = time.time_ns()
    bandwidth = skc.estimate_bandwidth(X = X, quantile = quantile ,\
                                       n_samples = n_samples, n_jobs = -1)

    meanshift = skc.MeanShift(bandwidth = bandwidth, seeds = None,            \
                              bin_seeding = True, min_bin_freq = min_bin_freq,\
                              cluster_all = True, n_jobs= -1, max_iter=300)
    meanshift.fit(X)
    etiquetas = meanshift.labels_

    if isinstance(fit_trazas, np.ndarray):
        etiquetas_predict = meanshift.predict(fit_trazas[:,1:3])
        etiquetas = np.concatenate((etiquetas, etiquetas_predict))


    labels_unique = np.unique(etiquetas)
    num_clusters = len(labels_unique)

    centroides = meanshift.cluster_centers_
    total_time = (time.time_ns()-t0)*1e-9
    # print(meanshift.n_iter_) # Muestra numero iteracciones realizado

    return num_clusters, centroides, etiquetas, total_time


def DBSCAN(X: np.array(2), lista_trazas: np.array(3),                         \
           fit_trazas: np.array(3) or None = None,                            \
           sample_weight: np.array(1) or None = None,                         \
           error_predict: np.array(1) or None = None, epsilon: float = 0.2,   \
           min_samples: int = 5, leaf_size: int = 10):
    """
    Realiza el ajuste al algoritmo DBSCAN y devuelve los valores de interes

    Parameters
    ----------
    X : np.array(2)
        Lista de las trazas a clusterizar.
    lista_trazas : np.array(3)
        Lista con todas las trazas y al vertice perteneciente.
    fit_trazas : np.array(3) or None
        Lista con las trazas y el vértice al que pertenecen que será predichas.
        Default None.
     sample_weight : np.array(1) or None
         Errores. Default None.
    error_predict : np.array(1) or None
        Error en fit_trazas. Default None.
    epsilon : float. OPTIONAL
        Valor de epsilon para el ajuste de DBSCAN. The default is 0.8
    min_samples : int, OPTIONAL
        The number of samples in a neighborhood for a point to be considered
        as a core point. This includes the point itself. Deafault is 5
    leaf_size : in, OPTIONAL
        Leaf size passed to BallTree or cKDTree. This can affect the speed of
        the construction and query, as well as the memory required to store
        the tree. Default is 10

    Returns
    -------
    num_clusters, centroides, etiquetas, total_time, num_noise

    """
    t0 = time.time_ns()
    dbscan = skc.DBSCAN(eps = epsilon, min_samples = min_samples,             \
                        metric_params = None, algorithm = 'auto',             \
                        leaf_size = leaf_size, p = None, n_jobs = -1)

    dbscan.fit(X, sample_weight = sample_weight)

    # if type(fit_trazas) != type(None):
    #     etiquetas_predict = dbscan.predict(fit_trazas[:,1:3],               \
    #                                             sample_weight = error_predict)
    #      etiquetas = np.concatenate((etiquetas, etiquetas_predict))

    # No existe función predict

    etiquetas = dbscan.labels_

    num_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
    num_noise = list(etiquetas).count(-1)

    centroides = FA.encontrar_centroides(etiquetas, lista_trazas,             \
                                         num_clusters)

    total_time = (time.time_ns()-t0)*1e-9

    return num_clusters, centroides, etiquetas, total_time, num_noise

def EM_GMM(X: np.array(2), lista_trazas: np.array(3), numcluster_manual: int, \
           fit_trazas: np.array(2) or None = None,\
           sample_weight: np.array(1) or None = None):
    """
    Realiza el ajuste al algoritmo EM-GMM y devuelve los valores de interes

    Parameters
    ----------
    X : np.array(2)
        Lista de las trazas a clusterizar.
    lista_trazas : np.array(3)
        Lista con todas las trazas y al vertice perteneciente.
    numcluster_manual : int
        Numero de cluster total
    fit_trazas : np.array(3) or None
        Lista con las trazas y el vértice al que pertenecen que será predichas.
        Default None
     sample_weight : np.array(1) or None
         Errores. Default None

    Returns
    -------
    num_clusters, centroides, etiquetas, total_time

    """

    t0 = time.time_ns()

    em_gmm = skm.GaussianMixture(n_components = numcluster_manual, n_init = 1,\
                                 init_params = 'kmeans', warm_start = True,   \
                                 weights_init = sample_weight)
    etiquetas = em_gmm.fit_predict(X)
    # print(em_gmm.n_iter_)

    if isinstance(fit_trazas, np.ndarray):
        etiquetas_predict = em_gmm.predict(fit_trazas[:,1:3])
        etiquetas = np.concatenate((etiquetas, etiquetas_predict))

    centroides = FA.encontrar_centroides(etiquetas, lista_trazas,             \
                                         numcluster_manual)

    total_time = (time.time_ns()-t0)*1e-9

    return numcluster_manual, centroides, etiquetas, total_time

def AHC(X: np.array(2), lista_trazas: np.array(3),                            \
        fit_trazas: np.array(3) or None = None,                               \
        distance_threshold: float = 1):
    """
    Realiza el ajuste al algoritmo Agglomerative Hierarchical Clustering
    y devuelve los valores de interes

    Parameters
    ----------
    X : np.array(2)
        Lista de las trazas a clusterizar.
    lista_trazas : np.array(3)
        Lista con todas las trazas y al vertice perteneciente.
    fit_trazas : np.array(3) or None
        Lista con las trazas y el vértice al que pertenecen que será predichas.
        Default None
    distance_threshold : float. DEFAULT 1
        The linkage distance threshold at or above which clusters will
        not be merged
    Returns
    -------
    num_clusters, centroides, etiquetas, total_time

    Si distance_threslhold --> numero
    --> n_clusters = None y compute_full_tree = True

    """
    t0 = time.time_ns()


    agglomerative = skc.AgglomerativeClustering(n_clusters = None,            \
                                      distance_threshold= distance_threshold, \
                                      compute_full_tree = True,               \
                                      linkage = 'ward')

    agglomerative.fit(X)

    etiquetas = agglomerative.labels_


    # if isinstance(fit_trazas, np.ndarray):
    #     etiquetas_predict = agglomerative.predict(fit_trazas[:,1:3])
    #     etiquetas = np.concatenate((etiquetas, etiquetas_predict))
    #  No existe función predict
    labels_unique = np.unique(etiquetas)
    num_clusters = len(labels_unique)
    centroides = FA.encontrar_centroides(etiquetas, lista_trazas, num_clusters)

    total_time = (time.time_ns()-t0)*1e-9

    return num_clusters, centroides, etiquetas, total_time

def BIRCH(X: np.array(2), threshold: float, branching: int,                   \
          fit_trazas: np.array(3) or None = None):
    """
    Realiza el ajuste al algoritmo Agglomerative Hierarchical Clustering
    y devuelve los valores de interes

    Parameters
    ----------
    X : np.array(2)
        Lista de las trazas a clusterizar.
    lista_trazas : np.array(3)
        Lista con todas las trazas y al vertice perteneciente.
    numcluster_manual : int or None
        Numero de cluster total si es un número se toma el número si es
        'None' se calcula el mejor ajuste con la nota ajustada.
    distance_threshold : float or None
        The linkage distance threshold at or above which clusters will
        not be merged
    branching : int
        Maximum number of CF subclusters in each node.
    fit_trazas : np.array(3) or None
        Lista con las trazas y el vértice al que pertenecen que será predichas.
        Default None.
    Returns
    -------
    num_clusters, centroides, etiquetas, total_time

    Si distance_threslhold --> numero
    --> n_clusters = None y compute_full_tree = True

    """
    t0 = time.time_ns()

    birch = skc.Birch(threshold = threshold, branching_factor = branching,    \
                      n_clusters = None, compute_labels = True, copy = False)
    # n_cluster: None, int, sklearn.cluster
    birch.fit(X)

    etiquetas = birch.labels_

    if isinstance(fit_trazas, np.ndarray):
        etiquetas_predict = birch.predict(fit_trazas[:,1:3])
        etiquetas = np.concatenate((etiquetas, etiquetas_predict))

    centroides = birch.subcluster_centers_
    num_clusters = len(centroides)

    total_time = (time.time_ns()-t0)*1e-9

    return num_clusters, centroides, etiquetas, total_time
