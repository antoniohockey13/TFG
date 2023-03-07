# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 21:18:49 2023

@author: Antonio
"""
import sklearn.cluster as skc
import numpy as np
import matplotlib.pyplot as plt
# import GenerarConjuntoVerticesyTrazas as gcvt
import Evaluar

# error_z = 0.02
# error_t = 10
# num_vertices = 200



# lista_vertices, lista_trazas, pos_trazas, num_trazas_en_v, X, num_trazas      \
#     = gcvt.VerticesyTrazasAleatorios( num_vertices = num_vertices,            \
#             mediatrazas = 70, sigmatrazas = 10, mediaz = 0, sigmaz = 5,       \
#             mediat = 0, sigmat = 200, mediar = 0, sigmar = 0.05,              \
#             error_z = error_z, error_t =error_t)

# num_trazas = len(lista_trazas)


def kmeans_num_clusters(X,  lista_trazas, num_min = 190, num_max = 210,       \
                        step = 21):

    notaajustada = []
    notanorm = []
    notamedia = []

    clusters = np.linspace(num_min, num_max, step, dtype = int)
    for inum_cluster in clusters:

        kmeans =skc.KMeans(n_clusters = inum_cluster, init = 'k-means++',         \
                           max_iter = 300, n_init = 10)

        kmeans.fit(X)

        etiquetas = kmeans.labels_
        inotaajustada, inotanorm = Evaluar.evaluacion(lista_trazas, etiquetas)
        notaajustada.append(inotaajustada)
        notanorm.append(inotanorm)
        notamedia.append((inotaajustada**2+inotanorm**2)/2)

    # plt.plot(clusters, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
    # plt.plot(clusters, notanorm, 'x', c = 'r', label = 'Nota normal')
    # plt.plot(clusters, notamedia, 'o', c = 'g', label = 'Nota media')
    # plt.xlabel('Num clusters')
    # plt.ylabel('Puntos')
    # plt.legend(loc='best')
    # plt.show()

    maximo_notaajustada = max(notaajustada)
    pos_maximo_notaajustada = notaajustada.index(maximo_notaajustada)
    num_cluster_ajustada = clusters[pos_maximo_notaajustada]
    # print(maximo_notaajustada, num_cluster_ajustada)

    maximo_notanorm = max(notanorm)
    pos_maximo_notanorm = notanorm.index(maximo_notanorm)
    num_cluster_norm = clusters[pos_maximo_notanorm]
    # print(maximo_notanorm, num_cluster_norm)

    maximo_notamedia = max(notamedia)
    pos_maximo_notamedia = notamedia.index(maximo_notamedia)
    num_cluster_media = clusters[pos_maximo_notamedia]
    # print(maximo_notamedia, num_cluster_media)

    print(num_cluster_ajustada, num_cluster_media, num_cluster_norm)
    return num_cluster_ajustada

    # return(num_cluster_ajustada, num_cluster_norm, num_cluster_media)
