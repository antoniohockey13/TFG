# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:05:15 2023

@author: Antonio
"""
import time
from tabulate import tabulate
import sklearn.mixture as skm
import numpy as np
import matplotlib.pyplot as plt
import GenerarConjuntoVerticesyTrazas as gcvt
import Evaluar
import FuncionesApoyo as FA

num_vertices = 200


trazas_totales = []
# puntos = []
distancia = []
inercia = []
tiempo = []
notaajustada = []
notanorm = []
trazas_bien = []
trazas_mal = []
clusters_bien =  []
clusters_mal = []
num_clusters = []

for i in range(100):
    print(i)
    lista_vertices, lista_trazas, pos_trazas, num_trazas_en_v, X, num_trazas  \
        = gcvt.VerticesyTrazasAleatorios( num_vertices = num_vertices,        \
                mediatrazas = 70, sigmatrazas = 10, mediaz = 0, sigmaz = 5,   \
                mediat = 0, sigmat = 200, mediar = 0, sigmar = 0.05,          \
                error_z = 0.02, error_t = 10)

    t0 = time.time_ns()

    inum_clusters = 200
    num_clusters.append(inum_clusters)

    em_gmm = skm.GaussianMixture(n_components = inum_clusters, n_init = 1,                  \
                                 init_params = 'kmeans', warm_start = True)
    em_gmm.fit(X)


    etiquetas = em_gmm.predict(X)
    centroides = FA.encontrar_centroides(etiquetas, lista_trazas,             \
                                         inum_clusters)
    t1 = time.time_ns()

    ctv = Evaluar.cluster_to_vertex(centroides, lista_vertices)

    idistancia = Evaluar.distancia_media(centroides, lista_vertices, ctv)

    # ipuntos = Evaluar.evaluar(lista_trazas, etiquetas, ctv, num_trazas)

    inotaajustada, inotanorm = Evaluar.evaluacion(lista_trazas, etiquetas)

    itrazas_bien, itrazas_mal, iclusters_bien, iclusters_mal =\
        Evaluar.tabla_trazas(lista_trazas, etiquetas, num_trazas_en_v, ctv)

    # puntos.append(ipuntos)
    distancia.append(idistancia)
    tiempo.append((t1-t0)*1e-9)
    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)

    trazas_bien.append(itrazas_bien)
    trazas_mal.append(itrazas_mal)
    clusters_bien.append(iclusters_bien)
    clusters_mal.append(iclusters_mal)

    trazas_totales.append(num_trazas)

    # plt.plot(lista_vertices[:,1], lista_vertices[:,2], 'x', label = 'Vertices')
    # plt.plot(lista_trazas[:,1], lista_trazas[:,2], 'o',c = 'r', label = 'Trazas')
    # # plt.errorbar(lista_trazas[:,1], lista_trazas[:,2], xerr = error_z, \
    #                 # yerr = error_t, fmt= '.r', linestyle="None")
    # plt.plot(centroides[:,0], centroides[:,1], 'o', c= 'g', label = 'Clusters')
    # plt.legend(loc = 'best')
    # # plt.xlim(-10, 10)
    # # plt.ylim(-1, 1)
    # plt.xlabel("$z/\sigma_z$")
    # plt.ylabel("$t/sigma_t$/")
    # plt.show()

print('Ajuste realizado con: EM-GMM')


tabla = [ [' ', '1', '2', 'media', 'error',],
          ['Trazas OK/Tot', trazas_bien[0]/trazas_totales[0],                 \
                trazas_bien[1]/trazas_totales[1],                             \
                np.mean(np.array(trazas_bien)/np.array(trazas_totales)),               \
                np.std(np.array(trazas_bien)/np.array(trazas_totales))],
          ['Trazas MAL/Tot', trazas_mal[0]/trazas_totales[0],                 \
                trazas_mal[1]/trazas_totales[1],                              \
                np.mean(np.array(trazas_mal)/np.array(trazas_totales)),                \
                np.std(np.array(trazas_mal)/np.array(trazas_totales))],
          ['Trazas tot', trazas_totales[0], trazas_totales[1],                \
                np.mean(trazas_totales), np.std(trazas_totales)],
          ['Vertices OK', clusters_bien[0], clusters_bien[1],                 \
                np.mean(clusters_bien), np.std(clusters_bien)],
          ['Vertices MAL', clusters_mal[0], clusters_mal[1],                  \
              np.mean(clusters_mal), np.std(clusters_mal)],
        ['Clusters totales', num_clusters[0], num_clusters[1],                \
            np.mean(num_clusters), np.std(num_clusters)] ]
print(tabulate(tabla, headers =  []))
print(f'Vertices totales = {num_vertices}')

# print(f'Puntos:{np.mean(puntos)} +- {np.std(puntos)}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número ' \
        f'vértices): {np.mean(distancia)} +- {np.std(distancia)}')

print(f'Tiempo en ejecutar = {np.mean(tiempo)}+-{np.std(tiempo)}s')
