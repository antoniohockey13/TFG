# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:37:50 2023

@author: Antonio
"""
import time
from tabulate import tabulate
import sklearn.cluster as skc
import numpy as np
import matplotlib.pyplot as plt
import GenerarConjuntoVerticesyTrazas as gcvt
import Evaluar
import FuncionesApoyo as FA



num_vertices = 200


trazas_totales = []
# puntos = []
distancia = []
tiempo = []
notaajustada = []
notanorm = []
trazas_bien = []
trazas_mal = []
clusters_bien =  []
clusters_mal = []
num_clusters = []
num_noise = []

for i in range(100):

    lista_vertices, lista_trazas, pos_trazas, num_trazas_en_v, X, num_trazas  \
        = gcvt.VerticesyTrazasAleatorios( num_vertices = num_vertices,        \
                mediatrazas = 70, sigmatrazas = 10, mediaz = 0, sigmaz = 5,   \
                mediat = 0, sigmat = 200, mediar = 0, sigmar = 0.05,          \
                error_z = 0.02, error_t = 10)

    t0 = time.time_ns()
    dbscan = skc.DBSCAN(eps = 0.8, min_samples = 3, metric_params = None,     \
                        algorithm = 'auto', leaf_size = 30, p = None,         \
                        n_jobs = -1)

    dbscan.fit(X)
    etiquetas = dbscan.labels_


    inum_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
    inum_noise = list(etiquetas).count(-1)
    num_clusters.append(inum_clusters)
    num_noise.append(inum_noise)

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


    # plt.plot(lista_vertices[:,1], lista_vertices[:,2], 'x')
    # # plt.errorbar(lista_trazas[:,1], lista_trazas[:,2], xerr = error_z, \
    #               # yerr = error_t, fmt= 'or', linestyle="None")
    # plt.plot(centroides[:,0], centroides[:,1], 'o', c= 'g')
    # # plt.xlim(-10, 10)
    # # plt.ylim(-100, 100)
    # plt.xlabel("$z$/cm")
    # plt.ylabel("$t$/ps")
    # plt.show()


print('Ajuste realizado con: DBSCAN')

tabla = [ [' ', '1', '2', 'media', 'error',],
          ['Trazas OK/Tot', trazas_bien[0]/trazas_totales[0], \
               trazas_bien[1]/trazas_totales[1],                  \
               np.mean(np.array(trazas_bien)/np.array(trazas_totales)),               \
               np.std(np.array(trazas_bien)/np.array(trazas_totales))],
          ['Trazas MAL/Tot', trazas_mal[0]/trazas_totales[0], \
               trazas_mal[1]/trazas_totales[1],                   \
               np.mean(np.array(trazas_mal)/np.array(trazas_totales)),                \
               np.std(np.array(trazas_mal)/np.array(trazas_totales))],
          ['Trazas tot', trazas_totales[0], trazas_totales[1],\
               np.mean(trazas_totales), np.std(trazas_totales)],
          ['Vertices OK', clusters_bien[0], clusters_bien[1],     \
               np.mean(clusters_bien), np.std(clusters_bien)],
          ['Vertices MAL', clusters_mal[0], clusters_mal[1],      \
              np.mean(clusters_mal), np.std(clusters_mal)],
        ['Clusters totales', num_clusters[0], num_clusters[1],\
            np.mean(num_clusters), np.std(num_clusters)] ]
print(tabulate(tabla, headers =  []))
print(f'Vertices totales = {num_vertices}')

# print(f'Puntos:{np.mean(puntos)} +- {np.std(puntos)}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {np.mean(distancia)} +- {np.std(distancia)}')

print(f'Tiempo en ejecutar = {np.mean(tiempo)}+-{np.std(tiempo)}s')
