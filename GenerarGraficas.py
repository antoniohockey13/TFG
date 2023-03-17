# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:50:12 2023

@author: Antonio
Generar gráficas
"""

import numpy as np
import matplotlib.pyplot as plt
import GenerarConjuntoVerticesyTrazas as gcvt
import Algoritmos
import Evaluar


#%% Generar vertices y trazas
num_vertices = 200
lista_vertices, lista_trazas, pos_trazas, num_trazas_en_v, X, num_trazas      \
    = gcvt.VerticesyTrazasAleatorios( num_vertices,            \
            mediatrazas = 70, sigmatrazas = 10, mediaz = 0, sigmaz = 5,       \
            mediat = 0, sigmat = 200, mediar = 0, sigmar = 0.05,              \
            error_z = 0.02, error_t = 10)

num_trazas = len(lista_trazas)

# %% Num cluster en K-Means


clusters1 = np.linspace(170, 194, 9, dtype = int)
clusters2 = np.linspace(195, 205, 11, dtype = int)
clusters3 = np.linspace(206, 230, 9, dtype = int)

clusters = np.concatenate((clusters1, clusters2, clusters3))

notaajustada = []
notanorm = []
notamedia = []


for inum_cluster in clusters:
    inum_clusters, centroides, etiquetas, total_time = Algoritmos.KMeans(X,   \
                                            lista_trazas,                     \
                                            numcluster_manual = inum_cluster, \
                                            n_init = 10, tol = 1e-4)
    inotaajustada, inotanorm = Evaluar.evaluacion(lista_trazas, etiquetas)
    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)


plt.plot(clusters, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(clusters, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(clusters, notamedia, 'o', c = 'g', label = 'Nota media')
plt.xlabel('Num clusters')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('K-Means')
plt.savefig('Gráficas/KMeans numclusters_vs_Notas 1')
plt.show()

#%% n init K-means
ninit = np.linspace(1, 100, 25, dtype = int)

notaajustada = []
notanorm = []
notamedia = []
tiempo =  []
num_clusters = []

for in_init in ninit:
    inum_clusters, centroides, etiquetas, total_time = Algoritmos.KMeans(X,   \
                                            lista_trazas,                     \
                                            numcluster_manual = 200,
                                            n_init = in_init, tol = 1e-4)

    inotaajustada, inotanorm = Evaluar.evaluacion(lista_trazas, etiquetas)
    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    tiempo.append(total_time)
    num_clusters.append(inum_clusters)


plt.plot(ninit, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(ninit, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(ninit, notamedia, 'o', c = 'g', label = 'Nota media')
plt.xlabel('n_init')
plt.ylabel('Puntos')
plt.legend(loc = 'best')
plt.title('K-Means')
plt.savefig('Gráficas/KMeans n_init_vs_Notas')
plt.show()

# plt.plot(ninit, num_clusters, 'x', c = 'b', label = '')
# plt.xlabel('n_init')
# plt.ylabel('Num clusters')
# # plt.legend(loc = 'best')
# plt.title('Mean Shift')
# plt.savefig('Gráficas/MeanShift n_init_vs_Num_clusters')
# plt.show()

plt.plot(ninit, tiempo,label = 'Tiempo')
plt.xlabel('n_init')
plt.ylabel('Tiempo')
plt.legend(loc = 'best')
plt.title('K-Means')
plt.savefig('Gráficas/KMeans n_init_vs_tiempo')
plt.show()

#%% MeanShift Quantile

quantiles = np.linspace(1e-2, 0.1, 10)

notaajustada = []
notanorm = []
notamedia = []
distancia = []
trazas_bien = []
clusters_bien = []
clusters_mal = []
num_clusters = []

for iquantile in quantiles:
    inum_clusters, centroides, etiquetas, total_time =                        \
        Algoritmos.MeanShift(X, quantile = iquantile)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal = Evaluar.evaluacion_total(lista_trazas,\
                        etiquetas, centroides, lista_vertices, num_trazas_en_v)


    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/num_trazas)
    clusters_bien.append(iclusters_bien/num_vertices)
    clusters_mal.append(iclusters_mal/num_vertices)
    num_clusters.append(inum_clusters)


plt.plot(quantiles, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(quantiles, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(quantiles, notamedia, 'o', c = 'g', label = 'Nota media')
plt.xlabel('quantile')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig('Gráficas/MeanShift Quantile_vs_Puntos 1')
plt.show()

plt.plot(quantiles, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(quantiles, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(quantiles, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.xlabel('quantile')
plt.ylabel('Num/Tot')
plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig('Gráficas/MeanShift Quantile_vs_OK-Mal 1')
plt.show()

plt.plot(quantiles, num_clusters, 'x', c = 'b', label = '')
plt.xlabel('quantiles')
plt.ylabel('Num clusters')
# plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig('Gráficas/MeanShift Quantile_vs_Num_clusters 1')
plt.show()

#%% n_samples Mean Shift

n_samples = np.linspace(280, 320, 50, dtype= int)


notaajustada = []
notanorm = []
notamedia = []
distancia = []
trazas_bien = []
clusters_bien = []
clusters_mal = []
num_clusters = []

for in_sample in n_samples:
    inum_clusters, centroides, etiquetas, total_time =                        \
        Algoritmos.MeanShift(X, quantile = 0.01, n_samples = in_sample)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal = Evaluar.evaluacion_total(lista_trazas,\
                        etiquetas, centroides, lista_vertices, num_trazas_en_v)


    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/num_trazas)
    clusters_bien.append(iclusters_bien/num_vertices)
    clusters_mal.append(iclusters_mal/num_vertices)
    num_clusters.append(inum_clusters)


plt.plot(n_samples, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(n_samples, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(n_samples, notamedia, 'o', c = 'g', label = 'Nota media')
plt.axvline(x = 299, c = 'r')
plt.xlabel('n_samples')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig('Gráficas/MeanShift nsamples_vs_Notas 1')
plt.show()

plt.plot(n_samples, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(n_samples, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(n_samples, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.axvline(x = 299, c = 'r')
plt.xlabel('n_samples')
plt.ylabel('Num/Tot')
plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig('Gráficas/MeanShift nsamples_vs_Ok-Mal 1')
plt.show()

plt.plot(n_samples, num_clusters, 'x', c = 'b', label = '')
plt.axvline(x = 299, c = 'r')
plt.xlabel('n_samples')
plt.ylabel('Num clusters')
# plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig('Gráficas/MeanShift nsamples_vs_Numclusters 1')
plt.show()

#%% min_bin_freq Mean Shift

min_bin_freq = np.linspace(25, 40, 16, dtype= int)


notaajustada = []
notanorm = []
notamedia = []
distancia = []
trazas_bien = []
clusters_bien = []
clusters_mal = []
num_clusters = []
tiempo = []

for imin_bin_freq in min_bin_freq:
    inum_clusters, centroides, etiquetas, total_time =                        \
        Algoritmos.MeanShift(X, quantile = 0.01, n_samples = 299,       \
                             min_bin_freq = imin_bin_freq)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal = Evaluar.evaluacion_total(lista_trazas,\
                        etiquetas, centroides, lista_vertices, num_trazas_en_v)


    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/num_trazas)
    clusters_bien.append(iclusters_bien/num_vertices)
    clusters_mal.append(iclusters_mal/num_vertices)
    num_clusters.append(inum_clusters)
    tiempo.append(total_time)


plt.plot(min_bin_freq, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(min_bin_freq, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(min_bin_freq, notamedia, 'o', c = 'g', label = 'Nota media')
plt.axvline(x = 31, c = 'r')
plt.xlabel('min_bin_freq')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig('Gráficas/MeanShift min_bin_freq_vs_Puntos')
plt.show()

plt.plot(min_bin_freq, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(min_bin_freq, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(min_bin_freq, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.axvline(x = 31, c = 'r')
plt.xlabel('min_bin_freq')
plt.ylabel('Num/Tot')
plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig('Gráficas/MeanShift min_bin_freq_vs_Ok Mal')
plt.show()

plt.plot(min_bin_freq, num_clusters, 'x', c = 'b', label = '')
plt.axvline(x = 31, c = 'r')
plt.xlabel('min_bin_freq')
plt.ylabel('Num clusters')
plt.title('Mean Shift')
plt.savefig('Gráficas/MeanShift min_bin_freq_vs_Numclusters')
plt.show()

plt.plot(min_bin_freq, tiempo)
plt.axvline(x = 31, c = 'r')
plt.xlabel('min_bin_freq')
plt.ylabel('Time')
plt.title('Mean Shift')
plt.savefig('Gráficas/MeanShift min_bin_freq_vs_time')
plt.show()


#%% Barrido de epsilon en DBSCAN

epsilons = np.linspace(1, 2, 50)


notaajustada = []
notanorm = []
notamedia = []
distancia = []
trazas_bien = []
clusters_bien = []
clusters_mal = []
num_clusters = []

for iepsilon in epsilons:
    inum_clusters, centroides, etiquetas, total_time, num_noise =             \
        Algoritmos.DBSCAN(X, lista_trazas, epsilon = iepsilon, min_samples = 3)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal = Evaluar.evaluacion_total(lista_trazas,\
                        etiquetas, centroides, lista_vertices, num_trazas_en_v)


    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/num_trazas)
    clusters_bien.append(iclusters_bien/num_vertices)
    clusters_mal.append(iclusters_mal/num_vertices)
    num_clusters.append(inum_clusters)


plt.plot(epsilons, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(epsilons, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(epsilons, notamedia, 'o', c = 'g', label = 'Nota media')
plt.axvline(1.45)
plt.xlabel('epsilon')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('DBSCAN')
plt.savefig('Gráficas/DBSCAN epsilon_vs_Notas 3')
plt.show()

plt.plot(epsilons, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(epsilons, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(epsilons, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.axvline(1.45)
plt.xlabel('epsilon')
plt.ylabel('Num/Tot')
plt.legend(loc = 'best')
plt.title('DBSCAN')
plt.savefig('Gráficas/DBSCAN epsilon_vs_OK Mal 3')
plt.show()

plt.plot(epsilons, num_clusters, 'x', c = 'b', label = '')
plt.axvline(1.45)
plt.xlabel('epsilon')
plt.ylabel('Num clusters')
plt.title('DBSCAN')
plt.savefig('Gráficas/DBSCAN epsilon_vs_numclusters 3')
plt.show()


#%% Barrido min_samples en DBSCAN


min_samples = np.linspace(1, 25, 24, dtype = int)


notaajustada = []
notanorm = []
notamedia = []
distancia = []
trazas_bien = []
clusters_bien = []
clusters_mal = []
num_clusters = []

for imin_sample in min_samples:
    inum_clusters, centroides, etiquetas, total_time, num_noise =             \
        Algoritmos.DBSCAN(X, lista_trazas, epsilon = 1.45,                     \
                          min_samples = imin_sample)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal = Evaluar.evaluacion_total(lista_trazas,\
                        etiquetas, centroides, lista_vertices, num_trazas_en_v)


    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/num_trazas)
    clusters_bien.append(iclusters_bien/num_vertices)
    clusters_mal.append(iclusters_mal/num_vertices)
    num_clusters.append(inum_clusters)


plt.plot(min_samples, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(min_samples, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(min_samples, notamedia, 'o', c = 'g', label = 'Nota media')
plt.axvline(2)
plt.xlabel('min samples')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('DBSCAN')
plt.savefig('Gráficas/DBSCAN minsamples_vs_Notas 1')
plt.show()

plt.plot(min_samples, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(min_samples, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(min_samples, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.axvline(2)
plt.xlabel('min samples')
plt.ylabel('Num/Tot')
plt.legend(loc='best')
plt.title('DBSCAN')
plt.savefig('Gráficas/DBSCAN minsamples_vs_OK-Mal 1')
plt.show()

plt.plot(min_samples, num_clusters, 'x', c = 'b', label = '')
plt.axvline(2)
plt.xlabel('min samples')
plt.ylabel('Num clusters')
# plt.legend(loc='best')
plt.title('DBSCAN')
plt.savefig('Gráficas/DBSCAN minsamples_vs_numclusters 1')
plt.show()


#%% Barrido leaf_size en DBSCAN

leaf_size = np.linspace(5, 15, 9, dtype = int)


notaajustada = []
notanorm = []
notamedia = []
distancia = []
trazas_bien = []
clusters_bien = []
clusters_mal = []
num_clusters = []
time = []

for ileaf in leaf_size:
    inum_clusters, centroides, etiquetas, total_time, num_noise =             \
        Algoritmos.DBSCAN(X, lista_trazas, epsilon = 1.45, leaf_size = ileaf)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal = Evaluar.evaluacion_total(lista_trazas,\
                        etiquetas, centroides, lista_vertices, num_trazas_en_v)


    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/num_trazas)
    clusters_bien.append(iclusters_bien/num_vertices)
    clusters_mal.append(iclusters_mal/num_vertices)
    num_clusters.append(inum_clusters)
    time.append(total_time)


plt.plot(leaf_size, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(leaf_size, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(leaf_size, notamedia, 'o', c = 'g', label = 'Nota media')
plt.xlabel('leaf_size')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('DBSCAN')
plt.savefig('Gráficas/DBSCAN leafsize_vs_notas 1')
plt.show()

plt.plot(leaf_size, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(leaf_size, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(leaf_size, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.xlabel('leaf_size')
plt.ylabel('Num/Tot')
plt.legend(loc='best')
plt.title('DBSCAN')
plt.savefig('Gráficas/DBSCAN leafsize_vs_OK-Mal 1')
plt.show()

plt.plot(leaf_size, num_clusters, 'x', c = 'b', label = '')
plt.xlabel('leaf_size')
plt.ylabel('Num clusters')
# plt.legend(loc='best')
plt.title('DBSCAN')
plt.savefig('Gráficas/DBSCAN leafsize_vs_numclusters 1')
plt.show()

plt.plot(leaf_size, time, c = 'b', label = '')
plt.xlabel('leaf_size')
plt.ylabel('Time/s')
# plt.legend(loc='best')
plt.title('DBSCAN')
plt.savefig('Gráficas/DBSCAN leafsive_vs_tiempo')

#%% Numero Clusters EM-GMM

clusters1 = np.linspace(170, 194, 9, dtype = int)
clusters2 = np.linspace(195, 205, 11, dtype = int)
clusters3 = np.linspace(206, 250, 15, dtype = int)

clusters = np.concatenate((clusters1, clusters2, clusters3))

notaajustada = []
notanorm = []
notamedia = []

for inum_cluster in clusters:
    inum_clusters, centroides, etiquetas, total_time = Algoritmos.EM_GMM(X,   \
                                            lista_trazas,                     \
                                            numcluster_manual = inum_cluster)
    inotaajustada, inotanorm = Evaluar.evaluacion(lista_trazas, etiquetas)
    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)


plt.plot(clusters, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(clusters, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(clusters, notamedia, 'o', c = 'g', label = 'Nota media')
plt.xlabel('Num clusters')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('EM-GMM')
plt.savefig('Gráficas/EM-GMM numclusters_vs_Notas 1')
plt.show()

#%% AHC distance_threshold

distances = np.linspace(15, 40, 30)


notaajustada = []
notanorm = []
notamedia = []
distancia = []
trazas_bien = []
clusters_bien = []
clusters_mal = []
num_clusters = []
time = []

for idistance in distances:
    inum_clusters, centroides, etiquetas, total_time =                         \
        Algoritmos.AHC(X, lista_trazas, numcluster_manual = None,             \
                       distance_threshold = idistance)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal = Evaluar.evaluacion_total(lista_trazas,\
                        etiquetas, centroides, lista_vertices, num_trazas_en_v)


    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/num_trazas)
    clusters_bien.append(iclusters_bien/num_vertices)
    clusters_mal.append(iclusters_mal/num_vertices)
    num_clusters.append(inum_clusters)
    time.append(total_time)


plt.plot(distances, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(distances, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(distances, notamedia, 'o', c = 'g', label = 'Nota media')
plt.xlabel('distances threshold')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('AHC')
plt.savefig('Gráficas/AHC distance_threshold_vs_notas 1')
plt.show()

plt.plot(distances, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(distances, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(distances, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.xlabel('distances threshold')
plt.ylabel('Num/Tot')
plt.legend(loc='best')
plt.title('AHC')
plt.savefig('Gráficas/AHC distance_threshold_vs_OK-Mal 1')
plt.show()

plt.plot(distances, num_clusters, 'x', c = 'b', label = '')
plt.xlabel('distances threshold')
plt.ylabel('Num clusters')
# plt.legend(loc='best')
plt.title('AHC')
plt.savefig('Gráficas/AHC threshold_vs_numclusters 1')
plt.show()

plt.plot(distances, time, c = 'b', label = '')
plt.xlabel('distances threshold')
plt.ylabel('Time/s')
# plt.legend(loc='best')
plt.title('AHC')
plt.savefig('Gráficas/AHC distance_threshold_vs_tiempo 1')
