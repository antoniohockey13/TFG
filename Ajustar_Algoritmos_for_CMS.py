# -*- coding: utf-8 -*-
"""
Created on Mon May 22 18:52:29 2023

@author: Antonio
"""
import matplotlib.pyplot as plt
import numpy as np

from Utilities_Functions import Evaluar
from Utilities_Functions import Read_Data
from Utilities_Functions import Algoritmos



num_evento = str(1)

lista_vertices, lista_trazas, errores, etiquetas_CMS, centroides_CMS,         \
    num_clustersCMS = Read_Data.read_data(                                    \
                              f'Data/SimulationDataCMS_Event{num_evento}.txt')

lista_trazas_medidas, errores_medidos, lista_trazas_no_medidas,               \
    errores_no_medidos = Read_Data.quit_not_measure_vertex(lista_trazas,      \
                                                           errores)

trazas_totales = len(lista_trazas)
num_vertices = len(lista_vertices)
X = np.column_stack((lista_trazas[:,1], lista_trazas[:,2]))

#%% n init K-means
ninit = np.linspace(1, 100, 25, dtype = int)
n_clusters = len(lista_vertices)
notaajustada = []
notanorm = []
notamedia = []
tiempo =  []
num_clusters = []

for in_init in ninit:
    inum_clusters, centroides, etiquetas, total_time = Algoritmos.KMeans(X,   \
                                            lista_trazas,                     \
                                            numcluster_manual = n_clusters,   \
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
plt.savefig(f'KMeans/Gráficas CMS/KMeans n_init_vs_Notas-{num_evento}')
plt.show()


plt.plot(ninit, tiempo,label = 'Tiempo-{num_evento}')
plt.xlabel('n_init')
plt.ylabel('Tiempo')
plt.legend(loc = 'best')
plt.title('K-Means')
plt.savefig(f'KMeans/Gráficas CMS/KMeans n_init_vs_tiempo-{num_evento}')
plt.show()

#%% min_bin_freq Mean Shift

min_bin_freq = np.linspace(1, 32, 16, dtype= int)


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
        Algoritmos.MeanShift(X, quantile = 0.01, n_samples = 299,             \
                              min_bin_freq = imin_bin_freq)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal, cluster_faltan =                       \
            Evaluar.evaluacion_total(lista_trazas, etiquetas, centroides,     \
                                      lista_vertices)


    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/trazas_totales)
    clusters_bien.append(iclusters_bien/num_vertices)
    clusters_mal.append(iclusters_mal/num_vertices)
    num_clusters.append(inum_clusters)
    tiempo.append(total_time)


plt.plot(min_bin_freq, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(min_bin_freq, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(min_bin_freq, notamedia, 'o', c = 'g', label = 'Nota media')
plt.xlabel('min_bin_freq')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig(f'MeanShift/Gráficas CMS/MeanShift min_bin_freq_vs_Puntos-'       \
            f'{num_evento}')
plt.show()

plt.plot(min_bin_freq, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(min_bin_freq, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(min_bin_freq, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.xlabel('min_bin_freq')
plt.ylabel('Num/Tot')
plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig(f'MeanShift/Gráficas CMS/MeanShift min_bin_freq_vs_Ok Mal'        \
            f'-{num_evento}')
plt.show()

plt.plot(min_bin_freq, num_clusters, 'x', c = 'b', label = '')
plt.axhline(y = num_vertices, c = 'r')
plt.xlabel('min_bin_freq')
plt.ylabel('Num clusters')
plt.title('Mean Shift')
plt.savefig(f'MeanShift/Gráficas CMS/MeanShift min_bin_freq_vs_Numclusters'   \
            f'-{num_evento}')
plt.show()


#%% MeanShift Quantile

quantiles = np.linspace(1e-2, 0.1, 50)

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
        Algoritmos.MeanShift(X, min_bin_freq = 1, quantile = iquantile)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal, clusters_faltan =                      \
            Evaluar.evaluacion_total(lista_trazas, etiquetas, centroides,     \
                                      lista_vertices)


    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/trazas_totales)
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
plt.savefig(f'MeanShift/Gráficas CMS/MeanShift Quantile_vs_Puntos-'           \
            f'{num_evento}')
plt.show()

plt.plot(quantiles, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(quantiles, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(quantiles, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.xlabel('quantile')
plt.ylabel('Num/Tot')
plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig(f'MeanShift/Gráficas CMS/MeanShift Quantile_vs_OK-Mal-'           \
            f'{num_evento}')
plt.show()

plt.plot(quantiles, num_clusters, 'x', c = 'b', label = '')
plt.axhline(y = num_vertices, c = 'r')
plt.xlabel('quantiles')
plt.ylabel('Num clusters')
# plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig(f'MeanShift/Gráficas CMS/MeanShift Quantile_vs_Num_clusters-'     \
            f'{num_evento}')
plt.show()

#%% n_samples Mean Shift

n_samples = np.linspace(200, 400, 50, dtype= int)

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
        Algoritmos.MeanShift(X, quantile = 0.01, min_bin_freq = 1,            \
                              n_samples = in_sample)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal, cluster_faltan =                       \
            Evaluar.evaluacion_total(lista_trazas, etiquetas, centroides,     \
                                      lista_vertices)


    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/trazas_totales)
    clusters_bien.append(iclusters_bien/num_vertices)
    clusters_mal.append(iclusters_mal/num_vertices)
    num_clusters.append(inum_clusters)


plt.plot(n_samples, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(n_samples, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(n_samples, notamedia, 'o', c = 'g', label = 'Nota media')
plt.axvline(x = 357, c= 'r')
plt.xlabel('n_samples')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig(f'MeanShift/Gráficas CMS/MeanShift nsamples_vs_Notas-{num_evento}')
plt.show()

plt.plot(n_samples, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(n_samples, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(n_samples, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.axvline(x = 357, c= 'r')
plt.xlabel('n_samples')
plt.ylabel('Num/Tot')
plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig(f'MeanShift/Gráficas CMS/MeanShift nsamples_vs_Ok-Mal-'           \
            f'{num_evento}')
plt.show()

plt.plot(n_samples, num_clusters, 'x', c = 'b', label = '')
plt.axhline(y = num_vertices, c = 'r')
plt.axvline(x = 357, c= 'r')
plt.xlabel('n_samples')
plt.ylabel('Num clusters')
# plt.legend(loc='best')
plt.title('Mean Shift')
plt.savefig(f'MeanShift/Gráficas CMS/MeanShift nsamples_vs_Numclusters-'      \
            f'{num_evento}')
plt.show()

#%% Barrido de epsilon en DBSCAN

epsilons = np.linspace(0.01, 0.1, 25)


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
        Algoritmos.DBSCAN(X, lista_trazas, epsilon = iepsilon, min_samples = 2)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal, clusters_faltan =                      \
            Evaluar.evaluacion_total(lista_trazas, etiquetas, centroides,     \
                                      lista_vertices)


    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/trazas_totales)
    clusters_bien.append(iclusters_bien/num_vertices)
    clusters_mal.append(iclusters_mal/num_vertices)
    num_clusters.append(inum_clusters)


plt.plot(epsilons, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(epsilons, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(epsilons, notamedia, 'o', c = 'g', label = 'Nota media')
plt.axvline(0.035, c = 'r')
plt.xlabel('epsilon')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('DBSCAN')
plt.savefig(f'DBSCAN/Gráficas CMS/DBSCAN epsilon_vs_Notas-{num_evento}')
plt.show()

plt.plot(epsilons, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(epsilons, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(epsilons, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.axvline(0.035, c = 'r')
plt.xlabel('epsilon')
plt.ylabel('Num/Tot')
plt.legend(loc = 'best')
plt.title('DBSCAN')
plt.savefig(f'DBSCAN/Gráficas CMS/DBSCAN epsilon_vs_OK Mal-{num_evento}')
plt.show()

plt.plot(epsilons, num_clusters, 'x', c = 'b', label = '')
plt.axvline(0.035, c = 'r')
plt.axhline(y = num_vertices, c = 'r')
plt.xlabel('epsilon')
plt.ylabel('Num clusters')
plt.title('DBSCAN')
plt.savefig(f'DBSCAN/Gráficas CMS/DBSCAN epsilon_vs_numclusters-{num_evento}')
plt.show()


#%% Barrido min_samples en DBSCAN


min_samples = np.linspace(2, 10, 8, dtype = int)


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
        Algoritmos.DBSCAN(X, lista_trazas, epsilon = 0.035,                   \
                          min_samples = imin_sample)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal, clusters_faltan =                      \
            Evaluar.evaluacion_total(lista_trazas, etiquetas, centroides,     \
                                     lista_vertices)


    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/trazas_totales)
    clusters_bien.append(iclusters_bien/num_vertices)
    clusters_mal.append(iclusters_mal/num_vertices)
    num_clusters.append(inum_clusters)


plt.plot(min_samples, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(min_samples, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(min_samples, notamedia, 'o', c = 'g', label = 'Nota media')
plt.axvline(2, c = 'r')
plt.xlabel('min samples')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('DBSCAN')
plt.savefig(f'DBSCAN/Gráficas CMS/DBSCAN minsamples_vs_Notas-{num_evento}')
plt.show()

plt.plot(min_samples, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(min_samples, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(min_samples, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.axvline(2, c = 'r')
plt.xlabel('min samples')
plt.ylabel('Num/Tot')
plt.legend(loc='best')
plt.title('DBSCAN')
plt.savefig(f'DBSCAN/Gráficas CMS/DBSCAN minsamples_vs_OK-Mal-{num_evento}')
plt.show()

plt.plot(min_samples, num_clusters, 'x', c = 'b', label = '')
plt.axvline(2, c = 'r')
plt.axhline(y = num_vertices, c = 'r')
plt.xlabel('min samples')
plt.ylabel('Num clusters')
# plt.legend(loc='best')
plt.title('DBSCAN')
plt.savefig(f'DBSCAN/Gráficas CMS/DBSCAN minsamples_vs_numclusters-{num_evento}')
plt.show()

#%% AHC distance_threshold

distances = np.linspace(0.2, 1, 30)


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
        Algoritmos.AHC(X, lista_trazas, distance_threshold = idistance)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal, clusters_faltan =                      \
            Evaluar.evaluacion_total(lista_trazas, etiquetas, centroides,     \
                                     lista_vertices)


    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/trazas_totales)
    clusters_bien.append(iclusters_bien/num_vertices)
    clusters_mal.append(iclusters_mal/num_vertices)
    num_clusters.append(inum_clusters)
    time.append(total_time)


plt.plot(distances, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(distances, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(distances, notamedia, 'o', c = 'g', label = 'Nota media')
plt.axvline(x = 0.45, c = 'r')
plt.xlabel('distances threshold')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('AHC')
plt.savefig(f'AHC/Gráficas CMS/AHC distance_threshold_vs_notas-{num_evento}')
plt.show()

plt.plot(distances, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(distances, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(distances, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.axvline(x = 0.45, c = 'r')
plt.xlabel('distances threshold')
plt.ylabel('Num/Tot')
plt.legend(loc='best')
plt.title('AHC')
plt.savefig(f'AHC/Gráficas CMS/AHC distance_threshold_vs_OK-Mal-{num_evento}')
plt.show()

plt.plot(distances, num_clusters, 'x', c = 'b', label = '')
plt.axvline(x = 0.45, c = 'r')
plt.axhline(y = num_vertices, c = 'r')
plt.xlabel('distances threshold')
plt.ylabel('Num clusters')
# plt.legend(loc='best')
plt.title('AHC')
plt.savefig(f'AHC/Gráficas CMS/AHC threshold_vs_numclusters-{num_evento}')
plt.show()


#%% threshold BIRCH
thresholds = np.linspace(0.05, 0.2, 30)


notaajustada = []
notanorm = []
notamedia = []
distancia = []
trazas_bien = []
clusters_bien = []
clusters_mal = []
num_clusters = []
time = []

for ithreshold in thresholds:
    inum_clusters, centroides, etiquetas, total_time =                        \
        Algoritmos.BIRCH(X, ithreshold, branching = 80)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal, clusters_faltan =                      \
            Evaluar.evaluacion_total(lista_trazas, etiquetas, centroides,     \
                                     lista_vertices)


    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/trazas_totales)
    clusters_bien.append(iclusters_bien/num_vertices)
    clusters_mal.append(iclusters_mal/num_vertices)
    num_clusters.append(inum_clusters)
    time.append(total_time)


plt.plot(thresholds, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(thresholds, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(thresholds, notamedia, 'o', c = 'g', label = 'Nota media')
plt.axvline(x = 0.11, c = 'r')
plt.xlabel('Threshold')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('Birch')
plt.savefig('BIRCH/Gráficas CMS/Birch distance_threshold_vs_notas'            \
            f'-{num_evento}')
plt.show()

plt.plot(thresholds, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(thresholds, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(thresholds, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.axvline(x = 0.11, c = 'r')
plt.xlabel('Threshold')
plt.ylabel('Num/Tot')
plt.legend(loc='best')
plt.title('Birch')
plt.savefig('BIRCH/Gráficas CMS/Birch distance_threshold_vs_OK-Mal-'          \
            f'{num_evento}')
plt.show()

plt.plot(thresholds, num_clusters, 'x', c = 'b', label = '')
plt.axvline(x = 0.11, c = 'r')
plt.axhline(y = num_vertices, c = 'r')
plt.xlabel('Threshold')
plt.ylabel('Num clusters')
# plt.legend(loc='best')
plt.title('Birch')
plt.savefig(f'BIRCH/Gráficas CMS/Birch threshold_vs_numclusters-{num_evento}')
plt.show()

#%% branching ratio BIRCH
branchings = np.linspace(2, 100, 30, dtype = int)


notaajustada = []
notanorm = []
notamedia = []
distancia = []
trazas_bien = []
clusters_bien = []
clusters_mal = []
num_clusters = []
time = []

for ibranching in branchings:
    inum_clusters, centroides, etiquetas, total_time =                        \
        Algoritmos.BIRCH(X, threshold = 0.11, branching = ibranching)

    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal, clusters_faltan =                      \
            Evaluar.evaluacion_total(lista_trazas, etiquetas, centroides,     \
                                     lista_vertices)
    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)
    distancia.append(idistancia)
    trazas_bien.append(itrazas_bien/trazas_totales)
    clusters_bien.append(iclusters_bien/num_vertices)
    clusters_mal.append(iclusters_mal/num_vertices)
    num_clusters.append(inum_clusters)
    time.append(total_time)


plt.plot(branchings, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(branchings, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(branchings, notamedia, 'o', c = 'g', label = 'Nota media')
plt.axvline(x = 40, c = 'r')
plt.xlabel('Branching factor')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('Birch')
plt.savefig(f'BIRCH/Gráficas CMS/Birch branching_vs_notas-{num_evento}')
plt.show()

plt.plot(branchings, trazas_bien, 'x', c = 'b', label = 'Trazas OK')
plt.plot(branchings, clusters_bien, 'x', c = 'r', label = 'Clusters OK')
plt.plot(branchings, clusters_mal, 'o', c = 'g', label = 'Clusters mal')
plt.axvline(x = 40, c = 'r')
plt.xlabel('Branching factor')
plt.ylabel('Num/Tot')
plt.legend(loc='best')
plt.title('Birch')
plt.savefig(f'BIRCH/Gráficas CMS/Birch branching_vs_OK-Mal-{num_evento}')
plt.show()

plt.plot(branchings, num_clusters, 'x', c = 'b', label = '')
plt.axvline(x = 40, c = 'r')
plt.axhline(y = num_vertices, c ='r')
plt.xlabel('Branching factor')
plt.ylabel('Num clusters')
# plt.legend(loc='best')
plt.title('Birch')
plt.savefig(f'BIRCH/Gráficas CMS/Birch branching_vs_numclusters-{num_evento}')
plt.show()
