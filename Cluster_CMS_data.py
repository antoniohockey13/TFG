# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:48:15 2023

@author: Antonio
"""
from tabulate import tabulate
import numpy as np

from Utilities_Functions import Evaluar
from Utilities_Functions import Grafica_Clusters
from Utilities_Functions import Read_Data
from Utilities_Functions import Algoritmos_for_CMS_data as Algorithm



num_evento = str(0)
# name = f"Data/DataCMS_momentum{num_evento}.txt"
name = f"Data/SimulationDataCMS_Event{num_evento}.txt"

lista_vertices, lista_trazas, errores, etiquetas_CMS, centroides_CMS,         \
    num_clustersCMS, momentum = Read_Data.read_data(name, pt = 0)

lista_trazas_medidas, errores_medidos, lista_trazas_no_medidas,               \
    errores_no_medidos = Read_Data.quit_not_measure_vertex(lista_trazas,      \
                                                           errores)

trazas_totales = len(lista_trazas)
num_vertices = len(lista_vertices)

Grafica_Clusters.grafica_colores_cluster(lista_trazas, lista_trazas[:,0],'CMS')

#%% Evaluar resultados CMS
notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, vertices_faltan = Evaluar.evaluacion_total(lista_trazas,    \
                                etiquetas_CMS, centroides_CMS, lista_vertices)

Grafica_Clusters.grafica_centroides_vertices(lista_vertices, centroides_CMS,  \
                                             'CMS')


print('\n Evaluacion resultados CMS: ')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Vertices faltan', vertices_faltan],
          ['Clusters totales', num_clustersCMS] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

#%% K-Means sin eliminar 0
num_clusters = len(lista_vertices)
notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
      clusters_mal, vertices_faltan, total_time, num_clusters =               \
          Algorithm.KMeans(lista_trazas = lista_trazas,                       \
                           lista_vertices = lista_vertices, fit_trazas = None,\
                           num_clusters = num_clusters, sample_weight = None, \
                           error_predict = None, n_init = 10, tol = 1e-6,     \
                           graficas = True)
print('\n Ajuste realizado con: KMeans sin eliminar 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Clusters totales', num_clusters],
          ['Vertices faltan', vertices_faltan]]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')

#%% K-Means eliminando trazas no detectadas temporalmente
numcluster_manual = 215

notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
      clusters_mal, vertices_faltan, total_time, num_clusters =               \
          Algorithm.KMeans(lista_trazas = lista_trazas_medidas,               \
                           lista_vertices = lista_vertices,                   \
                           fit_trazas = lista_trazas_no_medidas,              \
                num_clusters = numcluster_manual, sample_weight = None,       \
                error_predict = None, n_init = 10, tol = 1e-6, graficas = True)

print('\n Ajuste realizado con: KMeans eliminando 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Clusters totales', num_clusters],
          ['Vertices faltan', vertices_faltan] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')
#%% MeanShift sin eliminar 0
notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
      clusters_mal, vertices_faltan, total_time, num_clusters =               \
          Algorithm.MeanShift(lista_trazas = lista_trazas,                    \
                               lista_vertices = lista_vertices,                \
                              fit_trazas = None, quantile = 1e-2,             \
                              n_samples = 357, min_bin_freq = 1,             \
                              graficas = True)
print('\n Ajuste realizado con: MeanShift sin eliminar 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Clusters totales', num_clusters],
          ['Vertices faltan', vertices_faltan] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')

#%% MeanShift eliminando 0
notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
      clusters_mal, vertices_faltan, total_time, num_clusters =               \
          Algorithm.MeanShift(lista_trazas = lista_trazas_medidas,            \
                              lista_vertices = lista_vertices,                \
                              fit_trazas = lista_trazas_no_medidas,           \
                              quantile = 1e-2, n_samples = 299,               \
                              min_bin_freq = 31, graficas = True)

print('\n Ajuste realizado con: MeanShift eliminando 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Clusters totales', num_clusters],
          ['Vertices faltan', vertices_faltan] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')
#%% DBSCAN sin eliminar 0

notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
      clusters_mal, vertices_faltan, total_time, num_clusters =               \
          Algorithm.DBSCAN(lista_trazas = lista_trazas_medidas,               \
                              lista_vertices = lista_vertices,                \
                              fit_trazas = None, sample_weight = None,        \
                              error_predict = None, epsilon = 0.035,          \
                              min_samples = 2, leaf_size = 12, graficas = True)


print('\n Ajuste realizado con: DBSCAN sin eliminar 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Clusters totales', num_clusters],
          ['Vertices faltan', vertices_faltan] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')

#%% EM-GMM sin eliminar 0
num_clusters = len(lista_vertices)
notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
      clusters_mal, vertices_faltan, total_time, num_clusters =               \
          Algorithm.EM_GMM(lista_trazas = lista_trazas,                       \
                              lista_vertices = lista_vertices,                \
                              fit_trazas = None, sample_weight = None,        \
                              num_clusters = num_clusters, graficas =True)

print('\n Ajuste realizado con: EM_GMM sin eliminar 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Clusters totales', num_clusters],
          ['Vertices faltan', vertices_faltan] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')

#%% EM-GMM eliminando 0
notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
      clusters_mal, vertices_faltan, total_time, num_clusters =               \
          Algorithm.EM_GMM(lista_trazas = lista_trazas_medidas,               \
                              lista_vertices = lista_vertices,                \
                              fit_trazas = lista_trazas_no_medidas,           \
                              sample_weight = None, num_clusters = 215,       \
                              graficas = True)

print('\n Ajuste realizado con: EM_GMM con eliminar 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Clusters totales', num_clusters],
          ['Vertices faltan', vertices_faltan] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')

#%% AHC sin eliminar 0
notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
      clusters_mal, vertices_faltan, total_time, num_clusters =               \
          Algorithm.AHC(lista_trazas = lista_trazas,                          \
                        lista_vertices = lista_vertices,                      \
                        fit_trazas = None, distance_threshold = 0.45,         \
                        graficas = True)

print('\n Ajuste realizado con: AHC sin eliminar 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Clusters totales', num_clusters],
          ['Vertices faltan', vertices_faltan] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')


#%% BIRCH sin eliminar 0

notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
      clusters_mal, vertices_faltan, total_time, num_clusters =               \
          Algorithm.BIRCH(lista_trazas = lista_trazas,                        \
                        lista_vertices = lista_vertices,                      \
                        fit_trazas = None, threshold = 0.11, branching = 40,  \
                        graficas = True)


print('\n Ajuste realizado con: BIRCH sin eliminar 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Clusters totales', num_clusters],
          ['Vertices faltan', vertices_faltan]]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')

#%% BIRCH eliminando 0
notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
      clusters_mal, vertices_faltan, total_time, num_clusters =               \
          Algorithm.BIRCH(lista_trazas = lista_trazas_medidas,                \
                        lista_vertices = lista_vertices,                      \
                        fit_trazas = lista_trazas_no_medidas,                 \
                        threshold = 0.2, branching = 70, graficas = True)


print('\n Ajuste realizado con: BIRCH eliminando 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Clusters totales', num_clusters],
          ['Vertices faltan', vertices_faltan]]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')
