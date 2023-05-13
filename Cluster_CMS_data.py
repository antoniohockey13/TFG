# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:48:15 2023

@author: Antonio
"""
from tabulate import tabulate
import numpy as np

from Utilities_Functions import Algoritmos
from Utilities_Functions import Evaluar
from Utilities_Functions import Grafica_Clusters
from Utilities_Functions import Read_Data

def read_data(name: str):
    """
    Read data from .txt file

    Parameters
    ----------
    name : str
        Name of the file.

    Returns
    -------
    lista_vertices : np.array(3)
        Lista con las posiciones de los vértices.
    lista_trazas : np.array(3)
        Lista con las posiciones de las trazas y el vértice al que pertenecen.
    num_trazas_en_v : np.array(1)
        Número de trazas en cada vértice.
    errores : np.array(2)
        Errores en la medida de las trazas
    """
    num_evento, simvertices, recovertices, tracks =                           \
        Read_Data.digest_input(name)

    lista_vertices, lista_trazas, num_trazas_en_v, errores =                  \
        Read_Data.transform_data_into_own_variables(simvertices, recovertices,\
                                                    tracks)
    return lista_vertices, lista_trazas, num_trazas_en_v, errores

num_evento = str(0)
lista_vertices, lista_trazas, num_trazas_en_v, errores =                      \
    read_data(f'Data/SimulationDataCMS_Event{num_evento}.txt')
trazas_totales = len(lista_trazas)
num_vertices = len(lista_vertices)
X = np.column_stack((lista_trazas[:,1], lista_trazas[:,2]))

#%% K-Means
numcluster_manual = 215

num_clusters, centroides, etiquetas, total_time =                             \
    Algoritmos.KMeans(X = X, lista_trazas = lista_trazas, fit_trazas = None,  \
            sample_weight= None, error_predict = None,                        \
            numcluster_manual = numcluster_manual, n_init = 10, tol = 1e-6)

notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, clusters_faltan = Evaluar.evaluacion_total(lista_trazas,   \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)
Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'K-Means')


print('\n Ajuste realizado con: KMeans sin eliminar 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Vertices faltan', clusters_faltan],
          ['Clusters totales', num_clusters] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')
#%% K-Means eliminando trazas no detectadas temporalmente
numcluster_manual = 215

lista_trazas_medidas, errores_medidos, lista_trazas_no_medidas,               \
    errores_no_medidos = Read_Data.quit_not_measure_vertex(lista_trazas, errores)

num_clusters, centroides, etiquetas, total_time =                             \
    Algoritmos.KMeans(X = X, lista_trazas = lista_trazas_medidas,             \
                fit_trazas = lista_trazas_no_medidas, sample_weight= None,    \
                error_predict = None, numcluster_manual = numcluster_manual,  \
                n_init = 10, tol = 1e-6)

notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, clusters_faltan = Evaluar.evaluacion_total(lista_trazas,   \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)

Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'K-Means')

print('\n Ajuste realizado con: KMeans eliminando 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Vertices faltan', clusters_faltan],
          ['Clusters totales', num_clusters] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')
#%% MeanShift sin eliminar 0
num_clusters, centroides, etiquetas, total_time = Algoritmos.MeanShift(X = X, \
                                         fit_trazas = None, quantile = 1e-2,  \
                                         n_samples = 299, min_bin_freq = 31)

notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, clusters_faltan = Evaluar.evaluacion_total(lista_trazas,   \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)

Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'MeanShift')


print('\n Ajuste realizado con: MeanShift sin eliminar 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Vertices faltan', clusters_faltan],
          ['Clusters totales', num_clusters] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')

#%% MeanShift eliminando 0
lista_trazas_medidas, errores_medidos, lista_trazas_no_medidas,               \
    errores_no_medidos = Read_Data.quit_not_measure_vertex(lista_trazas, errores)

num_clusters, centroides, etiquetas, total_time = Algoritmos.MeanShift(X = X, \
                                         fit_trazas = None, quantile = 1e-2,  \
                                         n_samples = 299, min_bin_freq = 31)

notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, clusters_faltan = Evaluar.evaluacion_total(lista_trazas,   \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)

Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'MeanShift')



notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, clusters_faltan = Evaluar.evaluacion_total(lista_trazas,   \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)

Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'K-Means')

print('\n Ajuste realizado con: MeanShift eliminando 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Vertices faltan', clusters_faltan],
          ['Clusters totales', num_clusters] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')
#%% DBSCAN sin eliminar 0
num_clusters, centroides, etiquetas, total_time, num_noise =                  \
    Algoritmos.DBSCAN(X = X, fit_trazas = None, lista_trazas = lista_trazas,  \
                      sample_weight = None, error_predict = None,             \
                      epsilon = 0.2, min_samples = 20 , leaf_size = 12)

notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, clusters_faltan = Evaluar.evaluacion_total(lista_trazas,   \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)

Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'DBSCAN')


print('\n Ajuste realizado con: DBSCAN sin eliminar 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Vertices faltan', clusters_faltan],
          ['Clusters totales', num_clusters] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')

#%% DBSCAN Eliminando 0
lista_trazas_medidas, errores_medidos, lista_trazas_no_medidas,               \
    errores_no_medidos = Read_Data.quit_not_measure_vertex(lista_trazas, errores)

num_clusters, centroides, etiquetas, total_time, num_noise =                  \
    Algoritmos.DBSCAN(X = X, fit_trazas = None, lista_trazas = lista_trazas,  \
                      sample_weight = None, error_predict = None,             \
                      epsilon = 0.2, min_samples = 20, leaf_size = 12)

notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, clusters_faltan = Evaluar.evaluacion_total(lista_trazas,   \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)

Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'DBSCAN')

print('\n Ajuste realizado con: DBSCAN eliminando 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Vertices faltan', clusters_faltan],
          ['Clusters totales', num_clusters] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')

#%% EM-GMM sin eliminar 0
num_clusters, centroides, etiquetas, total_time =                             \
    Algoritmos.EM_GMM(X = X, lista_trazas = lista_trazas, fit_trazas = None,  \
           sample_weight = None, numcluster_manual = 215)

notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, clusters_faltan = Evaluar.evaluacion_total(lista_trazas,   \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)

Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'EM-GMM')


print('\n Ajuste realizado con: EM-GMM sin eliminar 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Vertices faltan', clusters_faltan],
          ['Clusters totales', num_clusters] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')

#%% EM-GMM Eliminando 0
lista_trazas_medidas, errores_medidos, lista_trazas_no_medidas,               \
    errores_no_medidos = Read_Data.quit_not_measure_vertex(lista_trazas, errores)

num_clusters, centroides, etiquetas, total_time =                             \
    Algoritmos.EM_GMM(X = X, lista_trazas = lista_trazas,                     \
        fit_trazas = lista_trazas_no_medidas, sample_weight = None,           \
        numcluster_manual = 215)


notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, clusters_faltan = Evaluar.evaluacion_total(lista_trazas,   \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)

Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'EM-GMM')

print('\n Ajuste realizado con: EM-GMM eliminando 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Vertices faltan', clusters_faltan],
          ['Clusters totales', num_clusters] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')

#%% AHC sin eliminar 0
num_clusters, centroides, etiquetas, total_time =                             \
    Algoritmos.AHC(X = X, lista_trazas = lista_trazas,                        \
                   fit_trazas = None, distance_threshold = 1)

notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, clusters_faltan = Evaluar.evaluacion_total(lista_trazas,   \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)

Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'AHC')


print('\n Ajuste realizado con: AHC sin eliminar 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Vertices faltan', clusters_faltan],
          ['Clusters totales', num_clusters] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')

#%% AHC Eliminando 0 no se puede
# lista_trazas_medidas, errores_medidos, lista_trazas_no_medidas,               \
#     errores_no_medidos = Read_Data.quit_not_measure_vertex(lista_trazas, errores)

# num_clusters, centroides, etiquetas, total_time =                             \
#     Algoritmos.AHC(X = X, lista_trazas = lista_trazas,                        \
#                   fit_trazas = lista_trazas_no_medidas, distance_threshold = 1)

# notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
#     clusters_mal, clusters_faltan = Evaluar.evaluacion_total(lista_trazas,   \
#                                 etiquetas, centroides, lista_vertices,        \
#                                 num_trazas_en_v)

# Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'AHC')

# print('\n Ajuste realizado con: AHC eliminando 0')

# tabla = [ [' ', 'Valor',],
#           ['Trazas OK/Tot', trazas_bien/trazas_totales],
#           ['Trazas MAL/Tot', trazas_mal/trazas_totales],
#           ['Trazas tot', trazas_totales],
#           ['Vertices OK', clusters_bien],
#           ['Vertices MAL', clusters_mal],
#           ['Vertices faltan', clusters_faltan],
#           ['Clusters totales', num_clusters] ]
# print(tabulate(tabla, headers = []))
# print(f'Vertices totales = {num_vertices}')

# print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
# print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

# print('Distancia de los centroides a los vértices (normalizada entre número '\
#         f'vértices): {distancia}')

# print(f'Tiempo en ejecutar = {total_time} s')

#%% BIRCH sin eliminar 0
num_clusters, centroides, etiquetas, total_time =                             \
    Algoritmos.BIRCH(X = X, fit_trazas = None, threshold = 0.2, branching = 70)

notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, clusters_faltan = Evaluar.evaluacion_total(lista_trazas,   \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)

Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'BIRCH')


print('\n Ajuste realizado con: BIRCH sin eliminar 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Vertices faltan', clusters_faltan],
          ['Clusters totales', num_clusters] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')

#%% BIRCH Eliminando 0
lista_trazas_medidas, errores_medidos, lista_trazas_no_medidas,               \
    errores_no_medidos = Read_Data.quit_not_measure_vertex(lista_trazas, errores)

num_clusters, centroides, etiquetas, total_time =                             \
    Algoritmos.BIRCH(X = X, fit_trazas = None, threshold = 0.2, branching = 70)

notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, clusters_faltan = Evaluar.evaluacion_total(lista_trazas,   \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)

Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'BIRCH')

print('\n Ajuste realizado con: BIRCH eliminando 0')

tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Vertices faltan', clusters_faltan],
          ['Clusters totales', num_clusters] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')
