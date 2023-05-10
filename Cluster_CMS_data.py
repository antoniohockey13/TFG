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
    num_evento, simvertices, recovertices, tracks =                           \
        Read_Data.digest_input(name)

    lista_vertices, lista_trazas, num_trazas_en_v, errores =                  \
        Read_Data.transform_data_into_own_variables(simvertices, recovertices, \
                                                    tracks)
    return lista_vertices, lista_trazas, num_trazas_en_v, errores

num_evento = str(0)
lista_vertices, lista_trazas, num_trazas_en_v, errores =                      \
    read_data(f'Data/SimulationDataCMS_Event{num_evento}.txt')
trazas_totales = len(lista_trazas)
num_vertices = len(lista_vertices)
X = np.column_stack((lista_trazas[:,1], lista_trazas[:,2]))
# Hacer errores bien
# errores = None

#%% K-Means
numcluster_manual = 215
num_clusters, centroides, etiquetas, trazas_predict, total_time =             \
    Algoritmos.KMeans(X = X, lista_trazas = lista_trazas, fit_trazas = None,  \
            sample_weight= None, error_predict = None,                        \
            numcluster_manual = numcluster_manual)

notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, vertices_faltan = Evaluar.evaluacion_total(lista_trazas,   \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)
Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'K-Means')
print('\n Ajuste realizado con: KMeans sin eliminar 0')
print(f'Vertices faltan {vertices_faltan}')


tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
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

num_clusters, centroides, etiquetas, trazas_predict, total_time =             \
    Algoritmos.KMeans(X = X, lista_trazas = lista_trazas_medidas,             \
                fit_trazas = lista_trazas_no_medidas, sample_weight= None,    \
                error_predict = None, numcluster_manual = numcluster_manual)
notaajustada, notanorm, distancia, trazas_bien, trazas_mal, clusters_bien,    \
    clusters_mal, vertices_faltan = Evaluar.evaluacion_total(lista_trazas,   \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)
Grafica_Clusters.grafica_colores_cluster(lista_trazas, etiquetas, 'K-Means')
print('\n Ajuste realizado con: KMeans eliminando 0')
print(f'Vertices faltan {vertices_faltan}')


tabla = [ [' ', 'Valor',],
          ['Trazas OK/Tot', trazas_bien/trazas_totales],
          ['Trazas MAL/Tot', trazas_mal/trazas_totales],
          ['Trazas tot', trazas_totales],
          ['Vertices OK', clusters_bien],
          ['Vertices MAL', clusters_mal],
          ['Clusters totales', num_clusters] ]
print(tabulate(tabla, headers = []))
print(f'Vertices totales = {num_vertices}')

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')

print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {distancia}')

print(f'Tiempo en ejecutar = {total_time} s')
