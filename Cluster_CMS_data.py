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

lista_vertices, lista_trazas, num_trazas_en_v, errores =                      \
    read_data('Data/SimulationDataCMS_Event0.txt')
trazas_totales = len(lista_trazas)
num_vertices = len(lista_vertices)
X = np.column_stack((lista_trazas[:,1], lista_trazas[:,2]))
# K-Means
numcluster_manual = 215
num_clusters, centroides, etiquetas, total_time = Algoritmos.KMeans(X,        \
                                                lista_trazas, numcluster_manual)
notaajustada, notanorm, distancia, trazas_bien, trazas_mal,         \
       clusters_bien, clusters_mal, vertices_faltan, \
           contador_trazas_bien, num_trazas_en_v, contador_trazas_mal = Evaluar.evaluacion_total(lista_trazas,          \
                                etiquetas, centroides, lista_vertices,        \
                                num_trazas_en_v)

print('Ajuste realizado con: KMeans')
print(vertices_faltan)


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
