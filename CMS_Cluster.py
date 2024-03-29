# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:07:43 2023

@author: Antonio
"""

from tabulate import tabulate
import numpy as np

from Utilities_Functions import Read_Data
from Utilities_Functions import Evaluar

trazas_totales = []
distancia = []
notaajustada = []
notanorm = []
trazas_bien = []
trazas_mal = []
clusters_bien =  []
clusters_mal = []
num_clusters = []
num_vertices = []
vertices_faltan = []

# for i in range(28):
#     if i < 8:
#         name = f'Data/SimulationDataCMS_Event{i+2}.txt'
#     else:
#         name = f'Data/DataCMS_momentum{i-8}.txt'
for i in range(20):
    name = f'Data/DataCMS_momentum{i}.txt'

    lista_vertices, lista_trazas, clustertovertex_CMS, errores,               \
        etiquetas_CMS, centroides_CMS, num_clustersCMS, momentum =            \
            Read_Data.read_data(name, pt = 1.5)


    num_trazas = len(lista_trazas)
    inum_vertices = len(lista_vertices)

    # Cluster data
    inum_clusters = len(lista_vertices)
    inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
        iclusters_bien, iclusters_mal, ivertices_faltan =                     \
            Evaluar.evaluacion_total(lista_trazas, etiquetas_CMS,             \
                                     centroides_CMS, lista_vertices)
    # inotaajustada, inotanorm, idistancia, itrazas_bien, itrazas_mal,          \
    #     iclusters_bien, iclusters_mal, ivertices_faltan =                     \
    #         Evaluar.evaluar_cms(lista_trazas, etiquetas_CMS, centroides_CMS,  \
    #                             lista_vertices, clustertovertex_CMS)
    num_clusters.append(inum_clusters)
    num_vertices.append(inum_vertices)
    vertices_faltan.append(ivertices_faltan)
    distancia.append(idistancia)
    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)

    trazas_bien.append(itrazas_bien)
    trazas_mal.append(itrazas_mal)
    clusters_bien.append(iclusters_bien)
    clusters_mal.append(iclusters_mal)

    trazas_totales.append(num_trazas)



print('Evaluación cluster CMS')

tabla = [ [' ', 'media', 'error',],
          ['Trazas OK/Tot',                                                   \
               np.mean(np.array(trazas_bien)/np.array(trazas_totales)),       \
               np.std(np.array(trazas_bien)/np.array(trazas_totales))],

          ['Trazas MAL/Tot',                                                  \
               np.mean(np.array(trazas_mal)/np.array(trazas_totales)),        \
               np.std(np.array(trazas_mal)/np.array(trazas_totales))],

          ['Trazas tot', np.mean(trazas_totales), np.std(trazas_totales)],

          ['Vertices OK', np.mean(clusters_bien), np.std(clusters_bien)],

          ['Vertices MAL', np.mean(clusters_mal), np.std(clusters_mal)],

          ['Vertices faltan', np.mean(vertices_faltan),                       \
               np.std(vertices_faltan)],

          ['Vertices totales', np.mean(num_vertices), np.std(num_vertices)],
          ['Clusters totales', np.mean(num_clusters), np.std(num_clusters)]]

print(tabulate(tabla, headers =  []))

print(f'Nota ajustada:{np.mean(notaajustada)} +- {np.std(notaajustada)}')
print(f'Nota no ajustada:{np.mean(notanorm)} +- {np.std(notanorm)}')
print('Distancia de los centroides a los vértices (normalizada entre número '\
        f'vértices): {np.mean(distancia)} +- {np.std(distancia)}')

verticesOK_tot = np.array(clusters_bien)/np.array(num_vertices)
print(f'Vértices OK/Vertices Totales =  {np.mean(verticesOK_tot)}+- '\
      f'{np.std(verticesOK_tot)}')
numclusters_relativo = np.array(num_clusters)/np.array(num_vertices)
print(f'Clusters/Vertices Totales =  {np.mean(numclusters_relativo)}+- '\
      f'{np.std(numclusters_relativo)}')
