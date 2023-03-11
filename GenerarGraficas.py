# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:50:12 2023

@author: Antonio Gómez
Generar gráficas
"""

import numpy as np
import matplotlib.pyplot as plt
import GenerarConjuntoVerticesyTrazas as gcvt
import Algoritmos
import Evaluar


#%% Generar vertices y trazas

lista_vertices, lista_trazas, pos_trazas, num_trazas_en_v, X, num_trazas      \
    = gcvt.VerticesyTrazasAleatorios( num_vertices = 200,            \
            mediatrazas = 70, sigmatrazas = 10, mediaz = 0, sigmaz = 5,       \
            mediat = 0, sigmat = 200, mediar = 0, sigmar = 0.05,              \
            error_z = 0.02, error_t = 10)

num_trazas = len(lista_trazas)

#%% Gráfica de notas vs num cluster en K-Means


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
plt.title('K-Means')
plt.savefig('Notas_vs_numclusters-KMeans1.pdf')
plt.show()

#%% Barrido de epsilon en DBSCAN

epsilons = np.linspace(0.7, 2.25, 30)


notaajustada = []
notanorm = []
notamedia = []

for iepsilon in epsilons:
    num_clusters, centroides, etiquetas, total_time, num_noise =              \
        Algoritmos.DBSCAN(X, lista_trazas, iepsilon)

    inotaajustada, inotanorm = Evaluar.evaluacion(lista_trazas, etiquetas)
    notaajustada.append(inotaajustada)
    notanorm.append(inotanorm)
    notamedia.append((inotaajustada**2+inotanorm**2)/2)

plt.plot(epsilons, notaajustada, 'x', c = 'b', label = 'Nota ajustada')
plt.plot(epsilons, notanorm, 'x', c = 'r', label = 'Nota normal')
plt.plot(epsilons, notamedia, 'o', c = 'g', label = 'Nota media')
plt.xlabel('epsilon')
plt.ylabel('Puntos')
plt.legend(loc='best')
plt.title('DBSCAN')
plt.savefig('Notas_vs_epsilon-DBSCAN2.pdf')
plt.show()
