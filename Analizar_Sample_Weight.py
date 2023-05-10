# -*- coding: utf-8 -*-
"""
Created on Sat May  6 21:34:33 2023

@author: Antonio
"""
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from Utilities_Functions import Algoritmos

# Sample Weight K-Means
num_centers = 4
num_trazas = 10

X, y = datasets.make_blobs(n_samples = num_trazas,                   \
                                centers = 1)
lista_trazas = []
for i in range(len(X)):
    lista = [0, X[0,0], X[0,1]]
    lista_trazas.append(lista)

for icenter in range(num_centers-1):

    X1, y = datasets.make_blobs(n_samples = num_trazas,                   \
                                    centers = 1)
    for i_X in range(len(X1)):
        lista = [icenter+1,X[0,0], X[0,1]]
        lista_trazas.append(lista)
    X = np.concatenate((X,X1))

lista_trazas = np.array(lista_trazas)
errores = [1,0,0,0,0,0,0,0,0,0, \
           1,0,0,0,0,0,0,0,0,0, \
           1,0,0,0,0,0,0,0,0,0, \
           1,0,0,0,0,0,0,0,0,0]
#%%K-Means
inum_clusters, centroides, etiquetas, total_time = Algoritmos.KMeans(X,   \
                                        None, errores ,           \
                                        numcluster_manual = 4, \
                                        n_init = 10, tol = 1e-4)
inum_clusters, centroidessin, etiquetas, total_time = Algoritmos.KMeans(X,   \
                                                    None, None ,           \
                                                    numcluster_manual = 4, \

                                                        n_init = 10, tol = 1e-4)
plt.plot(centroides[:,0], centroides[:,1], 'x',                               \
          label = 'Centroides con errores')
plt.plot(X[:,0], X[:,1], '.', label = 'Trazas')
for i in range(num_centers):
    traza = i*num_trazas
    plt.plot(X[traza,0], X[traza,1], 'o', label = 'Traza error alto' )
plt.legend(loc = 'best')
plt.title('K-Means')
plt.show()

plt.plot(centroidessin[:,0], centroidessin[:,1], 'o', c = 'orange',           \
          label = 'Centroides sin errores')
plt.plot(centroides[:,0], centroides[:,1], 'xb',                              \
          label = 'Centroides con errores')
plt.legend(loc='best')
plt.title('K-Means')
plt.show()

#%% DBSCAN

num_clusters, centroides, etiquetas, total_time, num_noise =                  \
                            Algoritmos.DBSCAN(X = X,                          \
                            lista_trazas = lista_trazas,    \
                            sample_weight = errores, epsilon =1,           \
                            min_samples = 1, leaf_size = 1)
num_clusters, centroidessin, etiquetas, total_time, num_noise =               \
                            Algoritmos.DBSCAN(X = X,                          \
                            lista_trazas = lista_trazas,     \
                            sample_weight = None, epsilon = 1,           \
                            min_samples = 1, leaf_size = 1)

plt.plot(centroides[:,0], centroides[:,1], 'x',                               \
          label = 'Centroides con errores')
plt.plot(X[:,0], X[:,1], '.', label = 'Trazas')
for i in range(num_centers):
    traza = i*num_trazas
    plt.plot(X[traza,0], X[traza,1], 'o', label = 'Traza error alto' )
plt.legend(loc = 'best')
plt.title('DBSCAN')
plt.show()


plt.plot(centroidessin[:,0], centroidessin[:,1], 'o', c = 'orange',           \
          label = 'Centroides sin errores')
plt.plot(centroides[:,0], centroides[:,1], 'xb',                              \
          label = 'Centroides con errores')
plt.legend(loc='best')
plt.title('DBSCAN')
plt.show()

#%% EM-GMM

num_clusters, centroides, etiquetas, total_time, num_noise =                  \
                            Algoritmos.EM_GMM(X = X,                          \
                                              lista_trazas = lista_trazas, \
                                              sample_weight = errores, \
                                              numcluster_manual = 4)

num_clusters, centroidessin, etiquetas, total_time, num_noise =                  \
                            Algoritmos.EM_GMM(X = X,                          \
                                              lista_trazas = lista_trazas, \
                                              sample_weight = None, \
                                              numcluster_manual = 4)
plt.plot(centroides[:,0], centroides[:,1], 'x',                               \
          label = 'Centroides con errores')
plt.plot(X[:,0], X[:,1], '.', label = 'Trazas')
for i in range(num_centers):
    traza = i*num_trazas
    plt.plot(X[traza,0], X[traza,1], 'o', label = 'Traza error alto' )
plt.legend(loc = 'best')
plt.title('EM-GMM')
plt.show()


plt.plot(centroidessin[:,0], centroidessin[:,1], 'o', c = 'orange',           \
          label = 'Centroides sin errores')
plt.plot(centroides[:,0], centroides[:,1], 'xb',                              \
          label = 'Centroides con errores')
plt.legend(loc='best')
plt.title('EM-GMM')
plt.show()
