# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 19:40:00 2023

@author: Antonio
"""

import numpy as np


def digest_input(file: str, pt: float = 0):
    """
    Parameters
    ----------
    file : str
        Name of the file wanted to be read.
    pt : float, OPTIPNAL
        Minimum momentum of the tracks. Default 0

    Returns
    -------
    num_evento : int
        Event number read
    simvertices : np.array([NºV, z, t, num_tracks])
        Rough read data from Vertex Simulation
    recovertice : np.array([NºrecoV, z, error_z, t, error_t,
                            CorrespondingSimVertex])
        Rough read data from Vertex Reconstruction
    tracks : np.arrray([NºTraza, z, error_z, t, error_t,
                        CorrespondingSimVertex, CorrespondingRecoVertex])
        Rough read data from Tracks
    """
    data_names = ['SimVertices', 'RecoVertices', 'Tracks']
    seleccion_datos = -1
    #   We open the file from which we are going to read the input
    try:
        with open(file, "r") as f:
            lines = f.readlines()
        simvertices = []
        recovertices = []
        tracks = []
        for line in lines:
            if len(line)>1:
                if line.startswith("Event "):
                    # obtenemos el número de evento de la línea actual
                    num_evento = int(line.split()[1])
                    seleccion_datos = 3
                if line.startswith(data_names[0]):
                    seleccion_datos = 0
                if line.startswith(data_names[1]):
                    seleccion_datos = 1
                if line.startswith(data_names[2]):
                    seleccion_datos = 2
                if seleccion_datos == 0:
                    data = line.strip().split()
                    if data[0] != data_names[0]:
                        data = list(map(float, data))
                        simvertices.append(data)
                if seleccion_datos == 1:
                    data = line.strip().split()
                    if data[0] != data_names[1]:
                        data = list(map(float, data))
                        recovertices.append(data)
                if seleccion_datos == 2:
                    data = line.strip().split()
                    if len(data) != 0:
                        if data[0] != data_names[2]:
                            data = list(map(float, data))
                            if len(data)<= 7 or data[6]>=pt:
                                tracks.append(data)

    except IOError:
        print(f"The file {file} does not exist")

    return num_evento, np.array(simvertices), np.array(recovertices),         \
        np.array(tracks)


def transform_data_into_own_variables(simvertices: np.array(4),               \
                                      recovertices: np.array(4),              \
                                      tracks: np.array(7)):
    """
    Transform CMS data into program variables

    Parameters
    ----------
    simvertices : np.array(4)
       Rough read data from Vertex Simulation
    recovertice : np.array(4)
            Rough read data from Vertex Reconstruction
    tracks : np.arrray(7)
        Rough read data from Tracks.

    Returns
    -------
    lista_vertices : np.array(3)
        Lista con las posiciones de los vértices.
    lista_trazas : np.array(3)
        Lista con las posiciones de las trazas y el vértice al que pertenecen.
    errores : np.array(2)
        Errores en la medida de las trazas.
    """
    num_vertice = simvertices[:,0]
    vertice_z = simvertices[:,1]
    vertice_t = simvertices[:,2]


    lista_trazas0 = tracks[:, -2]  # Columna correspondingSimVertex
    lista_trazasz = tracks[:, 1]  # Valor z
    media_z = np.mean(lista_trazasz)
    desviacion_z = np.std(lista_trazasz)
    # media_z = 0
    # desviacion_z = 1
    trazas_z = (lista_trazasz-media_z)/desviacion_z


    lista_trazast = tracks[:, 3] # Valor t
    media_t = np.mean(lista_trazast)
    desviacion_t = np.std(lista_trazast)
    # media_t = 0
    # desviacion_t = 1
    trazas_t = (lista_trazast-media_t)/desviacion_t


    lista_trazas = np.column_stack((lista_trazas0, trazas_z ,trazas_t))

    vertice_z = (vertice_z-media_z)/desviacion_z
    vertice_t = (vertice_t-media_t)/desviacion_t
    lista_vertices = np.column_stack((num_vertice, vertice_z, vertice_t))

    errores_z = (tracks[:,2]-media_z)/desviacion_z
    errores_t = (tracks[:,4]-media_t)/desviacion_t
    errores = np.column_stack((errores_z, errores_t))
    etiquetas_CMS = tracks[:,-1]

    clustertovertex_CMS = recovertices[:,-1]
    clustertovertex_original = recovertices[:,-2]

    cms_z = (recovertices[:,1]-media_z)/desviacion_z
    cms_t = (recovertices[:,3]-media_t)/desviacion_t
    centroide_CMS = np.column_stack((cms_z, cms_t))
    momentum =  None
    if tracks.shape[1] == 9:
        momentum = tracks[:,6]

    return lista_vertices, lista_trazas, clustertovertex_CMS, errores,        \
        etiquetas_CMS, centroide_CMS, momentum

def read_data(name: str, pt: float = 0):
    """
    Read data from .txt file

    Parameters
    ----------
    name : str
        Name of the file.
    pt : float, OPTIPNAL
        Minimum momentum of the tracks. Default 0

    Returns
    -------
    lista_vertices : np.array(3)
        Lista con las posiciones de los vértices.
    lista_trazas : np.array(3)
        Lista con las posiciones de las trazas y el vértice al que pertenecen.
    errores : np.array(2)
        Errores en la medida de las trazas
    """
    num_evento, simvertices, recovertices, tracks = digest_input(name, pt)
    num_clustersCMS = len(recovertices)
    lista_vertices, lista_trazas, clustertovertex_CMS, errores, etiquetas_CMS,\
        centroides_CMS, momentum =                                            \
        transform_data_into_own_variables(simvertices, recovertices, tracks)

    return lista_vertices, lista_trazas, clustertovertex_CMS, errores,        \
        etiquetas_CMS, centroides_CMS, num_clustersCMS, momentum

def quit_not_measure_vertex(lista_trazas, errores):
    """
    Se eliminan las trazas que no han sido detectadas temporalmente

    Parameters
    ----------
    lista_trazas : np.array(3)
        Lista con las posiciones de las trazas y el vértice al que pertenecen.
    errores : np.array(2)
        Errores en la medida de las trazas.

    Returns
    -------
    lista_trazas_medidas : np.array(3)
        Lista de trazas SÍ detectadas temporalmente.
    errores_medidos : np.array(2)
        Errores en las trazas SÍ detectadas.
    lista_trazas_no_medidas : np.array(2)
        Lista de trazas NO detectadas temporalmente.

    """
    lista_trazas_medidas = []
    lista_trazas_no_medidas = []
    errores_medidos = []
    errores_no_medidos = []
    for i, itraza in enumerate(lista_trazas):
        if itraza[2] == 0 and errores[i,1] == 0:
            lista_trazas_no_medidas.append(itraza)
            errores_no_medidos.append([errores[i,0], errores[i,1]])
        else:
            lista_trazas_medidas.append(itraza)
            errores_medidos.append([errores[i,0], errores[i,1]])
    return np.array(lista_trazas_medidas), np.array(errores_medidos),         \
        np.array(lista_trazas_no_medidas), np.array(errores_no_medidos)


def separar_eventos(file: str):
    try:
        with open(file, 'r') as f:
            lines = f.readlines()
        start = 0
        for i in range(1):
            output_file = open(f"DataCMS_2{str(i)}.txt", "w")
            for iline in range(start, len(lines)):
                line = lines[iline]
                evento = f"Event {i+1}\n"
                start = iline
                if line != evento:
                    output_file.write(line+'\n')
                else:
                    output_file.close()
                    break
        output_file.close()
    except IOError:
        print(f"The file {file} does not exist")
