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
    pt : float, OPTIPNAL
        Minimum momentum of the tracks. Default 0
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
                            if len(data)< 6 or data[6]>=pt:
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
    lista_vertices = simvertices[:, :3]

    lista_trazas0 = tracks[:, -2]  # Columna correspondingSimVertex
    lista_trazas1 = tracks[:, 1]  # Valor z
    lista_trazas2 = tracks[:, 3]  # Valor t
    lista_trazas = np.column_stack((lista_trazas0, lista_trazas1,             \
                                    lista_trazas2))

    errores_z = tracks[:,2]
    errores_t = tracks[:,4]
    errores = np.column_stack((errores_z, errores_t))
    etiquetas_CMS = tracks[:,-1]

    centroide_CMS = np.column_stack((recovertices[:,1], recovertices[:,3]))

    return lista_vertices, lista_trazas, errores, etiquetas_CMS, centroide_CMS

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
    lista_vertices, lista_trazas, errores, etiquetas_CMS, centroides_CMS =    \
        transform_data_into_own_variables(simvertices, recovertices, tracks)

    return lista_vertices, lista_trazas, errores, etiquetas_CMS,              \
        centroides_CMS, num_clustersCMS

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
        for i in range(20):
            output_file = open(f"DataCMS_momentum{str(i)}.txt", "w")
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
