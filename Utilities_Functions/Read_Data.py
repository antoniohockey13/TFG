# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 19:40:00 2023

@author: Antonio
"""

import numpy as np


def digest_input(file: str):
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
                if data[0] != data_names[2]:
                    data = list(map(float, data))
                    tracks.append(data)

    except IOError:
        print(f"The file {file} does not exist")

    return num_evento, np.array(simvertices), np.array(recovertices),         \
        np.array(tracks)


def transform_data_into_own_variables(simvertices, recovertices, tracks):
    lista_vertices = simvertices[:, :3]

    lista_trazas0 = tracks[:, 5]  # Columna correspondingSimVertex
    lista_trazas1 = tracks[:, 1]  # Valor z
    lista_trazas2 = tracks[:, 3]  # Valor t
    lista_trazas = np.column_stack((lista_trazas0, lista_trazas1,             \
                                    lista_trazas2))

    num_trazas_en_v = simvertices[:,3]

    errores_z = tracks[:,2]
    errores_t = tracks[:,4]

    errores = np.column_stack((errores_z, errores_t))

    return lista_vertices, lista_trazas, num_trazas_en_v, errores


# num_evento, simvertices, recovertices, tracks =                               \
    # digest_input('SimulationDataCMS_Event0.txt')

# lista_vertices, lista_trazas, num_trazas_en_v, errores =                      \
    # transform_data_into_own_variables(simvertices, recovertices, tracks)
