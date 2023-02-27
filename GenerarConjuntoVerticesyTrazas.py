# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:59:30 2023

@author: Antonio
Código para generar de manera aleatoria un conjunto de Vértices y a su
alrededor un número de trazas.
"""
import numpy as np
import matplotlib.pyplot as plt



def VerticesyTrazasAleatorios(num_vertices = 200, mediatrazas = 70,           \
                              sigmatrazas = 10, mediaz = 0, sigmaz = 5,       \
                              mediat = 0, sigmat = 200, mediar = 0,           \
                              sigmar = 0.05, error_z = 0.2, error_t =0.1):
    """
    Parameters
    ----------
    num_vertices : int, optional
        Número de vértices generados. The default is 200.
    mediatrazas : int, optional
        Número medio de trazas en cada vértice. The default is 70.
     sigmatrazas : int, optional
        Desviación estándar del número medio de trazas en cada vértice.
        The default is 10.
    mediaz : float, optional
        Valor medio de la posición z de los vértices en cm. The default is 0.
    sigmaz : floar, optional
        Desviación estándar de la posición z de los vértices. The default is 5.
    mediat : float, optional
        Valor medio del tiempo para los vértices en ps. The default is 0.
    sigmat : float, optional
        Desviación estándar de t para los vértices. The default is 70.
    mediar : float, optional
        Valor medio de la distancia a la que se encuentra la traza de su
        respectivo vértice. The default is 0.5.
    sigmar : float, optional
        Desviación estándar de r. The default is 0.1.
    error_z : float, optional
        Error en la medición en z de la traza en cm. The default is 0.2.
    error_t : float, optional
        Error en la medición de t en la traza en ps. The default is 0.1.

    Returns
    -------
    lista_vertices: np.array([V, z, t])
        Lista con las posiciones de los vértices
    lista_trazas: np.array([V, z, t])
        Lista con las posiciones de las trazas y el vértice al que pertenecen
    pos_trazas: np.array([z, t])
        Lista con las posiciones de las trazas
    num_trazas_en_v: list[int]
        Número de trazas en cada vértice
    Devuelve tres ficheros .dat con estos mismos valores
    Pinta una figura con los vértices y otra con vértices y trazas
    """
    # Lista vacía para guardar los vértices se guardan en formato [V,z,t]

    lista_vertices = []

    for ivert in range(num_vertices):
        zivert = np.random.normal(mediaz, sigmaz)
        tivert = np.random.normal(mediat, sigmat)
        verticei = [ivert, zivert, tivert]
        lista_vertices.append(verticei)

    lista_vertices = np.array(lista_vertices)

    # Graficar vértices
    # plt.plot(lista_vertices[:,1], lista_vertices[:,2], 'x', c = 'b')
    # plt.xlim(-10, 10)
    # plt.ylim(-100, 100)
    # plt.xlabel("$z$/cm")
    # plt.ylabel("$t$/ps")
    # plt.show()


    # Generar trazas
    # Se guarda en una lista cuantas trazas tiene cada vértice
    num_trazas_en_v = []
    for ivert in range(num_vertices):
        ivertnum_traza = round(np.random.normal(mediatrazas, sigmatrazas))
        num_trazas_en_v.append(ivertnum_traza)


    # Se generan las trazas alrededor de cada vértice siguiendo una distribución
    # normal para el radio y una distribución uniforme para el ángulo a su alrededor


    # Lista vacía para guardar trazas formato [V,z,t]
    lista_trazas = []
    pos_trazas = []
    for ivert in range(num_vertices):
        zivert = lista_vertices[ivert, 1]
        tivert = lista_vertices[ivert, 2]
        for itraza in range(num_trazas_en_v[ivert]):
            radio = np.random.normal(mediar, sigmar)
            angulo = np.random.random()*2*np.pi
            zitraza = zivert+radio*np.cos(angulo)
            titraza = tivert+radio*np.sin(angulo)
            trazai = [ivert, zitraza, titraza]
            postrazai = [zitraza, titraza]
            lista_trazas.append(trazai)
            pos_trazas.append(postrazai)
    lista_trazas = np.array(lista_trazas)
    pos_trazas = np.array(pos_trazas)

    # Gráfica vértices y trazas
    # plt.plot(lista_vertices[:,1], lista_vertices[:,2], 'x')
    # plt.errorbar(lista_trazas[:,1], lista_trazas[:,2], xerr = error_z, \
    #              yerr = error_t, fmt= 'or', linestyle="None")

    # plt.xlim(-10, 10)
    # plt.ylim(-100, 100)
    # plt.xlabel("$z$/cm")
    # plt.ylabel("$t$/ps")
    # plt.show()

    # Guardar datos en ficheros .dat
    output_file_vertices = open("vertices.dat", "w")
    output_file_vertices.write("Vértice    z/cm    t/ps \n")
    for i in range(num_vertices):
        for j in range(3):
            output_file_vertices.write(f"{lista_vertices[i,j]}    ")
        output_file_vertices.write("\n")
    output_file_vertices.close()

    output_file_trazas = open("trazas.dat", "w")
    output_file_trazas.write("Vértice    z/cm    t/ps \n")
    for i in range(len(lista_trazas)):
        for j in range(3):
            output_file_trazas.write(f"{lista_trazas[i,j]}    ")
        output_file_trazas.write("\n")
    output_file_trazas.close()

    output_file_trazassinvert = open("trazassinvert.dat", "w")
    output_file_trazassinvert.write("Vértice    z/cm    t/ps \n")
    for i in range(len(lista_trazas)):
        output_file_trazassinvert.write(f"{lista_trazas[i,1]}    "\
                                        f"{lista_trazas[i,2]}\n")
    output_file_trazassinvert.close()
    return(lista_vertices, lista_trazas, pos_trazas, num_trazas_en_v)
