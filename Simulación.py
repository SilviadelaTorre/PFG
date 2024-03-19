import csv
import datetime
import operator
import os
import re
from random import randint
import networkx as nx
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from operator import itemgetter
from shapely.wkt import loads
import copy
from os import path
import netwulf as nw
import cartopy.crs as ccrs
import cartopy.feature as cfeature


NOT_INFECTED=0
INFECTED=1
NUM_NODOS=1
PERCENTAGE = 0.60


# def GraficarRed_API(G):
#     print("Sacar imagen del gráfico usando netwulf")
#     #pos = nx.kamada_kawai_layout(G)
#     # print("kamada layout hecho")

#     nw.visualize(G) #,config=pos)

# def GraficarRed(G,n_color):

#         print("Graficando...")
#         pos = nx.kamada_kawai_layout(G)

#         # Dibuja los enlaces
#         nx.draw(G, pos, node_color=[n_color[node] for node in G.nodes()], node_size=30, edge_color="grey", width=1, alpha=0.5)
#         plt.xlim(-1.5, 1.5)  # Ajusta los límites x
#         plt.ylim(-1.5, 1.5)  # Ajusta los límites y
#         plt.savefig("Results/GRAFICOS REDES/3 COORDS/RIO-GARONA.png")
#         plt.show()
#         # Dibujar el grafo (solo para fines de visualización)
#         # GraficarRed(G)
        

def CrearGrafo(edge_list):
    # Crear un grafo no dirigido
    G = nx.DiGraph()

    # Analizar el edge list y agregar los nodos y las aristas al grafo
    with open(edge_list, 'r') as f:
        for line in f:
            nodos_y_atributos = line.strip().split(';')
            nodo1_coords = tuple(map(float, nodos_y_atributos[0][1:-1].split(', ')))
            nodo2_coords = tuple(map(float, nodos_y_atributos[1][1:-1].split(', ')))

            # Agregar la arista al grafo
            G.add_edge(nodo2_coords, nodo1_coords)

    print("Número de nodos:", G.number_of_nodes())
    print("Número de aristas:", G.number_of_edges())
    degrees = G.degree()
    nodes = G.nodes()
    n_color = np.asarray([degrees[n] for n in nodes])
    #GraficarRed(G,n_color)
    return G

def StatusNodes(V):
    global NodeInfom; df

    Nodes=V.nodes()
    NodeInfo = pd.DataFrame(Nodes)
    #print(NodeInfo)
    data = {'Node': Nodes, 'Infection Status': [NOT_INFECTED] * len(Nodes)}
    df = pd.DataFrame(data)

    print(f'Status data frame: {df}')

def Infected_Nodes(V):
    global df

    StatusNodes(V)
    TotalNodes = len(V.nodes())
    TotalNodesInfect = df['Infection Status'].sum()

    Rate = TotalNodesInfect/TotalNodes
    return Rate


def ObtenerPrimerNodo(G,agente):
    max = 0
    coord_max_fosfato = None
    print(str(G))
    with open(agente, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            # Obtiene el ID del río, la coordenada y el valor de fosfato
            id_rio = row[0]
            coord = row[1]
            valor_contaminante = float(row[2])
            # Comprueba si el valor actual de fosfato es mayor que el máximo
            if str(id_rio).startswith("10034") and valor_contaminante > max:
                max = valor_contaminante
                coord_max = coord
                print(coord_max)

    print(f'El valor máximo de {str(agente)} es {max} en rio {id_rio}-{coord_max}')

    return coord_max_fosfato

# def GraficarInfección(G,FirstNode):
#     print("Coloreando infección")

#     print(FirstNode)
#     node_colors = ['red' if node == FirstNode else 'blue' for node in G.nodes()]
#     print(G.nodes())
#     for node in G.nodes():
#         print(node)
#         if node == FirstNode:
#             print("ENCONTRADO")
#     print(node_colors)
#     GraficarRed(G,node_colors)

def EjecutarIteracionInfeccion(V,Tipo):

    global pathFile, Time, NodeInfo, Rate, TotalNodes

    #print("Iteracion" + str(Iter))
    # print(Iter)

    print("Propagando infeccion")

    Nodes = V.nodes()
    TotalNodes = len(Nodes)
    print("Total nodes " + str(TotalNodes))
    FirstNode = ObtenerPrimerNodo(V,Tipo)
    # FirstNode=Nodes[randint(1, len(Nodes))-1]
    print("Primer nodo infectado")
    print(FirstNode)
    #GraficarInfección(V,FirstNode)

def PropagarInfeccion2 (pR, pC, Tipo,V):

    global NDir

    print("Creando directorio")
    NDir= "/Users/silviadelatorre/Desktop/TFG/PFG/Results/PROPAGACIÓN/"+str(pC)+"_"+str(pR)+"_NodoconMayorContaminacion"
    if (path.exists(NDir)):
        print("")
    else:
        #shutil.rmtree(NDir)
        os.mkdir(NDir)
    print("Ejecutando iteraciones")
    
    if (Tipo == 'Am'):
        EjecutarIteracionInfeccion(V,'Am')
    elif (Tipo == 'Ni'):
        EjecutarIteracionInfeccion(V,'Ni')
    elif (Tipo == 'Fr'):
        EjecutarIteracionInfeccion(V,'Fr')
    elif (Tipo == 'Ft'):
        EjecutarIteracionInfeccion(V,'Ft')
    elif (Tipo == 'Gt'):
        EjecutarIteracionInfeccion(V,'Gt')
    else:
        EjecutarIteracionInfeccion(V,'Fit')


def PropagarInfeccionTodasProbabilidades(i, Tipo,V):
    global pC, pR
    
    pC=i
    pR=0
    print("Probabilidad contagio " + str(pC))

    PropagarInfeccion2 (pR, pC, Tipo,V)

def Menu1():
    while True:
        print("Choose an option:")
        print("1. Simulate propagation for the global network of rivers")
        print("2. Simulate propagation for each individual non-connected subnetwork")
        print("3. Exit")

        option = input("")
        print("You chose option: ",option)

        if option == "1":
            print("Simulating propagation for the entire network, infecting one node per subnetwork")
            G = CrearGrafo(enlaces_global)
            Menu2(G,"RIOS ESPAÑA")
            return option
        
        elif option == "2":
            print("Simulating propagation for each individual subnetwork")
            
            for enlaces,nombre_rio in rios:
                print(f"Propagar infección en rio {nombre_rio}")
                G = CrearGrafo(enlaces)
                Menu2(G,nombre_rio)
            
            return option
        elif option == "3":
            print("Exiting")
            return option
        else:
            print("Invalid option. Please choose again.")



def Menu2(graph,nombre_rio):
    while True:
        print("Choose an option:")
        print("1. Propagation with infection starting in node with highest nitrate contamination")
        print("2. Propagation with infection starting in node with highest phosphorus contamination")
        print("3. Propagation with infection starting in node with highest phosphate contamination")
        print("4. Propagation with infection starting in node with highest ammonium contamination")
        print("5. Propagation with infection starting in node with highest phytobenthos contamination")
        print("6. Propagation with infection starting in node with highest trophic grade contamination")
        print("7. Propagation with infection starting in node with highest overall contamination")
        print("8. Exit")

        option = input("")

        if option == "1":
            rate = Infected_Nodes(graph)
            while (rate < PERCENTAGE): #and not all(state == "I"):
                PropagarInfeccionTodasProbabilidades(0.5,"Nt",graph)
            break
        elif option == "2":
            rate = Infected_Nodes(graph)
            while (rate < PERCENTAGE): #and not all(state == "I"):
                PropagarInfeccionTodasProbabilidades(0.5,"Fr",graph)
            break
        elif option == "3":
            rate = Infected_Nodes(graph)
            while (rate < PERCENTAGE): #and not all(state == "I"):
                PropagarInfeccionTodasProbabilidades(0.5,"Ft",graph)
            break
        elif option == "4":
            rate = Infected_Nodes(graph)
            while (rate < PERCENTAGE): #and not all(state == "I"):
                PropagarInfeccionTodasProbabilidades(0.5,"Am",graph)
            break
        elif option == "5":
            rate = Infected_Nodes(graph)
            while (rate < PERCENTAGE): #and not all(state == "I"):
                PropagarInfeccionTodasProbabilidades(0.5,"Fit",graph)
            break
        elif option == "6":
            rate = Infected_Nodes(graph)
            while (rate < PERCENTAGE): #and not all(state == "I"):
                PropagarInfeccionTodasProbabilidades(0.5,"Gt",graph)
            break
        elif option == "7":
            break
        else:
            print("Invalid option. Please choose again.")

    return option

# MAIN =============================================================
enlaces_global = "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_20240311_125307_GrafoGlobal.csv"
enlaces_duero = "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_20240311_125307_Grafo_RIO DUERO.csv"
enlaces_tajo = "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_20240311_125307_Grafo_RIO TAJO.csv"
enlaces_guadiana = "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_20240311_125307_Grafo_RIO GUADIANA.csv"
enlaces_jucar = "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_20240311_125307_Grafo_RIO JUCAR.csv"
enlaces_ebro = "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_20240311_125307_Grafo_RIO EBRO.csv"
enlaces_segura = "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_20240311_125307_Grafo_RIO SEGURA.csv"
enlaces_garona = "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_20240311_125307_Grafo_RIO GARONA.csv"
enlaces_barbate = "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_20240311_125307_Grafo_RIO BARBATE.csv"
enlaces_ter =  "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_20240311_125307_Grafo_RIU TER.csv"
enlaces_palancia = "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_20240311_125307_Grafo_RIU PALANCIA.csv"
enlaces_guadalquivir = "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_20240311_125307_Grafo_RIO GUADALQUIVIR.csv"
rios = [(enlaces_guadalquivir,"RIO GUADALQUIVIR"),(enlaces_palancia,"RIO PALANCIA"),(enlaces_ter,"RIO TER"),(enlaces_barbate,"RIO BARBATE"),(enlaces_garona,"RIO GARONA"),(enlaces_segura,"RIO SEGURA"),(enlaces_guadiana,"RIO GUADIANA"),(enlaces_ebro,"RIO EBRO"),(enlaces_jucar,"RIO JUCAR"),(enlaces_tajo,"RIO TAJO")]

fosfato = "/Users/silviadelatorre/Desktop/TFG/DISTANCIAS SENSORES/3 COORDS/ATLANTICO/Fosfato.csv"


option1 = Menu1()


