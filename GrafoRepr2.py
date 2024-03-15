# PACKAGES
import networkx as nx
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from operator import itemgetter
from pyvis.network import Network
import sys
import datetime
import psutil
import time
import cProfile
import netwulf as nw

# FUNCTIONS
def leer_ficheros(file,data):
    name_file = open(file)
    reader = csv.reader(name_file)

    header = next(reader)  # first line is header
    for row in reader:  # modify data types
        # ['OBJECTID', 'cod_mar', 'PFAFRIO', 'COD_UNI', 'nom_rio', 'Shape_Leng', 'color']
        Object_ID = int(row[0])
        Cod_Mar = str(row[1])
        Pfafrio = int(row[2])
        Cod_uni = int(row[3])
        Nombre_Rio = str(row[4])
        Long_Rio = float(row[5])
        Color = int(row[6])

        data.append([Object_ID, Cod_Mar, Pfafrio, Cod_uni, Nombre_Rio, Long_Rio, Color])

def filtrar_rios(data):
    rios_ppales = []
    for sublista in data:
        # Verificar las condiciones
        if sublista[-1] == 0 and (str(sublista[2]).startswith('100') or str(sublista[2]).startswith('200')):
            rios_ppales.append(sublista)
        if sublista[2] in main_codes:
            rios_ppales.append(sublista)
            
    return rios_ppales

def conectar_con_nivel_anterior(G, niveles_de_rios, nivel, pfafrio, nombre):
    for nivel_anterior in range(nivel - 1, -1, -1):
        rios_nivel_ant = niveles_de_rios[nivel_anterior]  # Ríos del nivel anterior (lista de diccionarios)
        encontrado = False
        for dic in rios_nivel_ant:
            #print(f'Diccionario rios nivel anterior: {dic}')
            if str(pfafrio).startswith(str(dic['codigo'])):
                #print(f'CODIGO: {pfafrio}')
                G.add_edge(dic['codigo'], pfafrio)
                GrafoGlobal.add_edge(dic['codigo'], pfafrio)
                #print(f'Enlaces: {G.edges()}')
                encontrado = True
                break
        if encontrado:
            break

def GraficarRed(G,rio):
    
    print("Sacar imagen del gráfico usando netwulf")
    #pos = nx.kamada_kawai_layout(G)
    #print("kamada layout hecho")
    
    nw.visualize(G, config={'NodeLink': {'hierarchical': True}})

    

def Creacion_Grafo(rio):
    print(" ===== GRAFO DE:  "+str(rio[4]))
    G = nx.DiGraph()

    # Inicializa un diccionario vacío para almacenar los ríos por nivel
    niveles_de_rios = {}
    dic_nodos = {}

    # Agregar los afluentes del rio principal de manera jerárquica
    for nivel in range(14):
        print(" === Introducimos nodos de nivel "+str(nivel))
        # Verifica si el nivel ya está en el diccionario y, si no, crea una lista vacía
        if nivel not in niveles_de_rios:
            niveles_de_rios[nivel] = []
            dic_nodos[nivel] = []

        for sublista in data:
            nombre = sublista[4]
            nivel_afluente = sublista[-1]
            pfafrio = sublista[2]
            vertiente = sublista[1]

            # Iterar a través de los datos y agrégarlos al grafo
            if vertiente == rio[1] and nivel_afluente == nivel and str(pfafrio).startswith(str(rio[2])):
                #nodo_nombre = f'{pfafrio}-{nombre}'
                niveles_de_rios[nivel].append({'nombre': nombre, 'codigo': pfafrio})
                #print(f'Niveles: {niveles_de_rios[nivel]}')
                G.add_node(pfafrio)
                #print(f'Añadir nodo a la red: {G.nodes()}\n')
                GrafoGlobal.add_node(pfafrio)
                if (nivel_afluente != rio[-1]):
                    #print("Conectar con desembocadura (nivel anterior)")
                    conectar_con_nivel_anterior(G, niveles_de_rios, nivel, pfafrio, nombre)

    # Invertir la dirección de las aristas del grafo
    #G_invertido = G.reverse()
    #print(f'Enlaces invertidos: {G.edges()}')

    print("RESUMEN DEL RIO\n")
    number_nodes = nx.number_of_nodes(G)
    print(f'Nodos: {number_nodes}')
    number_edges = nx.number_of_edges(G)
    print(f'Enlaces: {number_edges}')

    nx.write_edgelist(G, "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/SIN COORDENADAS/Edgelist_"+fecha_hora_actual+"_Grafo_"+rio[4]+".csv")
    print("Lista de enlaces guardada...\n")

    print("Grafica usando kamada layout")
    
    #GraficarRed(G,str(rio[4]))

    


def CalculoParametros(GrafoGlobal, fecha_hora_actual):
    # Calcular la heterogeneidad
    number_nodes = nx.number_of_nodes(GrafoGlobal)
    number_edges = nx.number_of_edges(GrafoGlobal)
    degrees = [val for (node, val) in GrafoGlobal.degree()]
    max_degree = np.max(degrees)
    min_degree = np.min(degrees)
    mean_degree = np.mean(degrees)
    freq_degree = stats.mode(degrees)

    # COMPONENTE GIGANTE
    print("PARÁMETRO COMPONENTE GIGANTE =================")
    # Check whether the graph is weakly connected or not
    if nx.is_weakly_connected(GrafoGlobal):
        graph_connected = 'TRUE'
        connected_components = nx.number_weakly_connected_components(GrafoGlobal)
        Gcc = sorted(nx.weakly_connected_components(GrafoGlobal), key=len, reverse=True)
        largest_subgraph = GrafoGlobal.subgraph(Gcc[0])
        largest_graph_connected = 'TRUE' if nx.is_weakly_connected(largest_subgraph) else 'FALSE'
        diameter_largest = nx.diameter(largest_subgraph)
    else:
        graph_connected = 'FALSE'
        connected_components = 0
        largest_subgraph = GrafoGlobal
        largest_graph_connected = 'FALSE'
        diameter_largest = 0

    number_nodes_largest = largest_subgraph.number_of_nodes()
    number_edges_largest = largest_subgraph.number_of_edges()

    # MEDIDAS CENTRALIDAD
    print("MEDIDAS CENTRALIDAD =================\n")
    print("centrality\n")
    graph_centrality = nx.degree_centrality(largest_subgraph)
    max_de = max(graph_centrality.items(), key=itemgetter(1))
    print("closeness\n")
    graph_closeness = nx.closeness_centrality(largest_subgraph)
    max_clo = max(graph_closeness.items(), key=itemgetter(1))
    print("betweenness\n")
    '''graph_betweenness = nx.betweenness_centrality(largest_subgraph, normalized=True, endpoints=False)
    max_bet = max(graph_betweenness.items(), key=itemgetter(1))'''
    print("eigenvector\n")
    graph_eigenvector = nx.eigenvector_centrality(GrafoGlobal)
    print("pagerank\n")
    graph_pagerank = nx.pagerank(GrafoGlobal)
    print("k-core\n")
    # K - CORE
    core_number = nx.core_number(GrafoGlobal)
    k_core = nx.k_core(GrafoGlobal)
    print("Escribiendo parámetros en fichero\n")

    # Guardar los resultados en un archivo
    with open(f"/Users/silviadelatorre/Desktop/TFG/PARAMETROS/SIN COORDENADAS/Parametros_{fecha_hora_actual}_R1.txt", "w") as archivo:
        archivo.write("PARÁMETROS DE HETEROGENEIDAD =================\n")
        archivo.write(f"Number of nodes in the graph: {number_nodes}\n")
        archivo.write(f"Number of edges in the graph: {number_edges}\n")
        archivo.write(f"Maximum degree of the Graph: {max_degree}\n")
        archivo.write(f"Minimum degree of the Graph: {min_degree}\n")
        archivo.write(f"Average degree of the nodes in the Graph: {mean_degree}\n")
        archivo.write(f"Most frequent degree of the nodes in the Graph: {freq_degree.mode[0]}\n")
        archivo.write("PARÁMETROS COMPONENTE GIGANTE =================\n")
        archivo.write(f"Connected graph: {graph_connected}\n")
        archivo.write(f"Number of connected components: {connected_components}\n")
        archivo.write(f"Number of nodes of the largest subgraph in the graph: {number_nodes_largest}\n")
        archivo.write(f"Number of edges of the largest subgraph in the graph: {number_edges_largest}\n")
        archivo.write(f"Connected largest subgraph: {largest_graph_connected}\n")
        archivo.write(f"Diameter of the largest subgraph: {diameter_largest}\n")
        archivo.write("PARÁMETROS DE CENTRALIDAD =================\n")
        archivo.write("GRAPH CENTRALITY:\n")
        for node, centrality in graph_centrality.items():
            archivo.write(f" Nodo {node}: {centrality}\n")
        archivo.write(f"Maximum Graph Centrality: {max_de}\n")

        archivo.write("GRAPH CLOSENESS:\n")
        for node, closeness in graph_closeness.items():
            archivo.write(f"Nodo {node}: {closeness}\n")
        archivo.write(f"Maximum Graph Closeness: {max_clo}\n")

        '''archivo.write("GRAPH BETWEENNESS:\n")
        for node, betweenness in graph_betweenness.items():
            archivo.write(f"Nodo {node}: {betweenness}\n")
        archivo.write(f"Maximum Graph Betweenness: {max_bet}\n")'''

        archivo.write("GRAPH EIGENVECTOR:\n")
        for node, eigenvector in graph_eigenvector.items():
            archivo.write(f"Nodo {node}: {eigenvector}\n")

        archivo.write("GRAP PAGERANK:\n")
        for node, pagerank in graph_pagerank.items():
            archivo.write(f"Nodo {node}: {pagerank}\n")

        archivo.write("PARÁMETROS K-CORE ========================\n")
        archivo.write(f"K-Core del grafo: {k_core}\n")
        archivo.write(f"Core Number: {core_number}")

        print("Fin escritura fichero\n")

# Función para monitorear el uso de recursos del sistema
def monitor_system_resources():
    while True:
        # Obtenemos el uso de CPU y memoria
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Mostramos los resultados
        print(f"Uso de CPU: {cpu_percent}% - Uso de memoria: {memory_percent}%")
        
        # Esperamos un segundo antes de volver a verificar
        time.sleep(100)


# IMPORT DATA =============================================================
A_original = '/Users/silviadelatorre/Desktop/TFG/FICHEROS INPUT/Atlantico.csv'
M_original = '/Users/silviadelatorre/Desktop/TFG/FICHEROS INPUT/M_RiosCompletosv2.csv'
Rios_Proces = '/Users/silviadelatorre/Desktop/TFG/RiosProcesable.csv'
data = []
csv.field_size_limit(sys.maxsize)
# Obtener la fecha y hora actual
fecha_hora_actual = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Llamamos a la función para comenzar a monitorear
#monitor_system_resources()

# READ FILES
print("Leyendo ficheros")
leer_ficheros(A_original,data)
leer_ficheros(M_original,data)

main_codes = [10034,10038,10098,10094,20054,20052,20036,20016]
# Filtrar los ríos según las condiciones especificadas
rios_filtrados = filtrar_rios(data)
#print(rios_filtrados)
lista_rios_ordenados = sorted(rios_filtrados, key=lambda x: x[-2], reverse=True)


print("==== RIOS PRINCIPALES PENSINSULARES ORDENADOS DE MAYOR A MENOR LONGITUD==== ")
for i, rio in enumerate(lista_rios_ordenados, start=1):
    print(f"Río {i}: {rio[4]} - Longitud: {rio[5]}")

# CREACIÓN DEL GRAFO DIRIGIDO
GrafoGlobal = nx.DiGraph()

for rio in lista_rios_ordenados:
    Creacion_Grafo(rio)

# GUARDAR LISTA DE ENLACES
# Escribe el grafo global en un archivo de lista de aristas
#nx.write_edgelist(GrafoGlobal, "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/Edgelist_"+fecha_hora_actual+"_GrafoGlobal.csv")

#print("Lista de enlaces red global guardada...\n")


#print("Calculando parámetros estructurales...")
#CalculoParametros(GrafoGlobal,fecha_hora_actual)

