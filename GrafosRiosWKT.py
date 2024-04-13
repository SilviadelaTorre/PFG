# PACKAGES
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from operator import itemgetter
import py4cytoscape as cy
import datetime
from pyvis.network import Network
import networkx as nx
import shapely
from shapely.wkt import loads
import geopandas as gpd
from shapely.geometry import MultiLineString
import pandas as pd
import netwulf as nw
import os

# FUNCTIONS
def leer_ficheros(file,data):
    name_file = open(file)
    reader = csv.reader(name_file)

    header = next(reader)  # first line is header
    for row in reader:  # modify data types
        # ['OBJECTID', 'cod_mar', 'PFAFRIO', 'COD_UNI', 'nom_rio', 'Shape_Leng', 'color']
        WKT = row[0]

        multilinestring = loads(WKT)

        Object_ID = int(row[1])
        Cod_Mar = str(row[2])
        Pfafrio = int(row[3])
        Cod_uni = int(row[4])
        Nombre_Rio = str(row[5])
        Long_Rio = float(row[6])
        Color = int(row[7])

        wkt_simplificada = reducir_numCoordenadas(multilinestring)

        data.append([wkt_simplificada, Object_ID, Cod_Mar, Pfafrio, Cod_uni, Nombre_Rio, Long_Rio, Color])
        #print(data)

    print("Fin leyendo el fichero")

def reducir_numCoordenadas(wkt_multilinestring):
    for part in wkt_multilinestring.geoms:
        #Nos quedamos con la primera, la del medio y la ultima coordenada del tramo
        coords = list(part.coords)
        multilinestring_simplificada = [coords[0], coords[len(coords)//2] if len(coords) % 2 != 0 else coords[len(coords)//2 - 1], coords[-1]]
    return multilinestring_simplificada

def filtrar_rios(data):
    rios_ppales = []
    for sublista in data:
        # Verificar las condiciones
        if sublista[-1] == 0 and (str(sublista[3]).startswith('100') or str(sublista[3]).startswith('200')):
            rios_ppales.append(sublista)
            print(sublista[3])
        if sublista[3] in main_codes:
            rios_ppales.append(sublista)
            print(sublista[3])
    return rios_ppales

def GraficarRed(G,rio):
    config = {
    'NodeLink': {
        'sourceField': 'source',  # Field name for the source coordinates
        'targetField': 'target',  # Field name for the target coordinates
        'nodeSize': 4,             # Node size
        'nodeColor': 'blue',       # Node color
        'linkWidth': 2,            # Link width
        'linkColor': 'gray'        # Link color
    }
    }

    print("Sacar imagen del gráfico usando netwulf")
    #pos = nx.kamada_kawai_layout(G)
    #print("kamada layout hecho")
    
    nw.visualize(G,config=config)

def crear_enlaces(coords,Grafo):
    '''
    Crear nodos y enlaces en cada tramo del rio
    :param coords: Par de coordenadas de cada tramo
    :return:
    '''
    
    for i in range(len(coords)-1):
        nodo_origen = str(coords[i])
        #print(f'Nodo origen: {nodo_origen}')
        nodo_destino = str(coords[i + 1])
        #print(f'Nodo Destino: {nodo_destino}')
        
        # Agregar nodos al grafo
        Grafo.add_node(nodo_origen)
        Grafo.add_node(nodo_destino)

        # Agregar enlace entre tramos de la misma línea
        Grafo.add_edge(nodo_origen, nodo_destino)


def obtener_primer_nodo(coords):
    return coords[0]

def obtener_ultimo_nodo(coords):
    return coords[-1]

def extract_coordinates(wkt):
    """Extracts coordinates from either LineString or MultiLineString geometry.
    Args:
        wkt: A Shapely geometry object (LineString or MultiLineString).
    Returns:
        A list of coordinates represented as tuples.
    """

    if isinstance(wkt, shapely.geometry.linestring.LineString):
        # Single-segment LineString: access coordinates directly
        return wkt.coords

    elif isinstance(wkt, shapely.geometry.multilinestring.MultiLineString):
        # Multi-segment MultiLineString: loop through individual LineStrings
        coordinates = []
        for line in wkt.geoms:
            coordinates.extend(line.coords)  # Append coordinates from each line
        return coordinates

    else:
        raise ValueError("Unsupported geometry type: {}".format(type(wkt)))
    
def conectar_con_nivel_anterior(G, niveles_de_rios, nivel, pfafrio, nodo_con_rio_ant, nombre):
    for nivel_anterior in range(nivel - 1, -1, -1):
        rios_nivel_ant = niveles_de_rios[nivel_anterior]  # Ríos del nivel anterior (lista de diccionarios)
        encontrado = False
        for dic in rios_nivel_ant:
            #print(f'Diccionario rios nivel anterior: {dic}')
            if str(pfafrio).startswith(str(dic['codigo'])):
                #print(f'CODIGO: {pfafrio}')
                G.add_edge(dic['tramo_last'], nodo_con_rio_ant, etiqueta=nombre)
                GrafoGlobal.add_edge(dic['tramo_last'], nodo_con_rio_ant, etiqueta=nombre)
                encontrado = True
                break
        if encontrado:
            break

def Creacion_Grafo(rio,tabla_rios):
    
    print(f'=======GRAFO DE {rio[5]}: {rio[3]}')
    G = nx.DiGraph()

    #data.append([wkt_simplificada, Object_ID, Cod_Mar, Pfafrio, Cod_uni, Nombre_Rio, Long_Rio, Color])

    # Inicializa un diccionario vacío para almacenar los ríos por nivel
    niveles_de_rios = {}
    longitud_total = 0
    #Agregar los afluentes del rio principal de manera jerárquica
    for nivel in range(14):
        print(" === Introducimos nodos de nivel "+str(nivel))
        # Verifica si el nivel ya está en el diccionario y, si no, crea una lista vacía
        if nivel not in niveles_de_rios:
            niveles_de_rios[nivel] = []

        for lista in data:
            wkt = lista[0]
            nombre = lista[5]
            nivel_afluente = lista[-1]
            pfafrio = lista[3]
            vertiente = lista[2]
            longitud = lista[6]

            # Iterar a través de los datos y agrégarlos al grafo
            if vertiente == rio[2] and nivel_afluente == nivel and str(pfafrio).startswith(str(rio[3])):
                #print(pfafrio)
                #coordenadas = extract_coordinates(wkt)
                #print(type(coordenadas))
                #print(f'Numero de tramos: {len(coordenadas)}')
                
                nodo_con_rio_ant = obtener_primer_nodo(wkt)
                nodo_con_rio_sig = obtener_ultimo_nodo(wkt)

                #print(f'PRIMER TRAMO: {nodo_con_rio_ant}')
                longitud_total += longitud_total + longitud
                crear_enlaces(wkt, G)
                #print(f'ULTIMO TRAMO: {nodo_con_rio_sig}')

                niveles_de_rios[nivel].append({'nombre': nombre, 'codigo': pfafrio, 'tramo':wkt, 'tramo1':nodo_con_rio_ant, 'tramo_last':nodo_con_rio_sig})

                # Niveles > 0 hay que crear enlace con nodos del nivel anterior
                if(nivel_afluente != rio[-1]):
                    conectar_con_nivel_anterior(G, niveles_de_rios, nivel, pfafrio, nodo_con_rio_ant, nombre)

    
    # Invertir la dirección de las aristas del grafo
    #G_invertido = G.reverse()
    print("RESUMEN DEL RIO\n")
    #AÑADE LOS DATOS A LA TABLA DE RIOS con los datos de cada grafo (Nombre, Codigo rio, Vertiente, Cuenca, Numero de nodos, Longitud total)
    tabla_rios.append({
        'Nombre':rio[5],
        'Codigo rio':rio[3],
        'Vertiente':rio[2],
        'Numero de nodos':nx.number_of_nodes(G),
        'Longitud total':longitud_total})

    print(tabla_rios)

    print("calcular parametros estructural del grafo\n")

    # nx.write_edgelist(G, "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_"+fecha_hora_actual+"_Grafo_"+rio[5]+".csv",delimiter=';')
    # print("Lista de enlaces guardada...\n")
    # net=Network(notebook=True,cdn_resources='remote')
    # print("Graficar usando pyvis Network")
    # # Convertir identificadores de nodos a cadenas de texto
    # node_strings = [str(node) for node in G.nodes()]
    # # Convertir las aristas a tuplas de cadenas de texto
    # edges_strings = [(str(edge[0]), str(edge[1])) for edge in G.edges()]
    G.remove_edges_from(nx.selfloop_edges(G))
    
    return G

def CalculoParametros(GrafoGlobal, nombre_rio,df):
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
    #print("eigenvector\n")
    #graph_eigenvector = nx.eigenvector_centrality(GrafoGlobal)
    print("pagerank\n")
    graph_pagerank = nx.pagerank(GrafoGlobal)
    print("k-core\n")
    # K - CORE
    GrafoGlobal.remove_edges_from(nx.selfloop_edges(GrafoGlobal))
    core_number = nx.core_number(GrafoGlobal)
    k_core = nx.k_core(GrafoGlobal)
    print("Escribiendo parámetros en fichero\n")

    directorio_parametros = "/Users/silviadelatorre/Desktop/TFG/PFG/Results/PARAMETROS/3 COORDS/"

    # Verificar si el directorio ya existe
    if not os.path.exists(directorio_parametros):
        os.makedirs(directorio_parametros)  # Crea el directorio si no existe

    file_path = os.path.join(directorio_parametros, f'{nombre_rio}.txt')

    # Añadir los resultados al DataFrame
    new_row = pd.DataFrame({
        'Nombre del Río': [nombre_rio],
        'Número de Nodos': [number_nodes],
        'Número de Aristas': [number_edges],
        'Grado Máximo': [max_degree],
        'Grado Mínimo': [min_degree],
        'Grado Promedio': [mean_degree],
        'Grado Más Frecuente': [freq_degree[0]],
        'Conectado': [graph_connected],
        'Componentes Conectados': [connected_components],
        'Nodos en Subgrafo Grande': [number_nodes_largest],
        'Aristas en Subgrafo Grande': [number_edges_largest],
        'Subgrafo Grande Conectado': [largest_graph_connected],
        'Diámetro Subgrafo Grande': [diameter_largest],
        'Centralidad Máxima': [max_de],
        'Cercanía Máxima': [max_clo]
    })
    
    df = pd.concat([df, new_row], ignore_index=True)

    with open(file_path, "w") as archivo:
        archivo.write("PARÁMETROS DE HETEROGENEIDAD =================\n")
        archivo.write(f"Number of nodes in the graph: {number_nodes}\n")
        archivo.write(f"Number of edges in the graph: {number_edges}\n")
        archivo.write(f"Maximum degree of the Graph: {max_degree}\n")
        archivo.write(f"Minimum degree of the Graph: {min_degree}\n")
        archivo.write(f"Average degree of the nodes in the Graph: {mean_degree}\n")
        #archivo.write(f"Most frequent degree of the nodes in the Graph: {freq_degree.mode[0]}\n")
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
        archivo.write(f"Maximum Graph Betweenness: {max_bet}\n")

        archivo.write("GRAPH EIGENVECTOR:\n")
        for node, eigenvector in graph_eigenvector.items():
            archivo.write(f"Nodo {node}: {eigenvector}\n")'''

        archivo.write("GRAP PAGERANK:\n")
        for node, pagerank in graph_pagerank.items():
            archivo.write(f"Nodo {node}: {pagerank}\n")

        archivo.write("PARÁMETROS K-CORE ========================\n")
        archivo.write(f"K-Core del grafo: {k_core}\n")
        archivo.write(f"Core Number: {core_number}")

        print("Fin escritura fichero\n")
    return df
# ============================================================================
# ============================================================================
# ============================================================================

csv.field_size_limit(sys.maxsize)
# Obtener la fecha y hora actual
fecha_hora_actual = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# IMPORT DATA =============================================================
A_original = '/Users/silviadelatorre/Desktop/TFG/FICHEROS INPUT/RiosAtlantico.csv'
M_original = '/Users/silviadelatorre/Desktop/TFG/FICHEROS INPUT/RiosMediterraneo.csv'
Rios_Proces = '/Users/silviadelatorre/Desktop/TFG/FICHEROS INPUT/RiosProcesable.csv'

data = []
tabla_rios = []
# READ FILES
leer_ficheros(A_original,data)
leer_ficheros(M_original,data)

main_codes = [10034,10038,10098,10094,20054,20052,20036,20016]
# Filtrar los ríos según las condiciones especificadas
rios_filtrados = filtrar_rios(data)
lista_rios_ordenados = sorted(rios_filtrados, key=lambda x: x[-2], reverse=True)

print("==== RIOS PRINCIPALES PENSINSULARES==== ")
for i, rio in enumerate(lista_rios_ordenados, start=1):
    print(f"Río {i}: {rio[5]} - Longitud: {rio[6]}")

# CREACIÓN DEL GRAFO DIRIGIDO
GrafoGlobal = nx.DiGraph()

# Inicializa el DataFrame fuera de la función si aún no existe
columnas = ['Nombre del Río', 'Número de Nodos', 'Número de Aristas', 'Grado Máximo', 'Grado Mínimo', 'Grado Promedio',
            'Grado Más Frecuente', 'Conectado', 'Componentes Conectados', 'Nodos en Subgrafo Grande', 'Aristas en Subgrafo Grande',
            'Subgrafo Grande Conectado', 'Diámetro Subgrafo Grande', 'Centralidad Máxima', 'Cercanía Máxima']
df_rios = pd.DataFrame(columns=columnas)

for rio in lista_rios_ordenados:
    grafo = Creacion_Grafo(rio,tabla_rios)
    df_rios = CalculoParametros(grafo, rio[5], df_rios)

# exportar tabla de rios
#crear directorio si no existe
directorio_propiedades = '/Users/silviadelatorre/Desktop/TFG/PFG/Results/PARAMETROS/3 COORDS/PropiedadesEstructurales.csv'
if not os.path.exists(directorio_propiedades):
    os.makedirs(directorio_propiedades)  # Crea el directorio si no existe

df_rios = pd.DataFrame(columns=columnas)
df_rios.to_csv(directorio_propiedades, index=False)
print("Graficar en cytoscape la red")
# Create a Cytoscape network from the NetworkX graph


# df = pd.DataFrame(tabla_rios)
# #Exportar tabla de rios
# df.to_csv('/Users/silviadelatorre/Desktop/TFG/PFG/Results/TablaGrafos.csv', index=False)


# # # GUARDAR LISTA DE ENLACES
# # Escribe el grafo global en un archivo de lista de aristas
# nx.write_edgelist(GrafoGlobal, "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_"+fecha_hora_actual+"_GrafoGlobal.csv")

# print("Lista de enlaces red global guardada...\n")

# print("Calculando parámetros estructurales...")
# CalculoParametros(GrafoGlobal,fecha_hora_actual,rios_filtrados[0][5])

# print("Graficar GLOBAL\n")
# GraficarRed(GrafoGlobal,"Rios_España")
# nx.draw(GrafoGlobal, with_labels=False, node_color='skyblue', font_color='black', node_size=800)



