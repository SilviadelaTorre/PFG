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
    
    nw.visualize(G)

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

def Creacion_Grafo(rio):
    
    print(f'=======GRAFO DE {rio[5]}: {rio[3]}')
    G = nx.DiGraph()

    #data.append([wkt_simplificada, Object_ID, Cod_Mar, Pfafrio, Cod_uni, Nombre_Rio, Long_Rio, Color])

    # Inicializa un diccionario vacío para almacenar los ríos por nivel
    niveles_de_rios = {}

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

            # Iterar a través de los datos y agrégarlos al grafo
            if vertiente == rio[2] and nivel_afluente == nivel and str(pfafrio).startswith(str(rio[3])):
                #print(pfafrio)
                #coordenadas = extract_coordinates(wkt)
                #print(type(coordenadas))
                #print(f'Numero de tramos: {len(coordenadas)}')
                
                nodo_con_rio_ant = obtener_primer_nodo(wkt)
                nodo_con_rio_sig = obtener_ultimo_nodo(wkt)

                #print(f'PRIMER TRAMO: {nodo_con_rio_ant}')
                crear_enlaces(wkt, G)
                #print(f'ULTIMO TRAMO: {nodo_con_rio_sig}')

                niveles_de_rios[nivel].append({'nombre': nombre, 'codigo': pfafrio, 'tramo':wkt, 'tramo1':nodo_con_rio_ant, 'tramo_last':nodo_con_rio_sig})

                # Niveles > 0 hay que crear enlace con nodos del nivel anterior
                if(nivel_afluente != rio[-1]):
                    conectar_con_nivel_anterior(G, niveles_de_rios, nivel, pfafrio, nodo_con_rio_ant, nombre)

    
    # Invertir la dirección de las aristas del grafo
    #G_invertido = G.reverse()
    print("RESUMEN DEL RIO\n")

    number_nodes = nx.number_of_nodes(G)
    print(f'Nodos: {number_nodes}')
    number_edges = nx.number_of_edges(G)
    print(f'Enlaces: {number_edges}')

    nx.write_edgelist(G, "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_"+fecha_hora_actual+"_Grafo_"+rio[5]+".csv",delimiter=';')
    print("Lista de enlaces guardada...\n")
    

    print("Graficando...")
    
    # net=Network(notebook=True,cdn_resources='remote')

    # print("Graficar usando pyvis Network")
    # # Convertir identificadores de nodos a cadenas de texto
    # node_strings = [str(node) for node in G.nodes()]
    # # Convertir las aristas a tuplas de cadenas de texto
    # edges_strings = [(str(edge[0]), str(edge[1])) for edge in G.edges()]

    print("Sacar imagen del gráfico")
    degrees = G.degree()  # Dict with Node ID, Degree
    nodes = G.nodes()
    n_color = np.asarray([degrees[n] for n in nodes])

    pos = nx.kamada_kawai_layout(G)

    plt.figure(figsize=(20, 20))
    nx.draw(G, pos=pos, node_color=n_color, cmap=plt.cm.jet, edge_color="grey", node_size=60,
            with_labels=False)
    plt.savefig('/Users/silviadelatorre/Desktop/TFG/GRAFICOS REDES/3 COORDS/'+rio[5]+'.png',dpi=300)
    plt.show()
    GraficarRed(G,str(rio[5]))

def CalculoParametros(GrafoGlobal, fecha_hora_actual, nombre_rio):
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
    core_number = nx.core_number(GrafoGlobal)
    k_core = nx.k_core(GrafoGlobal)
    print("Escribiendo parámetros en fichero\n")

    # Guardar los resultados en un archivo
    with open(f"/Users/silviadelatorre/Desktop/TFG/PARAMETROS/3 COORDS/Parametros_{fecha_hora_actual}_GrafoGlobal.txt", "w") as archivo:
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

def TablaRios(rio):

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

for rio in lista_rios_ordenados:
    Creacion_Grafo(rio)
    TablaRios(rio)


# # GUARDAR LISTA DE ENLACES
# # Escribe el grafo global en un archivo de lista de aristas
# nx.write_edgelist(GrafoGlobal, "/Users/silviadelatorre/Desktop/TFG/EDGE LIST/3 COORDS/Edgelist_"+fecha_hora_actual+"_GrafoGlobal.csv")

# print("Lista de enlaces red global guardada...\n")

# print("Calculando parámetros estructurales...")
# CalculoParametros(GrafoGlobal,fecha_hora_actual,rios_filtrados[0][5])

# print("Graficar GLOBAL\n")
# GraficarRed(GrafoGlobal,"Rios_España")
# nx.draw(GrafoGlobal, with_labels=False, node_color='skyblue', font_color='black', node_size=800)

# print("Graficar en cytoscape la red de rios")
# # Create a Cytoscape network from the NetworkX graph
# cy.networks.create_network_from_networkx(GrafoGlobal)

