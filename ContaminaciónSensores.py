import os
import pandas as pd
import numpy as np
from shapely.wkt import loads
from shapely.geometry import LineString
import geopandas as gpd
import time

def calcular_distancias(fich_rios, contaminante_sf, fich_output, contaminante_col):
    for idx, rio in fich_rios.iterrows():
        linestring = rio['WKT']
        geometry = loads(linestring)
        # Obtener el número de tramos inicial
        for part in geometry.geoms:
            #Se reducen el número de tramos a 3
            coords = list(part.coords)
            coordinates_to_check = [coords[0], coords[len(coords)//2] if len(coords) % 2 != 0 else coords[len(coords)//2 - 1], coords[-1]]
            #print("Coordinates to check:",coordinates_to_check)
            #print(f"Número de tramos del rio simplificados: {len(coordinates_to_check)}")
            for coordinate in coordinates_to_check:
                #print("Calculando el sensor más cercano al punto: ",coordinate)
                distances = []
                for idx, row in contaminante_sf.iterrows():
                    sensor = row.geometry.coords[0]  # Obtenemos las coordenadas del sensor
                    distance = np.sqrt((coordinate[0] - sensor[0]) ** 2 + (coordinate[1] - sensor[1]) ** 2)
                    distances.append(distance)

                min_distance = min(distances)
                min_distance_index = distances.index(min_distance)
                closest_sensor = contaminante_sf.iloc[min_distance_index]

                # Crear un DataFrame con los datos
                data = {
                    'ID_RIO': [rio['PFAFRIO']],
                    'coord_TRAMO': [coordinate],
                    f'valor_{contaminante_col}': [closest_sensor['PromedioDe']]
                }
                info_tramo = pd.DataFrame(data)

                # Agregar datos al archivo CSV
                # Verificar si el archivo existe
                if os.path.exists(fich_output):
                    # El archivo existe, no escribir un encabezado
                    header = False
                else:
                    header = True

                # Escribir datos en el archivo CSV
                info_tramo.to_csv(fich_output, mode='a', header=header, index=False)

# Lectura de los datos de entrada
df_rios_A = pd.read_csv("/Users/silviadelatorre/Desktop/TFG/FICHEROS INPUT/RiosAtlantico.csv")
df_rios_M = pd.read_csv("/Users/silviadelatorre/Desktop/TFG/FICHEROS INPUT/RiosMediterraneo.csv")
df_amonio = pd.read_csv("/Users/silviadelatorre/Desktop/TFG/AGENTES CONTAMINANTES/Amonio.csv")
df_nitrato = pd.read_csv("/Users/silviadelatorre/Desktop/TFG/AGENTES CONTAMINANTES/Nitratos.csv")
df_fosforo = pd.read_csv("/Users/silviadelatorre/Desktop/TFG/AGENTES CONTAMINANTES/Fosforo.csv")
df_fosfato = pd.read_csv("/Users/silviadelatorre/Desktop/TFG/AGENTES CONTAMINANTES/Fosfato.csv")
df_grado_trofico = pd.read_csv("/Users/silviadelatorre/Desktop/TFG/AGENTES CONTAMINANTES/Grafo_Trofico.csv")
df_fitobentos = pd.read_csv("/Users/silviadelatorre/Desktop/TFG/AGENTES CONTAMINANTES/Fitobentos.csv")

# Convertir en objetos GeoDataFrame
df_rios_A_gdf = gpd.GeoDataFrame(df_rios_A, geometry=gpd.GeoSeries.from_wkt(df_rios_A['WKT']))
df_rios_M_gdf = gpd.GeoDataFrame(df_rios_M, geometry=gpd.GeoSeries.from_wkt(df_rios_M['WKT']))
# Formatear archivos
amonio_gdf = gpd.GeoDataFrame(df_amonio, geometry=gpd.GeoSeries.from_wkt(df_amonio['WKT']))
nitrato_gdf = gpd.GeoDataFrame(df_nitrato, geometry=gpd.GeoSeries.from_wkt(df_nitrato['WKT']))
fosforo_gdf = gpd.GeoDataFrame(df_fosforo, geometry=gpd.GeoSeries.from_wkt(df_fosforo['WKT']))
fosfato_gdf = gpd.GeoDataFrame(df_fosfato, geometry=gpd.GeoSeries.from_wkt(df_fosfato['WKT']))
grado_trofico_gdf = gpd.GeoDataFrame(df_grado_trofico, geometry=gpd.GeoSeries.from_wkt(df_grado_trofico['WKT']))
fitobentos_gdf = gpd.GeoDataFrame(df_fitobentos, geometry=gpd.GeoSeries.from_wkt(df_fitobentos['WKT']))


# FICHEROS INPUT RIOS
fich_rios_list = [df_rios_A_gdf, df_rios_M_gdf]

output_folder_a = "/Users/silviadelatorre/Desktop/TFG/DISTANCIAS SENSORES/3 COORDS/ATLANTICO"
output_folder_m = "/Users/silviadelatorre/Desktop/TFG/DISTANCIAS SENSORES/3 COORDS/MEDITERRANEO"


contaminantesResult_list = [(fitobentos_gdf,"Fitobentos"),(nitrato_gdf,"Nitrato"),(amonio_gdf,"Amonio"),(fosforo_gdf,"Fosforo"),(fosfato_gdf,"Fosfato"),(grado_trofico_gdf,"Grado Trofico")]

# Obtener el tiempo de inicio
start_time = time.time()

for vertiente in fich_rios_list:
    if vertiente is df_rios_A_gdf:
        output_folder = output_folder_a
    elif vertiente is df_rios_M_gdf:
        output_folder = output_folder_m
    for contaminante_sf, contaminante_col in contaminantesResult_list:
        print(contaminante_col)
        output_path = os.path.join(output_folder, f"{contaminante_col}.csv")
        calcular_distancias(vertiente, contaminante_sf, output_path,contaminante_col)
#calcular_distancias(df_rios_M_gdf, amonio_gdf, contaminacion_amonio,"Amonio")
# Obtener el tiempo de finalización
end_time = time.time()
elapsed_time = end_time - start_time
hours = elapsed_time/3600
print(f"Tiempo transcurrido: {hours} horas")