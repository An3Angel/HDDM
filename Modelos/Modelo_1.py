# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import hddm
data = hddm.load_csv('./Datos_fila_20240927.csv', encoding='utf-8')



############################################
################# LIMPIEZA #################
############################################


# Elimina filas con valores nulos
data = data.dropna()  

# El modelo está hecho para encontrar una columna con el nombre response
# Por lo que se modifican dichas columnas
data = data.rename(columns={'correcta': 'response'})

#Los datos de la columna response deben ser enteros (int)
data['response'] = data['response'].astype(int)

# El modelo necesita que los tiempos de respuesta sean de tipo flotante
data['rt'] = data['rt'].astype(float)
# Los tiempos de respusta deben tener un valor adecuado (no muy alto)
data['rt']=data['rt']/1000

# El modelo solo lee las id como subj_idx
data = data.rename(columns={'id': 'subj_idx'})

'''
Aquí se agregan columnas con las fracciones de weber correspondientes.
'''

# Columnas FW
data['num_min_value'] = data.apply(lambda row: min(row['numAZ'], row['numAM']), axis=1)
data['num_max_value'] = data.apply(lambda row: max(row['numAZ'], row['numAM']), axis=1)
data['numFW']=data['num_min_value']/data['num_max_value']

data['pix_min_value'] = data.apply(lambda row: min(row['pixAZ'], row['pixAM']), axis=1)
data['pix_max_value'] = data.apply(lambda row: max(row['pixAZ'], row['pixAM']), axis=1)
data['pixFW']=data['pix_min_value']/data['pix_max_value']

data['env_min_value'] = data.apply(lambda row: min(row['envAZ'], row['envAM']), axis=1)
data['env_max_value'] = data.apply(lambda row: max(row['envAZ'], row['envAM']), axis=1)
data['envFW']=data['env_min_value']/data['env_max_value']

##########################################
################# MODELO #################
##########################################


m = hddm.HDDMRegressor(data,
                       models=[]) #El corchete vacío implica que no hay regresión (es igual que usar hddm.HDDM() )
m.find_starting_values()
sample = m.sample(5000, burn = 100)
stats_df = m.gen_stats()
m.print_stats()

#################################################
################# VISUALIZACIÓN #################
#################################################

# Filtrar las filas donde el nombre contiene 'v'
v_stats = stats_df[stats_df.index.str.contains('^v', regex=True)]
v_stats.index = v_stats.index.str.encode('ascii', 'ignore').str.decode('utf-8')
plt.figure(figsize=(10, 6))
plt.bar(v_stats.index, v_stats['mean'],
         color='skyblue')
plt.xlabel('Parametro')
plt.ylabel('Media')
plt.title('Medias de los parametros que contienen "v"')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Filtrar las filas donde el nombre contiene 'a'
a_stats = stats_df[stats_df.index.str.contains('^a', regex=True)]
a_stats.index = a_stats.index.str.encode('ascii', 'ignore').str.decode('utf-8')
plt.figure(figsize=(10, 6))
plt.bar(a_stats.index, a_stats['mean'],
         color='skyblue')
plt.xlabel('Parametro')
plt.ylabel('Media')
plt.title('Medias de los parametros que contienen "a (umbral)"')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()