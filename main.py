import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title("PIDA")
st.title("KPI-1")
st.title("Reducir en 5% la tasa de mortalidad a nivel anual")

# Cargar los datos desde el archivo.
df = pd.read_csv("data_final.csv", index_col=False, sep=",", header=0)

# Normalizando
df['fecha'] = pd.to_datetime(df['fecha'])
df['all_aboard'] = df['all_aboard'].replace('?', np.nan).astype(float).astype('Int64')
df['cantidad de fallecidos'] = df['cantidad de fallecidos'].replace('?', np.nan).astype(float).astype('Int64')

# Limpiar los valores incorrectos o faltantes en las columnas relevantes
df['cantidad de fallecidos'] = pd.to_numeric(df['cantidad de fallecidos'], errors='coerce')
df['all_aboard'] = pd.to_numeric(df['all_aboard'], errors='coerce')

# Crear una nueva columna "Año" a partir de la columna de fechas
df['Año'] = df['fecha'].dt.year

# Calcular la tasa de mortalidad por año utilizando groupby
df_tasa_mortalidad = df.groupby('Año')[['cantidad de fallecidos', 'all_aboard']].sum()
df_tasa_mortalidad['tasa_mortalidad'] = (df_tasa_mortalidad['cantidad de fallecidos'] / df_tasa_mortalidad[
    'all_aboard']) * 100

# Obtener el rango de años para el filtro interactivo
año_min = int(df_tasa_mortalidad.index.min())
año_max = int(df_tasa_mortalidad.index.max())

# Filtro interactivo por año
selected_year = st.slider('Seleccione un año', min_value=año_min, max_value=año_max, value=año_max, step=1)

# Filtrar los datos para los años seleccionados
df_selected_years = df_tasa_mortalidad[(df_tasa_mortalidad.index >= año_min) & (df_tasa_mortalidad.index <= selected_year)]

# Obtener los últimos dos años del rango seleccionado
last_two_years = df_selected_years.tail(2).index.values

# Gráfico de línea con resalte de los años seleccionados
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=df_tasa_mortalidad, x=df_tasa_mortalidad.index, y='tasa_mortalidad', ci=None, color='gray', ax=ax)
sns.lineplot(data=df_selected_years, x=df_selected_years.index, y='tasa_mortalidad', ci=None, color='blue', ax=ax)
ax.set_title('Tasa de Mortalidad por Año')
ax.set_xlabel('Año')
ax.set_ylabel('Tasa de Mortalidad (%)')
ax.legend(['Todos los años', f'Años seleccionados'])
plt.xticks(rotation=45)

# Resaltar los últimos dos años en el rango seleccionado
for year in last_two_years:
    ax.axvline(x=year, color='red', linestyle='--')

plt.tight_layout()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)


########################################################################################


st.title("KPI-2")
st.title("Aumentar en un 5% el número de vidas salvadas a nivel anual")

# Normalizando
df['fecha'] = pd.to_datetime(df['fecha'])
df['all_aboard'] = df['all_aboard'].replace('?', np.nan).astype(float).astype('Int64')
df['cantidad de fallecidos'] = df['cantidad de fallecidos'].replace('?', np.nan).astype(float).astype('Int64')

# Limpiar los valores incorrectos o faltantes en las columnas relevantes
df['cantidad de fallecidos'] = pd.to_numeric(df['cantidad de fallecidos'], errors='coerce')
df['all_aboard'] = pd.to_numeric(df['all_aboard'], errors='coerce')

# Crear una nueva columna "Año" a partir de la columna de fechas
df['Año'] = df['fecha'].dt.year

# Calcular el número de vidas salvadas anualmente
vidas_salvadas_anual = df.groupby('Año')[['cantidad de fallecidos', 'all_aboard']].sum()
vidas_salvadas_anual['Vidas Salvadas'] = vidas_salvadas_anual['all_aboard'].diff().fillna(0)

# Filtrar los últimos 10 años
ultimo_ano = vidas_salvadas_anual.index.max()
ultimos_10_anos = vidas_salvadas_anual.loc[ultimo_ano-9:ultimo_ano]

# Definir una lista de colores para las barras
colores = ['green'] * (len(ultimos_10_anos)-2) + ['blue', 'blue']

# Filtro interactivo por año
selected_year = st.slider('Seleccione un año', min_value=int(ultimos_10_anos.index.min()), max_value=int(ultimos_10_anos.index.max()), value=int(ultimos_10_anos.index.max()), step=1)

# Filtrar los datos para el año seleccionado
df_filtered = ultimos_10_anos[ultimos_10_anos.index <= selected_year]

# Mostrar el gráfico utilizando Matplotlib y Streamlit
fig, ax = plt.subplots(figsize=(10, 6))  # Ajustar el tamaño de la figura

# Configurar las barras para los años dentro del rango seleccionado
contador_colores = 0
for idx, row in df_filtered.iterrows():
    if idx >= ultimo_ano - 1:
        ax.bar(idx, row['Vidas Salvadas'], color='orange')
    else:
        ax.bar(idx, row['Vidas Salvadas'], color=colores[contador_colores])
        contador_colores += 1

# Configurar la línea que une los dos últimos años
ultimo_ano_1 = df_filtered.index[-2]
ultimo_ano_2 = df_filtered.index[-1]
ax.plot([ultimo_ano_1, ultimo_ano_2], [df_filtered.loc[ultimo_ano_1, 'Vidas Salvadas'], df_filtered.loc[ultimo_ano_2, 'Vidas Salvadas']], color='red', linestyle='dashed')

ax.set_xticks(df_filtered.index)
ax.set_xticklabels(df_filtered.index)  # Mostrar los años en el eje x

ax.set_title('Número de Vidas Salvadas Anual (Últimos 10 años)')
ax.set_xlabel('Año')
ax.set_ylabel('Número de vidas salvadas')

plt.tight_layout()
st.pyplot(fig)


########################################################################################

st.title("KPI-3")
st.title("Top 10 Operadores con mayor tasa de mortalidad promedio")

# Calcular la frecuencia de cada operador
frecuencia_operador = df['OperadOR'].value_counts().reset_index()
frecuencia_operador.columns = ['OperadOR', 'Frecuencia']

# Obtener el top 10 de operadores con mayor frecuencia
top_operadores_frecuencia = frecuencia_operador.head(10)

# Filtrar los datos por los operadores con mayor frecuencia
data_top_operadores = df[df['OperadOR'].isin(top_operadores_frecuencia['OperadOR'])]

# Calcular la tasa de mortalidad promedio por operador
mortalidad_por_operador = data_top_operadores.groupby('OperadOR')[['cantidad de fallecidos', 'all_aboard']].mean()
mortalidad_por_operador['Tasa de Mortalidad'] = (mortalidad_por_operador['cantidad de fallecidos'] / mortalidad_por_operador['all_aboard']) * 100

# Reemplazar valores infinitos por NaN
mortalidad_por_operador.replace([np.inf, -np.inf], np.nan, inplace=True)

# Combinar la frecuencia con los datos de mortalidad por operador
mortalidad_por_operador = mortalidad_por_operador.merge(top_operadores_frecuencia, on='OperadOR', how='left')

# Eliminar filas con valores NaN en la columna 'Tasa de Mortalidad'
mortalidad_por_operador = mortalidad_por_operador.dropna(subset=['Tasa de Mortalidad'])

# Reiniciar los índices
mortalidad_por_operador = mortalidad_por_operador.reset_index()

# Definir una lista de colores para las barras
colores = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'brown', 'gray', 'teal']

# Ordenar el DataFrame por la columna 'Tasa de Mortalidad' de manera descendente
mortalidad_por_operador.sort_values('Tasa de Mortalidad', ascending=False, inplace=True)

# Generar el gráfico de barras horizontales con colores personalizados
plt.figure(figsize=(10, 6))  # Ajustar el tamaño de la figura

# Obtener los nombres de los operadores y la tasa de mortalidad correspondiente
nombres_operadores = mortalidad_por_operador['OperadOR']
tasa_mortalidad = mortalidad_por_operador['Tasa de Mortalidad']

# Obtener el índice para el rango de barras
rango_barras = np.arange(len(nombres_operadores))

# Mostrar el gráfico utilizando Matplotlib y Streamlit
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(rango_barras, tasa_mortalidad, color=colores)
ax.set_yticks(rango_barras)
ax.set_yticklabels(nombres_operadores)

ax.set_title('Top 10 Operadores con mayor tasa de mortalidad promedio')
ax.set_xlabel('Tasa de mortalidad (%)')
ax.set_ylabel('Operador aéreo')

plt.tight_layout()
st.pyplot(fig)

#######################################################################################

st.title("KPI-4")
st.title("Top 10 Operadores con mayor tasa de mortalidad por ruta")

import plotly.express as px


# Calcular la frecuencia de cada ruta
frecuencia_ruta = df['Ruta'].value_counts().reset_index()
frecuencia_ruta.columns = ['Ruta', 'Frecuencia']

# Obtener las 10 rutas con mayor frecuencia
top_rutas_frecuencia = frecuencia_ruta.head(10)

# Filtrar los datos por las rutas con mayor frecuencia
data_top_rutas = df[df['Ruta'].isin(top_rutas_frecuencia['Ruta'])]

# Calcular la tasa de mortalidad promedio por ruta
mortalidad_por_ruta = data_top_rutas.groupby('Ruta')[['cantidad de fallecidos', 'all_aboard']].mean()
mortalidad_por_ruta['Tasa de Mortalidad'] = (mortalidad_por_ruta['cantidad de fallecidos'] / mortalidad_por_ruta['all_aboard']) * 100

# Reemplazar valores infinitos por NaN
mortalidad_por_ruta.replace([np.inf, -np.inf], np.nan, inplace=True)

# Combinar la frecuencia con los datos de mortalidad por ruta
mortalidad_por_ruta = mortalidad_por_ruta.merge(top_rutas_frecuencia, on='Ruta', how='left')

# Eliminar filas con valores NaN en la columna 'Tasa de Mortalidad'
mortalidad_por_ruta = mortalidad_por_ruta.dropna(subset=['Tasa de Mortalidad'])

# Ordenar el DataFrame por la columna 'Tasa de Mortalidad' de manera descendente
mortalidad_por_ruta.sort_values('Tasa de Mortalidad', ascending=False, inplace=True)

# Crear el gráfico de barras horizontales con colores personalizados
fig, ax = plt.subplots(figsize=(10, 6))  # Ajustar el tamaño de la figura

# Obtener los nombres de las rutas y la tasa de mortalidad correspondiente
nombres_rutas = mortalidad_por_ruta['Ruta']
tasa_mortalidad = mortalidad_por_ruta['Tasa de Mortalidad']

# Obtener el índice para el rango de barras
rango_barras = np.arange(len(nombres_rutas))

ax.barh(rango_barras, tasa_mortalidad, color='red')
ax.set_yticks(rango_barras)
ax.set_yticklabels(nombres_rutas)
ax.set_title('Top 10 Rutas con mayor tasa de mortalidad promedio')
ax.set_xlabel('Tasa de mortalidad (%)')
ax.set_ylabel('Ruta')

plt.tight_layout()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

# Calcular la tasa de mortalidad promedio por ruta utilizando los datos filtrados
mortalidad_por_ruta = data_top_rutas.groupby('Ruta')[['cantidad de fallecidos', 'all_aboard']].mean()
mortalidad_por_ruta['Tasa de Mortalidad'] = (mortalidad_por_ruta['cantidad de fallecidos'] / mortalidad_por_ruta['all_aboard']) * 100

# Reemplazar valores infinitos por NaN
mortalidad_por_ruta.replace([np.inf, -np.inf], np.nan, inplace=True)

# Combinar la frecuencia con los datos de mortalidad por ruta
mortalidad_por_ruta = mortalidad_por_ruta.merge(top_rutas_frecuencia, on='Ruta', how='left')

# Eliminar filas con valores NaN en la columna 'Tasa de Mortalidad'
mortalidad_por_ruta = mortalidad_por_ruta.dropna(subset=['Tasa de Mortalidad'])

# Ordenar el DataFrame por la columna 'Tasa de Mortalidad' de manera descendente
mortalidad_por_ruta.sort_values('Tasa de Mortalidad', ascending=False, inplace=True)

# Crear un gráfico de mapa interactivo con Plotly
fig = px.choropleth(mortalidad_por_ruta,
                    locations='Ruta',
                    locationmode='country names',
                    color='Tasa de Mortalidad',
                    color_continuous_scale='Reds',
                    labels={'Tasa de Mortalidad': 'Tasa de mortalidad (%)'},
                    title='Tasa de mortalidad promedio por ruta (Top 10)')

# Configurar el diseño del gráfico
fig.update_layout(geo=dict(showframe=False,
                           showcoastlines=False,
                           projection_type='equirectangular'))

# Mostrar el gráfico interactivo en Streamlit
st.plotly_chart(fig)
