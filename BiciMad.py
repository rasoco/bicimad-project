from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import folium.plugins
import plotly.express as px
from folium.plugins import MarkerCluster
from bokeh.plotting import figure, output_file, show
from streamlit_folium import folium_static
import altair as alt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split



def main():
    ### Leyendo datos y limpiando datos
    df = pd.read_csv("https://raw.githubusercontent.com/rasoco/master_data_science/master/TFM/data/originalbikestations.csv",sep=';', encoding='utf-8')
    df['Year'] = pd.DatetimeIndex(df['Fecha de Alta']).year
    year = df['Year']
    df = df.rename(columns = {'Número':'id','Número de Plazas':'Anclajes'}) #change de name of column Número
    df['ids'] = df['id'].str.rstrip('ampliacionb') #to get only number values
    # df_stations = df_stations[onlycolumns]
    df['ids'] = df['ids'].astype(int).sort_values() #change values ids and order
    d_b = df.groupby(['Distrito'])['Barrio'].count()
    locations = df[['LATITUD', 'LONGITUD']] # (Y, X)
    locationlist = locations.values.tolist()
    df['Locationlist'] = locationlist

    st.sidebar.title("Menu")
    st.sidebar.text_input('Link to Project', "https://github.com/rasoco/master_data_science/tree/master/TFM") 
    app_mode = st.sidebar.selectbox("Please select a page", ["Homepage", "BiciMadStations", "Do you use BiciMad?"])
                                                            

    if app_mode == "Homepage":
        st.title(" BiciMad Project")
        st.write("Este proyecto tiene por finalidad analizar el sistema de bicicletas públicas en la ciudad de Madrid")
        # img_file_buffer = st.file_uploader("project.jpeg")
        # image= Image.open("img_file_buffer")
        st.image("https://drive.google.com/uc?export=view&id=1oKQ8WH_BlXZddEoelfAfaj62-JUSac4G", use_column_width=True)
   
    elif app_mode == "BiciMadStations":
          st.title("Exploración de estacionamientos")
          st.subheader("Puntos de estacionamientos por Distrito")
          map1 = folium.Map(location=[40.417110795315295, -3.70199802576925], zoom_start=12)
          for point in range(0, len(locationlist)):
              folium.Marker(locationlist[point], tooltip='Estacionamiento:'+df['id'][point], icon=folium.Icon(color='darkblue', icon_color='white', icon='bicycle', angle=0, prefix='fa')).add_to(map1)
          folium_static(map1)
          st.subheader("Cluster de los estacionamientos")
          folium.plugins.MarkerCluster()
          map2 = folium.Map(location=[40.417110795315295, -3.70199802576925], tiles='Stamen Terrain', zoom_start=11)
          marker_cluster = folium.plugins.MarkerCluster().add_to(map2)
          for point in range(0, len(locationlist)):
              folium.Marker(locationlist[point], tooltip=df['DIRECCION'].astype(str)[point]+',Total Anclajes: '+(df['Anclajes'].astype(str)[point]), icon=folium.Icon(color='darkblue', icon_color='green', icon='bicycle', angle=0, prefix='fa')).add_to(marker_cluster)
          folium_static(map2)     
# Total de Anclajes por Año
          st.subheader("Evolución por año de anclajes por Distrito")
          year_dist = alt.Chart(df).mark_bar().encode(
              x='Year:O',
              y='sum(Anclajes)',
              color='Distrito').properties(title='Total de Anclajes por Año', height=700, width=900).interactive()
          year_dist       

  
# Total de estacionamientos por distrito
          bike_dist = alt.Chart(df).mark_bar().encode(
              x=alt.X('Distrito'),
              y= 'count()').properties(title="Distribución del total de estaciones por Distrito", width=400).interactive()   
# Total de Anclajes por Distrito
          bike_dist2 = alt.Chart(df).mark_bar().encode(
              x=alt.X('Distrito'),
              y= 'sum(Anclajes)').properties(
              title='Distribución del total de anclajes por Distrito', width=400).interactive()
          (bike_dist|bike_dist2)

    elif app_mode == "Do you use BiciMad?":
          st.title("¿Cuánto utilizas BiciMad?")
          #Creando variables
          distance_user = st.sidebar.number_input('¿Que distancia en km recorres por trayecto en BiciMad?',min_value=0.0, max_value=20.0, step=0.01)
          travel_minutes_user = st.sidebar.number_input('¿Cuántos minutos recorres por trayecto en BiciMad?',min_value=0.0, max_value=2000.0, step=0.01)
          agerange_dict = {'Entre 0 y 16 años':1, 'Entre 17 y 18 años':2, 'Entre 19 y 26 años':3, 'Entre 27 y 40 años':4, 'Entre 41 y 65 años':5, 'Tengo 66 años o más':6}
          ageRange_user = st.sidebar.selectbox('¿En qué rango de edad te encuentras? ', ('Entre 0 y 16 años','Entre 17 y 18 años','Entre 19 y 26 años',
                                                                                         'Entre 27 y 40 años','Entre 41 y 65 años','Tengo 66 años o más'))
  
          count_travel_user = st.sidebar.number_input('¿Cuántos trayectos realizas por semana?',0,1000,0)
          ### Modelo 
          data=pd.read_csv("https://raw.githubusercontent.com/rasoco/master_data_science/master/TFM/data/balanced_data.csv",',', None)
          y8 = data['user_type'] 
          x8 = data.loc[:,['distance','travel_minutes', 'ageRange','count_travel']] 
          class_names = y8.sort_values().unique()
          model8 = KNeighborsClassifier(n_neighbors=3)
          df_user = pd.DataFrame({'distance':[distance_user],
                        'travel_minutes':[travel_minutes_user],
                        'ageRange':[agerange_dict[ageRange_user]],
                        'count_travel':[count_travel_user]})
          x8_train, x8_test, y8_train, y8_test =train_test_split(x8, y8, test_size=0.30, random_state=0)
          model8.fit(x8_train,y8_train)
          pred8 = model8.predict(df_user[['distance','travel_minutes','ageRange','count_travel']])

          st.header(result(pred8))

@st.cache
def result(pred8):
    if pred8 == 1:
      return 'Utilizas con frecuencia las bicis eléctricas de Madrid'
    elif pred8 == 2:
      return 'Utilizas de manera ocasional las bicis eléctricas de Madrid'
    else:
      return 'Probablemente seas trabajador/a de BiciMad'


main()