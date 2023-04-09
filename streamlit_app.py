###############################################################################################################
# App de explicacion como funciona Algoritmo kmeans
###############################################################################################################

#**************************************************************************************************************
# [A] Importar LIbrerias a Utilizar
#**************************************************************************************************************

import pandas as pd
import numpy as np
import pickle

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st


#**************************************************************************************************************
# [B] Importar Objetos utiles
#**************************************************************************************************************



df_stats = pickle.load(open('df_stats.pkl','rb'))

Modelo_RL = pickle.load(open('Modelo_RL.pkl','rb'))
Modelo_RF = pickle.load(open('Modelo_RF.pkl','rb'))
# Modelo_XGB = pickle.load(open('Modelo_XGB.pkl','rb'))

robust_scaler = pickle.load(open('robust_scaler.pkl','rb'))

df_glosa = pd.read_excel('DICCIONARIO.xlsx', sheet_name='Hoja1') 
dic_glosa = df_glosa.set_index('Campo').T.to_dict('records')[0]

# creacion de otros objetos utiles
dic_glosa2 = dict(map(reversed, dic_glosa.items()))

listado_vars = [
  'Compras_Mto_Uso',
  'Compras_Mto_cuotas_P',
  'Compras_Mto_Retail',
  'Compras_Mto_onthem',
  'Compras_Mto',
  'Compras_Nro',
  'Compras_Mto_onus',
  'Log_N',
  'Edad',
  'Compras_N_Rubros',
  'Compras_Mto_paris',
  'Compras_Mto_cuotas',
  'Cupo'
]

listado_glosas = [dic_glosa[x] for x in listado_vars]

Modelos = {
  'Regresion Logistica': Modelo_RL,
  'Random Forest': Modelo_RF # 'XGBoost': Modelo_XGB  
}


#**************************************************************************************************************
# [C] Crear funciones utiles para posterior uso
#**************************************************************************************************************

# crear funcion de calcular pbb segun 2 variables
@st.cache(suppress_st_warning=True,allow_output_mutation=True) # https://docs.streamlit.io/library/advanced-features/caching
def calculo_pbb(v1,n1,v2,n2,Modelo,Modelo_usar,escalador,dic_valores1):
  valor_vars_aux = dic_valores1.copy()
  valor_vars_aux[v1]=n1
  valor_vars_aux[v2]=n2
  
  df_aux = pd.DataFrame.from_dict(valor_vars_aux,orient='index').T
  
  Proba = np.where(
    Modelo_usar=='Regresion Logistica',
    Modelo.predict_proba(escalador.transform(df_aux))[:,1].item(),
    Modelo.predict_proba(df_aux.values)[:,1].item()
    ).item()

  return Proba


#**************************************************************************************************************
# [Y] Comenzar a diseñar App
#**************************************************************************************************************

def main():

  # Use the full page instead of a narrow central column
  st.set_page_config(layout="wide")
  
  #=============================================================================================================
  # [01] Elementos del sidebar: textos 
  #=============================================================================================================   
  
  # autoria 
  # st.sidebar.markdown('**Autor: [Sebastian Barrera](https://www.linkedin.com/in/sebasti%C3%A1n-nicolas-barrera-varas-70699a28)**')
  # st.sidebar.markdown('Version SB_V20230128')


  Modelo_usar = st.sidebar.selectbox(
    label = 'Seleccionar Modelo',
    options = list(Modelos.keys())
    )
  
  # definir modelo a usar segun seleccion 
  Modelo = Modelos[Modelo_usar]

  #______________________________________________________________________________
  # agregar variables 
  
  # generar columnas 
  col_a = [None] * 13
  col_b = [None] * 13
  variable = [None] * 13
  
  # iterar agregando columnas 
  for i in range(13):
    
    col_a[i],col_b[i] =st.sidebar.columns((3,6))
        
    variable[i] = col_a[i].number_input(
      label = dic_glosa[listado_vars[i]],
      value = round(df_stats.loc[df_stats['variable']==listado_vars[i],'Prom'].item(),3)   
    )
    
    # para saber cuantos decimales asignar 
    v = df_stats.loc[df_stats['variable']==listado_vars[i],'Prom'].item()
    d = np.where(v<1,2,0).item()
  
    col_b[i].write(
      df_stats.loc[
        df_stats['variable']==listado_vars[i],
        ['P25','P50','Prom','P75']
        ].style.hide_index().format(decimal=',', precision=d).to_html(),
      unsafe_allow_html=True
      )
  
  
  
  #=============================================================================================================
  # [02] Elementos del main
  #=============================================================================================================   
  
  # titulo inicial 
  st.markdown('# Sensibilidad Modelo segun variables')
  
  #-------------------------------------------------------------------------------------------------------------
  # [02.1] Elementos del main: Grafico1 de lineas con opcion de segunda variable
  #-------------------------------------------------------------------------------------------------------------

  #_______________________________________________________________________
  # Titulo + variables 

  # subtitulo
  st.markdown('### A. Grafico de linea: Probabilidad vs Variable')
  
  # colocar widgets
  col1_a,col1_b = st.columns((1,1))
  
  var1a = col1_a.selectbox(
    label = 'Seleccionar Variable eje x',
    options = [dic_glosa[x] for x in listado_vars],
    key=1
    )
  
  var2a = col1_b.selectbox(
    label = 'Seleccionar Variable color',
    options = ['Ninguna']+[dic_glosa[x] for x in listado_vars],
    key=2
    )


  #_______________________________________________________________________
  # Definir funcion 


  # @st.cache(suppress_st_warning=True,allow_output_mutation=True) # https://docs.streamlit.io/library/advanced-features/caching
  def grafico1(variable,var1,var2,Modelo_usar):
           
    # definir modelo
    Modelo = Modelos[Modelo_usar]

    # calcular nombre de variables en base
    var1_df = dic_glosa2[var1]

    # calcular valores minimos y maximos y luego lista 
    valor1_min = round(df_stats.loc[df_stats['variable']==var1_df,'P10'].item(),3)
    valor1_max = round(df_stats.loc[df_stats['variable']==var1_df,'P90'].item(),3)
    valores1 = np.linspace(valor1_min, valor1_max, 50)

    # crear df a partir de variables ingresadas
    df1 = pd.DataFrame(variable).T
    df1.columns = listado_vars   
    df2 = df1.loc[df1.index.repeat(len(valores1))].reset_index(drop=True)

    # modificar variable1
    for i in range(0,len(df2)):
      df2.loc[df2.index[i],var1_df] = valores1[i]


    # generar entregable dependiendo de si se ingreso o no valor 2
    if(var2=='Ninguna'):
      
      df2['Pbb'] = np.where(
        Modelo_usar=='Regresion Logistica',
        Modelo.predict_proba(robust_scaler.transform(df2))[:,1],
        Modelo.predict_proba(df2)[:,1]
        )
      
      # generar grafico 
      fig1 = px.line(
        df2,
        x = var1_df,
        y = 'Pbb'
        )
      fig1.update_layout(
        title='Probabilidad vs '+var1,
        xaxis_title=var1,
        yaxis_title='Probabilidad'
        )
      
    else:
      
      var2_df = dic_glosa2[var2]
      valor2_min = round(df_stats.loc[df_stats['variable']==var2_df,'P10'].item(),3)
      valor2_max = round(df_stats.loc[df_stats['variable']==var2_df,'P90'].item(),3)
      valores2 = np.linspace(valor2_min, valor2_max, 5)

      # modificar variable2
      df3 = pd.DataFrame()
      for i in range(0,len(valores2)):
        
        df2[var2_df] = valores2[i]
        df3 = pd.concat([df3,df2])

      # calcular Pbb y arrojar grafico 
      df3['Pbb'] = np.where(
        Modelo_usar=='Regresion Logistica',
        Modelo.predict_proba(robust_scaler.transform(df3))[:,1],
        Modelo.predict_proba(df3)[:,1]
      )


      # generar grafico 
      fig1 = px.line(
        df3,
        x = var1_df,
        y = 'Pbb',
        color = var2_df
      )
      fig1.update_layout(
        title='Probabilidad vs '+var1,
        xaxis_title=var1,
        yaxis_title='Probabilidad'
      )
      
    return fig1



  #_______________________________________________________________________
  # Mostrar resultado 

  fig1 = grafico1(variable,var1a,var2a,Modelo_usar)
  st.plotly_chart(fig1, use_container_width=True)


  #-------------------------------------------------------------------------------------------------------------
  # [02.2] Elementos del main: Grafico2 de densidad
  #-------------------------------------------------------------------------------------------------------------

  #_______________________________________________________________________
  # Titulo + variables 

  # subtitulo
  st.markdown('### B. Grafico de densidad: Variable1 + Variable2')
  
  # colocar widgets
  col2_a,col2_b = st.columns((1,1))
  
  var1b = col2_a.selectbox(
    label = 'Seleccionar Variable eje x',
    options = [dic_glosa[x] for x in listado_vars],
    key=3
    )
  
  var2b = col2_b.selectbox(
    label = 'Seleccionar Variable eje y',
    options = [dic_glosa[x] for x in listado_vars],
    key=4
    )


  #_______________________________________________________________________
  # Definir funcion 


  # @st.cache(suppress_st_warning=True,allow_output_mutation=True) # https://docs.streamlit.io/library/advanced-features/caching
  def grafico2(variable,var1,var2,Modelo_usar):
    
    # crear diccionario de variables que se van a utilizar
    valor_vars = dict(zip(listado_vars, variable))
    
    # definir modelo
    Modelo = Modelos[Modelo_usar]

    # calcular nombre de variables en base
    var1_df = dic_glosa2[var1]
    var2_df = dic_glosa2[var2]

    # calcular valores minimos y maximos y luego lista -> para var1 
    valor1_min = round(df_stats.loc[df_stats['variable']==var1_df,'P10'].item(),3)
    valor1_max = round(df_stats.loc[df_stats['variable']==var1_df,'P90'].item(),3)
    valores1 = np.linspace(valor1_min, valor1_max, 15)
    
    # calcular valores minimos y maximos y luego lista -> para var2 
    valor2_min = round(df_stats.loc[df_stats['variable']==var2_df,'P10'].item(),3)
    valor2_max = round(df_stats.loc[df_stats['variable']==var2_df,'P90'].item(),3)
    valores2 = np.linspace(valor2_min, valor2_max, 15)

    # calcular probabilidades 
    PBBs = [[calculo_pbb(
      v1 = var1_df,
      n1 = x,
      v2 = var2_df,
      n2 = y,
      Modelo = Modelo,
      Modelo_usar = Modelo_usar,
      escalador = robust_scaler,
      dic_valores1 = valor_vars
      ) for x in valores1] for y in valores2]

    # crear grafico 
    fig2 = go.Figure(data=go.Contour(
      z=PBBs,
      x=valores1,
      y=valores2,
      colorscale='RdYlGn' # px.colors.diverging.swatches_continuous() (https://plotly.com/python/builtin-colorscales/)
        ))
    fig2.update_layout(
      title=var1+' vs '+var2,
      xaxis_title=var1,
      yaxis_title=var2
    )
      
    return fig2



  #_______________________________________________________________________
  # Mostrar resultado 

  fig2 = grafico2(variable,var1b,var2b,Modelo_usar)
  st.plotly_chart(fig2, use_container_width=True)



  #-------------------------------------------------------------------------------------------------------------
  # [02.3] Elementos del main: Grafico3 3D
  #-------------------------------------------------------------------------------------------------------------

  #_______________________________________________________________________
  # Titulo + variables 

  # subtitulo
  st.markdown('### C. Grafico 3D: Variable1 + Variable2 vs Probabilidad')
  
  # colocar widgets
  col3_a,col3_b = st.columns((1,1))
  
  var1c = col3_a.selectbox(
    label = 'Seleccionar Variable eje x',
    options = [dic_glosa[x] for x in listado_vars],
    key=5
    )
  
  var2c = col3_b.selectbox(
    label = 'Seleccionar Variable eje y',
    options = [dic_glosa[x] for x in listado_vars],
    key=6
    )


  #_______________________________________________________________________
  # Definir funcion 


  # @st.cache(suppress_st_warning=True,allow_output_mutation=True) # https://docs.streamlit.io/library/advanced-features/caching
  def grafico3(variable,var1,var2,Modelo_usar):
    
    # crear diccionario de variables que se van a utilizar
    valor_vars = dict(zip(listado_vars, variable))
    
    # definir modelo
    Modelo = Modelos[Modelo_usar]

    # calcular nombre de variables en base
    var1_df = dic_glosa2[var1]
    var2_df = dic_glosa2[var2]

    # calcular valores minimos y maximos y luego lista -> para var1 
    valor1_min = round(df_stats.loc[df_stats['variable']==var1_df,'P10'].item(),3)
    valor1_max = round(df_stats.loc[df_stats['variable']==var1_df,'P90'].item(),3)
    valores1 = np.linspace(valor1_min, valor1_max, 15)
    
    # calcular valores minimos y maximos y luego lista -> para var2 
    valor2_min = round(df_stats.loc[df_stats['variable']==var2_df,'P10'].item(),3)
    valor2_max = round(df_stats.loc[df_stats['variable']==var2_df,'P90'].item(),3)
    valores2 = np.linspace(valor2_min, valor2_max, 15)

    # calcular probabilidades 
    PBBs = [[calculo_pbb(
      v1 = var1_df,
      n1 = x,
      v2 = var2_df,
      n2 = y,
      Modelo = Modelo,
      Modelo_usar = Modelo_usar,
      escalador = robust_scaler,
      dic_valores1 = valor_vars
      ) for x in valores1] for y in valores2]

    # crear grafico 
    fig3 = go.Figure(data=go.Surface(
      z=PBBs,
      x=valores1,
      y=valores2,
      colorscale='RdYlGn' # px.colors.diverging.swatches_continuous() (https://plotly.com/python/builtin-colorscales/)
        ))
    fig3.update_traces(contours_z=dict(
      show=True, 
      usecolormap=True,
      highlightcolor='limegreen', 
      project_z=True
      ))
    fig3.update_layout(
      width=800, 
      height=700,                       
      title=var1+' vs '+var2,
      scene = dict(
        xaxis_title=var1,
        yaxis_title=var2,
        zaxis_title='Probabilidad'
        ),
      )
      
    return fig3



  #_______________________________________________________________________
  # Mostrar resultado 

  fig3 = grafico3(variable,var1c,var2c,Modelo_usar)
  st.plotly_chart(fig3, use_container_width=True)

#**************************************************************************************************************
# [Z] Lanzar App
#**************************************************************************************************************

    
# arrojar main para lanzar App
if __name__=='__main__':
    main()
    
# Escribir en terminal: streamlit run App_Capstone_v2.py
# !streamlit run App_Capstone_v2.py

# para obtener todos los requerimientos de librerias que se usan
# !pip freeze > requirements.txt


# para obtener el archivo "requirements.txt" de los requerimientos puntuales de los .py
# !pipreqs "/Seba/Universidad/MAGISTER Data Science UAI (27-01-2021)/2. Cursos/34. CAPSTONE PPROJECT 0 V(22-04-22)/Ppt Nro2 Capstone/App Streamlit ppt3/"


