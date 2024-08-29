import warnings
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

def tratamiento(fichero):
    dataframe = pd.read_csv(fichero)
    dataframe = dataframe.rename({'clave1':'fecha', 'uniTotal':'unidades'}, axis=1)
    dataframe['fecha'] = pd.to_datetime(dataframe['fecha'], format="%Y%m%d")
    dataframe = dataframe.set_index(['fecha'])
    dataframe = dataframe.resample('D').first()
    dataframe.fillna(value=0, inplace=True)
    dataframe = dataframe.groupby(pd.Grouper(freq='B'))
    dataframe = dataframe.sum()
    dataframe[dataframe['unidades'] < 0] = 0
    dataframe = dataframe["2012-01-01":"2024-06-30"]
    name = fichero[fichero.find('/')+1:fichero.find('.')]
    return name, dataframe

def agrupar(ficheros: list):
    df = pd.DataFrame()
    lista = []
    for archivo in ficheros:
        name, aux = tratamiento(archivo)
        aux = aux.rename({'unidades':name}, axis=1)
        df = pd.concat([df, aux], axis=1)
        lista.append(name)
    return df, lista

def grafico(dataframe):
    fig, ax = plt.subplots()
    ax.plot(dataframe)
    ax.minorticks_on()
    ax.grid(which='major', alpha = 0.65, linestyle='-')
    ax.grid(which='minor', alpha = 0.25, linestyle='--')
    fig.autofmt_xdate()

df, names = agrupar(['data/filtros.csv', 'data/baterias.csv', 'data/aceites.csv', 'data/limpiaparabrisas.csv'])

# Meteorologia
df2 = pd.read_csv('data/meteo_olvera.csv', decimal=',')
df2['fecha'] = pd.to_datetime(df2['fecha'], format="%Y-%m-%d")
df2.set_index('fecha', inplace=True)
df2 = df2[['tmed', 'prec', 'hrMedia']]
df2 = df2.resample('B').first()
df2 = df2.fillna(method='backfill')
df2 = df2["2012-01-01":"2023-12-31"]
df = pd.concat([df, df2], axis=1)

# Precio combustible
df2 = pd.read_csv('data/carburante.csv', decimal=',')
df2['fecha'] = pd.to_datetime(df2['fecha']+'0', format="%Y-%W%w")
df2 = df2.set_index('fecha').sort_index()
df2 = df2.resample('B').first().ffill()
df2 = df2["2012-01-01":"2023-12-31"]

df = pd.concat([df, df2], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))




# Figuras de error cuadrático medio
for name in names:
    dfe = pd.read_csv(f'results/errores_{name}.csv',index_col=0)
    dfe = dfe.round(2)

    # Valores escalados
    dfe = dfe[1:2]
    # # Valores absolutos
    # dfe = dfe[2:]

    fig, ax = plt.subplots()
    x = np.arange(len(dfe.columns))
    width = 0.4

    for tipo, medida in dfe.iterrows():          
        offset = width
        rects = ax.bar(x= x + offset, height=medida, width=width, label=tipo)
        ax.bar_label(rects, padding=len(dfe))

    ax.set_ylabel('Error (unidades)')
    # ax.set_xlabel('Modelo')
    ax.set_title(f'Raíz del error cuadrático medio (RMSE) de {name}')
    ax.set_xticks(x + width, dfe.columns)
    ax.legend()
    ax.axis(ymin=0, ymax=max(dfe.max())*1.07)
    ax.minorticks_on()
    ax.grid(which='major', alpha = 0.65, linestyle='-')
    ax.grid(which='minor', alpha = 0.25, linestyle='--')
    fig.autofmt_xdate()
    fig.savefig(f'figs/rmse_{name}.pdf')


# Figuras predicción mes
for name in names:
    scaler.fit(df[[name]])
    
    dfp = pd.read_csv(f'results/resultados_{name}.csv')
    dfp['fecha'] = pd.to_datetime(dfp['fecha'], format="%Y-%m-%d")
    dfp = dfp.set_index(['fecha'])
    dfp[dfp.columns] = scaler.inverse_transform(dfp)
    dfp_mes = dfp.groupby(pd.Grouper(freq='MS')).sum()
    
    fig, ax = plt.subplots()
    lines = ax.plot(dfp_mes, label=dfp.columns)
    lines[0].set_color('k')
    lines[0].set_label('Ventas reales')
    lines[0].set_linewidth(3)
    lines[0].zorder = 3
    ax.minorticks_on()
    ax.grid(which='major', alpha = 0.65, linestyle='-')
    ax.grid(which='minor', alpha = 0.25, linestyle='--')
    ax.axis([datetime(2024,1,1), datetime(2024,6,1), min(dfp_mes.min())-5, max(dfp_mes.max())+5])
    ax.legend(loc='upper left')
    ax.set_ylabel('Unidades')
    ax.set_title(f'{name}')
    plt.savefig(f'figs/prediccion_mes_{name}.pdf')