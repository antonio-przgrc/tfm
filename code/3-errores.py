import warnings
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

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
df2 = df2['2012':]
df = pd.concat([df, df2], axis=1)

# Precio combustible
df2 = pd.read_csv('data/carburante.csv', decimal=',')
df2['fecha'] = pd.to_datetime(df2['fecha']+'0', format="%Y-%W%w")
df2 = df2.set_index('fecha').sort_index()
df2 = df2.resample('B').first().ffill()
df2 = df2['2012':]

df = pd.concat([df, df2], axis=1)

df = df.reset_index()

scaler = MinMaxScaler(feature_range=(0, 1))

# Definicion TimeSeries
cols = ['tmed','prec', 'hrMedia', 'gasolina', 'diesel']

series = TimeSeries.from_dataframe(df, time_col='fecha', value_cols=names+cols)

series = series.add_holidays(country_code='ES', prov='AN')

transformer = Scaler(scaler)
series = transformer.fit_transform(series)

train, test = series.split_after(pd.Timestamp(year=2023, month=12, day=31))



for name in names:
    errores = pd.DataFrame()
    scaler.fit(df[[name]])

    df_res = pd.read_csv(f'results/resultados_{name}.csv')

    df_res['fecha'] = pd.to_datetime(df_res['fecha'])
    df_res = df_res.set_index('fecha')
    df_res = df_res[:130]

    df_res_m = df_res.groupby(pd.Grouper(freq='MS')).sum()[:datetime(2024,1,31)]
    df_res_q = df_res.groupby(pd.Grouper(freq='QS')).sum()[:datetime(2024,3,31)]
    df_res_y = df_res.groupby(pd.Grouper(freq='YS')).sum()[:datetime(2024,6,30)]

    idx_names = ['Error diario', 'Error mensual', 'Error cuatrimestre','Error semestre']
    for model in df_res.columns:
        if model == name:
            continue
        error_d = rmse(df_res[name], df_res[model])
        error_m = rmse(df_res_m[name], df_res_m[model])
        error_q = rmse(df_res_q[name], df_res_q[model])
        error_y = rmse(df_res_y[name], df_res_y[model])
        errores = pd.concat([errores, pd.DataFrame({model:[error_d, error_m, error_q, error_y]}, index=idx_names)],axis=1)
    errores2 = pd.DataFrame()
    errores2[errores.columns] = scaler.inverse_transform(errores)
    errores2.index = ['Error diario (abs)', 'Error mensual (abs)', 'Error cuatrimestre (abs)','Error semestre (abs)']

    errores = pd.concat([errores, errores2])

    errores.to_csv(f'results/errores_{name}.csv')
