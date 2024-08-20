from datetime import datetime, timedelta

import pandas as pd

from copy import copy as cp
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error as rootmse

import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.utils.callbacks import TFMProgressBar
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, TransformerModel, Prophet, BlockRNNModel, NBEATSModel
from darts.metrics import rmse
from darts.utils.statistics import check_seasonality, plot_acf
import darts.utils.timeseries_generation as tg
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.missing_values import fill_missing_values
from darts.utils.likelihood_models import GaussianLikelihood
from darts.timeseries import concatenate

from neuralprophet import NeuralProphet, utils

import warnings
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

df, names = agrupar(['data/filtros.xlsx', 'data/baterias.xlsx', 'data/discos.xlsx', 'data/pastillas.xlsx'])

df2 = pd.read_csv('data/meteo_olvera.csv', decimal=',')
df2['fecha'] = pd.to_datetime(df2['fecha'], format="%Y-%m-%d")
df2.set_index('fecha', inplace=True)
df2 = df2[['tmed', 'prec', 'hrMedia']]
df2 = df2.resample('B').first()
df2 = df2.fillna(method='backfill')
df2 = df2['2012':]
df = pd.concat([df, df2], axis=1)

df = df.reset_index()

train_df = df[:-130]
test_df = df[-130:]

scaler = MinMaxScaler(feature_range=(0, 1))
#scaler.fit(train_df[['unidades']])


# Definicion TimeSeries
cols = ['tmed', 'prec', 'hrMedia']

train = TimeSeries.from_dataframe(train_df, time_col='fecha', value_cols=names+cols)
test = TimeSeries.from_dataframe(test_df, time_col='fecha', value_cols=names+cols)

transformer = Scaler(scaler)
train = transformer.fit_transform(train)
test = transformer.transform(test)
train = train.add_holidays(country_code='ES', prov='AN')


# def reset_models():
#     mod_blockrnn.reset_model()
#     mod_blockrnn_multi.reset_model()
#     mod_blocklstm.reset_model()
#     mod_blocklstm_multi.reset_model()
#     mod_blockgru.reset_model()
#     mod_blockgru_multi.reset_model()
#     mod_nbeats.reset_model()
#     mod_nbeats_multi.reset_model()
#     global mod_neuralprophet
#     mod_neuralprophet = NeuralProphet()


# Prediccion
# for name in names:
#     reset_models()

#name = 'baterias'

for name in names:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[[name]])


    mod_blockrnn = BlockRNNModel.load(f'models/{name}/blockrnn')

    mod_blocklstm = BlockRNNModel.load(f'models/{name}/blocklstm')

    mod_blockgru = BlockRNNModel.load(f'models/{name}/blockgru')

    mod_prophet = Prophet.load(f'models/{name}/prophet')

    mod_neuralprophet = utils.load(f'models/{name}/neuralprophet')

    mod_nbeats = NBEATSModel.load(f'models/{name}/nbeats')

    # mod_blockrnn_multi = BlockRNNModel.load(f'models/multi/blockrnn_multi')

    # mod_blocklstm_multi = BlockRNNModel.load(f'models/multi/blocklstm_multi')

    # mod_blockgru_multi = BlockRNNModel.load(f'models/multi/blockgru_multi')

    # mod_nbeats_multi = NBEATSModel.load(f'models/multi/nbeats_multi')

    models = [mod_blockrnn, mod_blocklstm, mod_blockgru, mod_prophet, mod_nbeats]
    model_name = ['RNN', 'LSTM', 'GRU', 'Prophet', 'N-Beats']

    #def pred_models(models: list, model_name: list, test: TimeSeries):
    resultados = test[name].pd_dataframe()
    errores = pd.DataFrame()

    # Se realizan las predicciones y calcula el error RMSE
    for i, model in enumerate(models):
        print(model_name[i])
        pred = model.predict(n=130)
        predicion = pred[name].pd_dataframe().rename({name:model_name[i]}, axis=1)
        resultados = pd.concat([resultados, predicion], axis=1)
        err = rmse(pred[name], test[name])
        errores = pd.concat([errores, pd.DataFrame({model_name[i]:[err]}, index=['Diario_escalado'])],axis=1)
        #pred[name].plot(label=model_name[i])

    # Se desescala las predicciones y calcula el error RMSE desescalado
    errores_aux = pd.DataFrame()
    for columna in resultados.columns:
        resultados[columna] = scaler.inverse_transform(resultados[[columna]])
        if columna == name:
            continue
        err=rootmse(resultados[name], resultados[columna])
        errores_aux = pd.concat([errores_aux, pd.DataFrame({columna:[err]}, index=['Diario'])],axis=1)
    errores = pd.concat([errores, errores_aux])

    # Predicciones para NeuralProphet
    predicion = mod_neuralprophet.predict(test_df[['fecha',name]].rename({'fecha':'ds', name:'y'}, axis=1))
    predicion = predicion.rename({'ds':'fecha', 'yhat1':'NeuralProphet'}, axis=1)
    predicion['fecha'] = pd.to_datetime(predicion['fecha'], format="%Y%m%d")
    predicion = predicion.set_index(['fecha'])
    predicion = predicion[['NeuralProphet']]
    resultados = pd.concat([resultados, predicion], axis=1)
    
    err1m=rootmse(resultados[:datetime(2024,1,31)][name], resultados[:datetime(2024,1,31)][columna])
    err3m=rootmse(resultados[:datetime(2024,3,31)][name], resultados[:datetime(2024,3,31)][columna])
    err6m=rootmse(resultados[:datetime(2024,6,30)][name], resultados[:datetime(2024,6,30)][columna])
    errores_aux = pd.concat([errores_aux, pd.DataFrame({columna:[err1m]}, index=['Mensual 1 mes'])],axis=1)
    errores_aux = pd.concat([errores_aux, pd.DataFrame({columna:[err3m]}, index=['Mensual 3 meses'])],axis=1)
    errores_aux = pd.concat([errores_aux, pd.DataFrame({columna:[err6m]}, index=['Mensual 6 meses'])],axis=1)

    err=rootmse(resultados[name], resultados[columna])
    errores = pd.concat([errores, pd.DataFrame({'NeuralProphet':[err]}, index=['Diario'])],axis=1)

    #Agrupación por mes
    resultados_mes = resultados.groupby(pd.Grouper(freq='M'))
    resultados_mes = resultados_mes.sum()

    # Se calcula el error RMSE con los datos agrupados por meses
    errores_aux = pd.DataFrame()
    for columna in resultados_mes.columns:
        if columna == name:
            continue
        err1m=rootmse(resultados_mes[:datetime(2024,1,31)][name], resultados_mes[:datetime(2024,1,31)][columna])
        err3m=rootmse(resultados_mes[:datetime(2024,3,31)][name], resultados_mes[:datetime(2024,3,31)][columna])
        err6m=rootmse(resultados_mes[:datetime(2024,6,30)][name], resultados_mes[:datetime(2024,6,30)][columna])
        errores_aux = pd.concat([errores_aux, pd.DataFrame({columna:[err1m]}, index=['Mensual 1 mes'])],axis=1)
        errores_aux = pd.concat([errores_aux, pd.DataFrame({columna:[err3m]}, index=['Mensual 3 meses'])],axis=1)
        errores_aux = pd.concat([errores_aux, pd.DataFrame({columna:[err6m]}, index=['Mensual 6 meses'])],axis=1)
    errores = pd.concat([errores, errores_aux])

    resultados.to_csv(f'results/resultados_{name}.csv')
    errores_aux.fillna(0, inplace=True)
    errores_aux.to_csv(f'results/errores_{name}.csv')
    print(errores_aux)