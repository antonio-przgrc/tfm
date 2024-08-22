from datetime import datetime, timedelta

import pandas as pd

from copy import copy as cp
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error as rootmse

import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.utils.callbacks import TFMProgressBar
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, TransformerModel, Prophet, BlockRNNModel, NBEATSModel, NHiTSModel, TCNModel, TFTModel, DLinearModel, NLinearModel, TiDEModel, TSMixerModel
from darts.metrics import rmse
from darts.utils.statistics import check_seasonality, plot_acf
import darts.utils.timeseries_generation as tg
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.missing_values import fill_missing_values
from darts.utils.likelihood_models import GaussianLikelihood
from darts.timeseries import concatenate

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

df, names = agrupar(['data/filtros.csv', 'data/baterias.csv', 'data/aceites.csv', 'data/limpiaparabrisas.csv'])

# Meteorología
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
    mod_blockrnn = BlockRNNModel.load(f'models/{name}/blockrnn')
    mod_blockrnn_multi = BlockRNNModel.load(f'models/{name}/blockrnn_multi')
    mod_blocklstm = BlockRNNModel.load(f'models/{name}/blocklstm')
    mod_blocklstm_multi = BlockRNNModel.load(f'models/{name}/blocklstm_multi')
    mod_blockgru = BlockRNNModel.load(f'models/{name}/blockgru')
    mod_blockgru_multi =  BlockRNNModel.load(f'models/{name}/blockgru_multi')
    mod_prophet = Prophet.load(f'models/{name}/prophet')
    mod_nbeats = NBEATSModel.load(f'models/{name}/nbeats')
    mod_nbeats_multi = NBEATSModel.load(f'models/{name}/nbeats_multi')
    mod_nhits = NHiTSModel.load(f'models/{name}/nhits')
    mod_nhits_multi = NHiTSModel.load(f'models/{name}/nhits_multi')
    mod_tcn = TCNModel.load(f'models/{name}/tcn')
    mod_tcn_multi = TCNModel.load(f'models/{name}/tcn_multi')
    mod_dlinear = DLinearModel.load(f'models/{name}/dlinear')
    mod_dlinear_multi = DLinearModel.load(f'models/{name}/dlinear_multi')
    mod_nlinear = NLinearModel.load(f'models/{name}/nlinear')
    mod_nlinear_multi = NLinearModel.load(f'models/{name}/nlinear_multi')
    mod_tide = TiDEModel.load(f'models/{name}/tide')
    mod_tide_multi =  TiDEModel.load(f'models/{name}/tide_multi')
    mod_tsmixer = TSMixerModel.load(f'models/{name}/tsmixer')
    mod_tsmixer_multi = TSMixerModel.load(f'models/{name}/tsmixer_multi')

    models_uni = [mod_blockrnn, mod_blocklstm, mod_blockgru, mod_nbeats, mod_nhits,
    mod_tcn, mod_dlinear, mod_nlinear, mod_tide, mod_tsmixer]

    models_multi = [mod_blockrnn_multi, mod_blocklstm_multi, mod_blockgru_multi, mod_nbeats_multi,
    mod_nhits_multi, mod_tcn_multi, mod_dlinear_multi, mod_nlinear_multi, mod_tide_multi, mod_tsmixer_multi]

    resultados = test[name].pd_dataframe()
    errores = pd.DataFrame()

    print(name)
    print('prophet')
    pred = mod_prophet.predict(n=130)
    predicion = pred[name].pd_dataframe().rename({name:'prophet'}, axis=1)
    resultados = pd.concat([resultados, predicion], axis=1)

    # Se realizan las predicciones de los modelos sin regresor
    for i, model in enumerate(models_uni):
        print(model.model_name)
        pred = model.predict(n=130)
        predicion = pred[name].pd_dataframe().rename({name:model.model_name}, axis=1)
        resultados = pd.concat([resultados, predicion], axis=1)
    
    # # Se realizan las predicciones de los modelos con regresor
    for i, model in enumerate(models_multi):
        print(model.model_name)
        pred = model.predict(n=130)
        predicion = pred[name].pd_dataframe().rename({name:model.model_name+'_multi'}, axis=1)
        resultados = pd.concat([resultados, predicion], axis=1)
        
    resultados.to_csv(f'results/resultados_{name}.csv')
