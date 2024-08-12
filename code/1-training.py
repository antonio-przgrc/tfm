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
    dataframe = pd.read_excel(fichero)
    dataframe = dataframe.rename({'clave1':'fecha', 'uniTotal':'unidades'}, axis=1)
    dataframe['fecha'] = pd.to_datetime(dataframe['fecha'], format="%Y%m%d")
    dataframe = dataframe.set_index(['fecha'])
    dataframe = dataframe.resample('D').first()
    dataframe.fillna(value=0, inplace=True)
    dataframe = dataframe.groupby(pd.Grouper(freq='B'))
    dataframe = dataframe.sum()
    dataframe[dataframe['unidades'] < 0] = 0
    dataframe = dataframe["2010-01-01":"2024-06-30"]
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
df2 = df2[['tmed', 'tmin', 'tmax', 'prec', 'dir', 'velmedia', 'hrMedia']]
df2 = df2.resample('B').first()
df2 = df2.fillna(method='backfill')
df = pd.concat([df, df2], axis=1)

df = df.reset_index()

train_df = df[:-130]
test_df = df[-130:]

scaler = MinMaxScaler(feature_range=(0, 1))
#scaler.fit(train_df[['unidades']])

# Definicion TimeSeries
cols = ['tmed', 'tmin', 'tmax', 'prec', 'dir', 'velmedia', 'hrMedia']

train = TimeSeries.from_dataframe(train_df, time_col='fecha', value_cols=names+cols)
test = TimeSeries.from_dataframe(test_df, time_col='fecha', value_cols=names+cols)

transformer = Scaler(scaler)
train = transformer.fit_transform(train)
test = transformer.transform(test)
train = train.add_holidays(country_code='ES', prov='AN')

# Definición de modelos
epochs = 400
batch = 256

mod_blockrnn = BlockRNNModel(
    model='RNN',
    input_chunk_length=60,
    output_chunk_length=60,
    hidden_dim=25,
    n_rnn_layers=2,
    dropout=0.1,
    n_epochs=epochs,
    batch_size=batch,
    #optimizer_kwargs={"lr": 1e-3},
    show_warnings=True,
    model_name='RNN',
    #pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True} 
    pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"} 
)

mod_blockrnn_multi = cp(mod_blockrnn)

mod_blocklstm = BlockRNNModel(
    model='LSTM',
    input_chunk_length=60,
    output_chunk_length=60,
    hidden_dim=25,
    n_rnn_layers=2,
    dropout=0.1,
    n_epochs=epochs,
    batch_size=batch,
    #optimizer_kwargs={"lr": 1e-3},
    show_warnings=True,
    model_name='LSTM',
    pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"} 
)
mod_blocklstm_multi = cp(mod_blocklstm)

mod_blockgru = BlockRNNModel(
    model='GRU',
    input_chunk_length=60,
    output_chunk_length=60,
    hidden_dim=25,
    n_rnn_layers=2,
    dropout=0.1,
    n_epochs=epochs,
    batch_size=batch,
    #optimizer_kwargs={"lr": 1e-3},
    show_warnings=True,
    model_name='GRU',
    pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"} 
)
mod_blockgru_multi = cp(mod_blockgru)

mod_prophet = Prophet(
    add_seasonalities=None, 
    country_holidays='ES', 
    suppress_stdout_stderror=True, 
    add_encoders=None, 
    cap=None, 
    floor=None
)

mod_neuralprophet = NeuralProphet()

mod_nbeats = NBEATSModel(
    input_chunk_length=60,
    output_chunk_length=60,
    dropout=0.1,
    n_epochs=epochs,
    show_warnings=True,
    batch_size=batch,
    model_name='LSTM',
    pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"} 
)

mod_nbeats_multi = cp(mod_nbeats)

def reset_models():
    mod_blockrnn.reset_model()
    mod_blockrnn_multi.reset_model()
    mod_blocklstm.reset_model()
    mod_blocklstm_multi.reset_model()
    mod_blockgru.reset_model()
    mod_blockgru_multi.reset_model()
    mod_nbeats.reset_model()
    mod_nbeats_multi.reset_model()
    global mod_neuralprophet
    mod_neuralprophet = NeuralProphet()


# Entrenamiento
for name in names:
    reset_models()

    # mod_blockrnn.fit(train[f'{name}'])
    # mod_blockrnn.save(f'models/{name}/blockrnn')

    # mod_blocklstm.fit(train[f'{name}'])
    # mod_blocklstm.save(f'models/{name}/blocklstm')

    # mod_blockgru.fit(train[f'{name}'])
    # mod_blockgru.save(f'models/{name}/blockgru')

    # mod_prophet.fit(train[f'{name}'])
    # mod_prophet.save(f'models/{name}/prophet')

    # mod_neuralprophet.fit(train_df[['fecha',f'{name}']].rename({'fecha':'ds', f'{name}':'y'}, axis=1), batch_size=8)
    # utils.save(mod_neuralprophet, f'models/{name}/neuralprophet')

    # mod_nbeats.fit(train[f'{name}'])
    # mod_nbeats.save(f'models/{name}/nbeats')

# mod_blockrnn_multi.fit(train)
# mod_blockrnn_multi.save(f'models/multi/blockrnn_multi')

# mod_blocklstm_multi.fit(train)
# mod_blocklstm_multi.save(f'models/multi/blocklstm_multi')

# mod_blockgru_multi.fit(train)
# mod_blockgru_multi.save(f'models/multi/blockgru_multi')

# mod_nbeats_multi.fit(train)
# mod_nbeats_multi.save(f'models/multi/nbeats_multi')



##### SCALER PARA NEURALPROPHET