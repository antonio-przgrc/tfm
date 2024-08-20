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

# Definición de modelos
epochs = 200
batch = 256

mod_blockrnn = BlockRNNModel(
    model='RNN',
    input_chunk_length=130,
    output_chunk_length=130,
    hidden_dim=25,
    n_rnn_layers=2,
    dropout=0.2,
    n_epochs=epochs,
    batch_size=batch,
    #optimizer_kwargs={"lr": 1e-3},
    show_warnings=True,
    model_name='rnn',
    #pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True} 
    pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"} 
)
mod_blockrnn_multi = cp(mod_blockrnn)

mod_blocklstm = BlockRNNModel(
    model='LSTM',
    input_chunk_length=130,
    output_chunk_length=130,
    hidden_dim=25,
    n_rnn_layers=2,
    dropout=0.2,
    n_epochs=epochs,
    batch_size=batch,
    #optimizer_kwargs={"lr": 1e-3},
    show_warnings=True,
    model_name='lstm',
    pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"} 
)
mod_blocklstm_multi = cp(mod_blocklstm)

mod_blockgru = BlockRNNModel(
    model='GRU',
    input_chunk_length=130,
    output_chunk_length=130,
    hidden_dim=25,
    n_rnn_layers=2,
    dropout=0.2,
    n_epochs=epochs,
    batch_size=batch,
    #optimizer_kwargs={"lr": 1e-3},
    show_warnings=True,
    model_name='gru',
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

mod_nbeats = NBEATSModel(
    input_chunk_length=130,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=epochs,
    show_warnings=True,
    batch_size=batch,
    model_name='nbeats',
    pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True} 
)
mod_nbeats_multi = cp(mod_nbeats)

mod_nhits = NHiTSModel(
    input_chunk_length=130,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=epochs,
    show_warnings=True,
    batch_size=batch,
    model_name='nhits',
#    pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"} 
    pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True} 
)
mod_nhits_multi = cp(mod_nhits)

mod_tcn = TCNModel(
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=epochs,
    show_warnings=True,
    batch_size=batch,
    model_name='tcn',
    #pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"} 
    pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True} 

)
mod_tcn_multi = cp(mod_tcn)

mod_dlinear = DLinearModel(
    input_chunk_length=130,
    output_chunk_length=130,
    n_epochs=epochs,
    show_warnings=True,
    batch_size=batch,
    model_name='dlinear',
    pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True} 
#    pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"} 
)
mod_dlinear_multi = cp(mod_dlinear)

mod_nlinear = NLinearModel(
    input_chunk_length=130,
    output_chunk_length=130,
    n_epochs=epochs,
    show_warnings=True,
    batch_size=batch,
    model_name='nlinear',
    pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True} 
#    pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"} 
)
mod_nlinear_multi = cp(mod_nlinear)

mod_tide = TiDEModel(
    input_chunk_length=130,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=epochs,
    show_warnings=True,
    batch_size=batch,
    model_name='tide',
#    pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"}
    pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True} 
)
mod_tide_multi = cp(mod_tide)

mod_tsmixer = TSMixerModel(
    input_chunk_length=130,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=epochs,
    show_warnings=True,
    batch_size=batch,
    model_name='tsmixer',
#    pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"}     
    pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True} 
)
mod_tsmixer_multi = cp(mod_tsmixer)

def reset_models():
    mod_blockrnn.reset_model()
    mod_blocklstm.reset_model()
    mod_blockgru.reset_model()
    mod_nbeats.reset_model()
    mod_nhits.reset_model()
    mod_tcn.reset_model()
    mod_dlinear.reset_model()
    mod_nlinear.reset_model()
    mod_tide.reset_model()
    mod_tsmixer.reset_model()

    mod_blockrnn_multi.reset_model()
    mod_blocklstm_multi.reset_model()
    mod_blockgru_multi.reset_model()
    mod_nbeats_multi.reset_model()
    mod_nhits_multi.reset_model()
    mod_tcn_multi.reset_model()
    mod_dlinear_multi.reset_model()
    mod_nlinear_multi.reset_model()
    mod_tide_multi.reset_model()
    mod_tsmixer_multi.reset_model()

# Entrenamiento
for name in names:
    reset_models()
    print(name)   

    mod_blockrnn.fit(train[f'{name}'])
    mod_blockrnn.save(f'models/{name}/blockrnn')

    mod_blocklstm.fit(train[f'{name}'])
    mod_blocklstm.save(f'models/{name}/blocklstm')

    mod_blockgru.fit(train[f'{name}'])
    mod_blockgru.save(f'models/{name}/blockgru')

    mod_prophet.fit(train[f'{name}'])
    mod_prophet.save(f'models/{name}/prophet')

    mod_nbeats.fit(train[f'{name}'])
    mod_nbeats.save(f'models/{name}/nbeats')

    mod_nhits.fit(train[f'{name}'])
    mod_nhits.save(f'models/{name}/nhits')

    mod_tcn.fit(train[f'{name}'])
    mod_tcn.save(f'models/{name}/tcn')

    mod_dlinear.fit(train[f'{name}'])
    mod_dlinear.save(f'models/{name}/dlinear')

    mod_nlinear.fit(train[f'{name}'])
    mod_nlinear.save(f'models/{name}/nlinear')

    mod_tide.fit(train[f'{name}'])
    mod_tide.save(f'models/{name}/tide')
    
    mod_tsmixer.fit(train[f'{name}'])
    mod_tsmixer.save(f'models/{name}/tsmixer')


    # mod_blockrnn_multi.fit(series=train[name], past_covariates=train.drop_columns(names))
    # mod_blockrnn_multi.save(f'models/{name}/blockrnn_multi')

    # mod_blocklstm_multi.fit(series=train[name], past_covariates=train.drop_columns(names))
    # mod_blocklstm_multi.save(f'models/{name}/blocklstm_multi')

    # mod_blockgru_multi.fit(series=train[name], past_covariates=train.drop_columns(names))
    # mod_blockgru_multi.save(f'models/{name}/blockgru_multi')

    # mod_nbeats_multi.fit(series=train[name], past_covariates=train.drop_columns(names))
    # mod_nbeats_multi.save(f'models/{name}/nbeats_multi')

    # mod_nhits_multi.fit(series=train[name], past_covariates=train.drop_columns(names))
    # mod_nhits_multi.save(f'models/{name}/nhits_multi')

    # mod_tcn_multi.fit(series=train[name], past_covariates=train.drop_columns(names))
    # mod_tcn_multi.save(f'models/{name}/tcn_multi')

    # mod_dlinear_multi.fit(series=train[name], past_covariates=train[['tmed', 'prec', 'hrMedia', 'gasolina', 'diesel']], future_covariates=series['holidays'])
    # mod_dlinear_multi.save(f'models/{name}/dlinear_multi')

    # mod_nlinear_multi.fit(series=train[name], past_covariates=train[['tmed', 'prec', 'hrMedia', 'gasolina', 'diesel']], future_covariates=series['holidays'])
    # mod_nlinear_multi.save(f'models/{name}/nlinear_multi')

    # mod_tide_multi.fit(series=train[name], past_covariates=train[['tmed', 'prec', 'hrMedia', 'gasolina', 'diesel']], future_covariates=series['holidays'])
    # mod_tide_multi.save(f'models/{name}/tide_multi')
    
    # mod_tsmixer_multi.fit(series=train[name], past_covariates=train[['tmed', 'prec', 'hrMedia', 'gasolina', 'diesel']], future_covariates=series['holidays'])
    # mod_tsmixer_multi.save(f'models/{name}/tsmixer_multi')