import warnings
import time
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import BlockRNNModel, NBEATSModel, NHiTSModel, TCNModel, TransformerModel, TFTModel, DLinearModel, NLinearModel, TiDEModel, TSMixerModel 
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
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

df = df.reset_index()

scaler = MinMaxScaler(feature_range=(0, 1))

# Definicion TimeSeries
cols = ['tmed','prec', 'hrMedia', 'gasolina', 'diesel']

series = TimeSeries.from_dataframe(df, time_col='fecha', value_cols=names+cols)

series = series.add_holidays(country_code='ES', prov='AN')

dat_atr = [
    "day",
    "dayofweek",
    "dayofyear",
    "month",
    "quarter"
]

for atr in dat_atr:
    series = series.add_datetime_attribute(attribute=atr, cyclic=True)

transformer = Scaler(scaler)
series = transformer.fit_transform(series)
series = series.astype(np.float32)

train, test = series.split_after(pd.Timestamp(year=2023, month=12, day=31))
_, val = train.split_after(pd.Timestamp(year=2022, month=6, day=30))

# Guardado de para proximos usos
series.to_pickle('results/series.pickle')
train.to_pickle('results/train.pickle')
test.to_pickle('results/test.pickle')
with open('results/transformer.pickle', 'wb') as f:
    pickle.dump(transformer, f)
with open('results/names.pickle', 'wb') as f:
    pickle.dump(names, f)

# Definicion de parametros
EPOCHS = 300
BATCH = 32
INPUT = 260
OUTPUT = 130
DROPOUT = 0.1

my_stopper = EarlyStopping(
    monitor="val_loss",  
    patience=20,
    min_delta=0.0001,
    mode='min',
)

pl_trainer_kwargs = {
    "callbacks": [my_stopper],
    "precision": '32',
    "accelerator": "gpu",
    "devices": -1,
    "auto_select_gpus": True
}


mod_blockrnn = BlockRNNModel(
    model='RNN',
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='RNN',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
)

mod_blocklstm = BlockRNNModel(
    model='LSTM',
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='LSTM',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
)

mod_blockgru = BlockRNNModel(
    model='GRU',
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    batch_size=BATCH,
    show_warnings=True,
    model_name='GRU',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
)

mod_nbeats = NBEATSModel(
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='NBEATS',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
)

mod_nhits = NHiTSModel(
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='NHiTS',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
)

mod_tcn = TCNModel(
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='TCN',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
)

mod_transformer = TransformerModel(
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='Transformer',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
)

mod_tft = TFTModel(
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='TFT',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
)

mod_dlinear = DLinearModel(
    input_chunk_length=260,
    output_chunk_length=130,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='DLinear',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
)

mod_nlinear = NLinearModel(
    input_chunk_length=260,
    output_chunk_length=130,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='NLinear',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
)

mod_tide = TiDEModel(
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='TiDE',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
)

mod_tsmixer = TSMixerModel(
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='TSMixer',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
)

models = [
    mod_blockrnn,
    mod_blocklstm,
    mod_blockgru,
    mod_nbeats,
    mod_nhits,
    mod_tcn,
    mod_transformer,
    mod_tft,
    mod_dlinear,
    mod_nlinear,
    mod_tide,
    mod_tsmixer,
]


train_log = pd.DataFrame()

for model in models:
    print(model.model_name)
    
    #Reinicio de EarlyStopper
    my_stopper.best_score = torch.tensor(np.Inf)
    my_stopper.wait_count = 0

    tiempo1 = time.time()

    if model.supports_future_covariates:
        model.fit(
            series=train[names],
            past_covariates=train[['tmed', 'prec', 'hrMedia', 'gasolina', 'diesel']],
            future_covariates=series[['holidays', 'day_sin', 'day_cos', 'dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin', 'dayofyear_cos', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']],
            val_series=val[names],
            val_past_covariates=val[['tmed', 'prec', 'hrMedia', 'gasolina', 'diesel']],
            val_future_covariates=val[['holidays', 'day_sin', 'day_cos', 'dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin', 'dayofyear_cos', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']],
            dataloader_kwargs={"num_workers": 12}
            )
    else:
        model.fit(
            series=train[names],
            past_covariates=train.drop_columns(names),
            val_series=val[names],
            val_past_covariates=val.drop_columns(names),
            dataloader_kwargs={"num_workers": 12},
            )
    
    train_log = pd.concat([train_log, pd.DataFrame({"tiempo":(time.time() - tiempo1), "epochs": model.epochs_trained},index=[model.model_name])])


train_log.to_csv('results/tiempos.csv')