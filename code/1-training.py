import warnings
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import Prophet, BlockRNNModel, NBEATSModel, NHiTSModel, TCNModel, DLinearModel, NLinearModel, TiDEModel, TSMixerModel
from pytorch_lightning.callbacks import EarlyStopping

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

transformer = Scaler(scaler)
series = transformer.fit_transform(series)
series = series.astype(np.float32)

train, test = series.split_after(pd.Timestamp(year=2023, month=12, day=31))
_, val = train.split_after(pd.Timestamp(year=2022, month=12, day=31))

# Definicion de modelos
EPOCHS = 200
BATCH = 16
INPUT = 260
OUTPUT = 130
DROPOUT = 0.2

my_stopper = EarlyStopping(
    monitor="train_loss",  
    patience=10,
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

encoders = {
    'cyclic': {'past': ['quarter','dayofyear']},
    'transformer': Scaler()
}

mod_blockrnn = BlockRNNModel(
    model='RNN',
    input_chunk_length=260,
    output_chunk_length=130,
    hidden_dim=25,
    n_rnn_layers=2,
    dropout=0.2,
    n_epochs=EPOCHS,
    batch_size=BATCH,
    show_warnings=True,
    model_name='RNN',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
    add_encoders=encoders
)

mod_blocklstm = BlockRNNModel(
    model='LSTM',
    input_chunk_length=260,
    output_chunk_length=130,
    hidden_dim=25,
    n_rnn_layers=2,
    dropout=0.2,
    n_epochs=EPOCHS,
    batch_size=BATCH,
    show_warnings=True,
    model_name='LSTM',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
    add_encoders=encoders
)

mod_blockgru = BlockRNNModel(
    model='GRU',
    input_chunk_length=260,
    output_chunk_length=130,
    hidden_dim=25,
    n_rnn_layers=2,
    dropout=0.2,
    n_epochs=EPOCHS,
    batch_size=BATCH,
    show_warnings=True,
    model_name='GRU',
    save_checkpoints=True,
    log_tensorboard=True,
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
    add_encoders=encoders
)

mod_prophet = Prophet(
    add_seasonalities=None, 
    country_holidays='ES', 
    suppress_stdout_stderror=True, 
    force_reset=True,
    pl_trainer_kwargs=pl_trainer_kwargs,
    add_encoders=encoders
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
    add_encoders=encoders
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
    add_encoders=encoders
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
    add_encoders=encoders
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
    add_encoders=encoders
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
    add_encoders=encoders
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
    add_encoders=encoders
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
    add_encoders=encoders
)

models = [
    mod_blockrnn,
    mod_blocklstm,
    mod_blockgru,
    mod_nbeats,
    mod_nhits,
    mod_tcn,
    mod_dlinear,
    mod_nlinear,
    mod_tide,
    mod_tsmixer,
]


def reset_models():
    for model in models:
        model.reset_model()

tiempo_ejecucion = pd.DataFrame()

# Entrenamiento
# for name in names:
#     reset_models()
#     print(name)

for model in models:
    print(model.model_name)
    my_stopper.wait_count = 0 # Reinicio de patience

    tiempo1 = time.time()
    if model.supports_future_covariates:
        model.fit(series=train[names], past_covariates=train[['tmed', 'prec', 'hrMedia', 'gasolina', 'diesel']], future_covariates=series['holidays'], dataloader_kwargs={"num_workers": 12}, val_past_covariates=val)
    else:
        model.fit(series=train[names], past_covariates=train.drop_columns(names), dataloader_kwargs={"num_workers": 12}, val_past_covariates=val)
    tiempo_ejecucion["tiempo"] = {model.model_name: (time.time() - tiempo1)}

    # model.save(f'models/{name}/{model.model_name}')

tiempo_ejecucion.to_csv('results/tiempos.csv')