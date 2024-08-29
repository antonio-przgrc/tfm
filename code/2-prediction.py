import warnings
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import Prophet, BlockRNNModel, NBEATSModel, NHiTSModel, TCNModel, DLinearModel, NLinearModel, TiDEModel, TSMixerModel

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

train, test = series.split_after(pd.Timestamp(year=2023, month=12, day=31))


models = [
    BlockRNNModel.load_from_checkpoint(model_name="RNN", best=True),
    BlockRNNModel.load_from_checkpoint(model_name="LSTM", best=True),
    BlockRNNModel.load_from_checkpoint(model_name="GRU", best=True),
    NBEATSModel.load_from_checkpoint(model_name="NBEATS", best=True),
    NHiTSModel.load_from_checkpoint(model_name="NHiTS", best=True),
    TCNModel.load_from_checkpoint(model_name="TCN", best=True),
    DLinearModel.load_from_checkpoint(model_name="DLinear", best=True),
    NLinearModel.load_from_checkpoint(model_name="NLinear", best=True),
    TiDEModel.load_from_checkpoint(model_name="TiDE", best=True),
    TSMixerModel.load_from_checkpoint(model_name="TSMixer", best=True),
]

resultados = test[names].pd_dataframe()

print(names)

# Se realizan las predicciones de cada modelo
for model in models:
    print(model.model_name)
    pred = model.predict(n=130)
    prediccion = pred[names].pd_dataframe()
    col_name = (model.model_name + "_" + prediccion.columns.values).tolist()
    prediccion = prediccion.set_axis(labels=col_name, axis=1)
    resultados = pd.concat([resultados, prediccion], axis=1)


# Se separa por familia
for name in names:
    res = resultados.filter(regex=name)

    res.columns = res.columns.str.replace("_" + name, "")

    res.to_csv(f'results/resultados_{name}.csv')