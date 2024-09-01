import warnings
from datetime import datetime
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


scaler = MinMaxScaler(feature_range=(0, 1))

with open('results/df.pickle', 'rb') as f:
    df = pickle.load(f)

with open('results/names.pickle', 'rb') as f:
    names = pickle.load(f)

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
    ax.axis([datetime(2024,1,1), datetime(2024,6,1), min(dfp_mes.min())*0.9, max(dfp_mes.max())*1.1])
    leg = ax.legend(loc='best', bbox_to_anchor=(1, 1))
    ax.set_ylabel('Unidades')
    ax.set_title(f'{name}')
    plt.savefig(f'figs/prediccion_mes_{name}.pdf', bbox_inches='tight')


# Figuras resultados entrenamiento
models = [
    "RNN",
    "LSTM",
    "GRU",
    "NBEATS",
    "NHiTS",
    "TCN",
    "Transformer",
    "TFT",
    "DLinear",
    "NLinear",
    "TiDE",
    "TSMixer"
]
df_train_reg = pd.read_csv('results/tiempos.csv', index_col=0)

for model in models:
    dftrain = pd.read_csv(f'results/train-logs/{model}_train_loss.csv', index_col=0)
    dfval = pd.read_csv(f'results/train-logs/{model}_val_loss.csv', index_col=0)
    
    scaler = MinMaxScaler(feature_range=(0, df_train_reg.loc[model].epochs))
    dftrain["step"] = scaler.fit_transform(dftrain[["step"]])
    dfval["step"] = scaler.fit_transform(dfval[["step"]])

    dftrain = dftrain.set_index(['step'])
    dfval = dfval.set_index(['step'])

    fig, ax = plt.subplots()
    ax.plot(dftrain.value, label='train_loss')
    ax.plot(dfval.value, label='val_loss')
    ax.set_title(f'Entrenamiento de {model}')
    ax.minorticks_on()
    ax.grid(which='major', alpha = 0.65, linestyle='-')
    ax.grid(which='minor', alpha = 0.25, linestyle='--')
    ax.axis(xmin = dfval.head(1).index[0], xmax = dftrain.tail(1).index[0])
    leg = ax.legend(loc='upper right')
    ax.set_ylabel('Valor de error')
    ax.set_xlabel('Epoch')
    ax.set_title(f'Entrenamiento del modelo {model}')
    plt.savefig(f'figs/{model}_loss.pdf', bbox_inches='tight')