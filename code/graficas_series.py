import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
names = ['baterias','filtros','limpiaparabrisas','aceites']

for name in names:
    df = pd.read_csv(f'~/tfm/code/data/{name}.csv')
    df = df.rename({'clave1':'fecha', 'uniTotal':'unidades'}, axis=1)
    df['fecha'] = pd.to_datetime(df['fecha'], format="%Y%m%d")
    df = df.set_index(['fecha'])
    df = df.resample('D').first()
    df.fillna(value=0, inplace=True)
    df = df.groupby(pd.Grouper(freq='B')).sum()
    df[df['unidades'] < 0] = 0
    df = df["2012-01-01":"2024-06-30"]

    fig,ax = plt.subplots()
    ax.plot(df)
    ax.minorticks_on()
    ax.grid(which='major', alpha = 0.65, linestyle='-')
    ax.grid(which='minor', alpha = 0.25, linestyle='--')
    ax.set_title(f'Unidades de {name} vendidas')
    ax.set_ylabel('Unidades vendidas')
    ax.axis(ymin=0, ymax=max(df['unidades']), xmin=datetime(2012,1,1), xmax=datetime(2024,6,30))
    fig.autofmt_xdate()
    fig.savefig(f'/home/antonio/tfm/memoria/imagenes/grafica_{name}.pdf')