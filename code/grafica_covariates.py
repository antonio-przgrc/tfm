import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv(f'~/tfm/code/data/holidays.csv')
df['fecha'] = pd.to_datetime(df['fecha'], format="%Y-%m-%d")
df = df.set_index(['fecha'])
df = df.resample('D').first()
df.fillna(value=0, inplace=True)
df = df.groupby(pd.Grouper(freq='B')).sum()
df = df["2012-01-01":"2024-06-30"]

fig,ax = plt.subplots()
ax.plot(df)
ax.minorticks_on()
ax.grid(which='major', alpha = 0.65, linestyle='-')
ax.grid(which='minor', alpha = 0.25, linestyle='--')
ax.set_title(f'Días festivos')
ax.axis(ymin=0, ymax=1.02, xmin=datetime(2012,1,1), xmax=datetime(2024,1,1))
fig.autofmt_xdate()
fig.savefig(f'/home/antonio/tfm/memoria/imagenes/grafica_holidays.pdf')


# Precio combustible
df = pd.read_csv('data/carburante.csv', decimal=',')
df['fecha'] = pd.to_datetime(df['fecha']+'0', format="%Y-%W%w")
df = df.set_index('fecha').sort_index()
df = df.resample('B').first().ffill()
df = df['2012':]

fig,ax = plt.subplots()
ax.plot(df)
ax.minorticks_on()
ax.grid(which='major', alpha = 0.65, linestyle='-')
ax.grid(which='minor', alpha = 0.25, linestyle='--')
ax.set_title(f'Evolución precio de los carburantes')
ax.legend(['Gasolina','Diesel'])
ax.set_ylabel('Precio (€)')
ax.axis(ymin=min(df.min()), ymax=max(df.max()), xmin=datetime(2012,1,1), xmax=datetime(2024,1,1))
fig.autofmt_xdate()
fig.savefig(f'/home/antonio/tfm/memoria/imagenes/grafica_carburantes.pdf')