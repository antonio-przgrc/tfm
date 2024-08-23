from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/meteo_olvera.csv', decimal=',')
df['fecha'] = pd.to_datetime(df['fecha'], format="%Y-%m-%d")
df.set_index('fecha', inplace=True)
df = df[['tmed', 'prec', 'hrMedia']]
df = df.resample('B').first()
df = df.fillna(method='backfill')
df = df['2012':]

fig, ax = plt.subplots(nrows=3, ncols=1)
ax[0].plot(df['tmed'])
ax[1].plot(df['prec'])
ax[2].plot(df['hrMedia'])

for i in ax:
    i.minorticks_on()
    i.grid(which='major', alpha = 0.65, linestyle='-')
    i.grid(which='minor', alpha = 0.25, linestyle='--')
    i.axis(xmin=datetime(2012,1,1), xmax=datetime(2024,1,1))

ax[2].axis(ymax=99.9999)

ax[0].set_ylabel('Temperatura (ºC)')
ax[0].legend(['Temperatura'], loc='upper left')
ax[1].set_ylabel('Precipitaciones (mm)')
ax[1].legend(['Precipitaciones'], loc='upper left')
ax[2].set_ylabel('Humedad relativa (%)')
ax[2].legend(['Humedad'], loc='upper left')

fig.tight_layout()

fig.savefig('/home/antonio/tfm/memoria/imagenes/grafica_meteo.pdf')
