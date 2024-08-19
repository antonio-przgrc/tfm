import pandas as pd
from prophet import Prophet
import numpy as np

names = ['baterias','filtros','aceites','limpiaparabrisas']

for name in names:
    df = pd.read_csv(f'data/{name}.csv')

    df = df.rename({'clave1':'fecha', 'uniTotal':'unidades'}, axis=1)
    df['fecha'] = pd.to_datetime(df['fecha'], format="%Y%m%d")
    df = df.set_index(['fecha'])
    df = df.resample('D').first()
    df.fillna(value=0, inplace=True)
    df = df.groupby(pd.Grouper(freq='B'))
    df = df.sum()
    df[df['unidades'] < 0] = 0
    df = df["2012-01-01":"2023-12-31"]

    df = df.reset_index()
    df = df.rename({'fecha':'ds', 'unidades':'y'}, axis=1)

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=1)

    forecast = m.predict(future)

    fig = m.plot_components(forecast,weekly_start=1)
    trend, weekly, yearly = fig.get_axes()

    trend.set_ylabel('Tendencia')
    weekly.set_ylabel('Estacionalidad semanal')
    yearly.set_ylabel('Estacionalidad anual')

    trend.set_xlabel('Fecha')
    weekly.set_xlabel('Día de la semana')
    yearly.set_xlabel('Día del año')

    x = np.arange(7)
    weekly.set_xticks(x,['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'])

    trend.minorticks_on()
    trend.grid(which='major', alpha = 0.65, linestyle='-')
    trend.grid(which='minor', alpha = 0.25, linestyle='--')

    weekly.minorticks_on()
    weekly.grid(which='major', alpha = 0.65, linestyle='-')
    weekly.grid(which='minor', alpha = 0.25, linestyle='--')

    yearly.minorticks_on()
    yearly.grid(which='major', alpha = 0.65, linestyle='-')
    yearly.grid(which='minor', alpha = 0.25, linestyle='--')

    fig.savefig(f'../memoria/imagenes/comps_{name}.pdf')