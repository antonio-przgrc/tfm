import pandas as pd

df = pd.read_csv(f'data/filtros.csv')
df = df.rename({'clave1':'fecha', 'uniTotal':'unidades'}, axis=1)
df['fecha'] = pd.to_datetime(df['fecha'], format="%Y%m%d")
df = df.set_index(['fecha'])
df = df.resample('D').first()
df.fillna(value=0, inplace=True)
df = df.groupby(pd.Grouper(freq='B')).sum()
df[df['unidades'] < 0] = 0
df = df["2010-01-01":"2024-06-30"]