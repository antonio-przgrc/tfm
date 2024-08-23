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
