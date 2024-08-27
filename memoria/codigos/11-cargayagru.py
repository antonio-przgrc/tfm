errores = pd.DataFrame()
scaler.fit(df[[name]])

df_res = pd.read_csv(f'results/resultados_{name}.csv')

df_res['fecha'] = pd.to_datetime(df_res['fecha'])
df_res = df_res.set_index('fecha')
df_res = df_res[:130]

df_res_m = df_res.groupby(pd.Grouper(freq='MS')).sum()