import warnings
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error as rmse

warnings.filterwarnings('ignore')

# Definicion de scaler y carga de nombres y datos
scaler = MinMaxScaler(feature_range=(0, 1))

with open('results/df.pickle', 'rb') as f:
    df = pickle.load(f)

with open('results/names.pickle', 'rb') as f:
    names = pickle.load(f)

for name in names:
    errores = pd.DataFrame()
    scaler.fit(df[[name]])

    df_res = pd.read_csv(f'results/resultados_{name}.csv')

    df_res['fecha'] = pd.to_datetime(df_res['fecha'])
    df_res = df_res.set_index('fecha')
    df_res = df_res[:130]

    df_res_m = df_res.groupby(pd.Grouper(freq='MS')).sum()

    idx_names = ['Error diario', 'Error mensual']
    for model in df_res.columns:
        if model == name:
            continue
        error_d = rmse(df_res[name], df_res[model])
        error_m = rmse(df_res_m[name], df_res_m[model])
        errores = pd.concat([errores, pd.DataFrame({model:[error_d, error_m]}, index=idx_names)],axis=1)
    errores2 = pd.DataFrame()
    errores2[errores.columns] = scaler.inverse_transform(errores)
    errores2.index = ['Error diario (abs)', 'Error mensual (abs)']

    errores = pd.concat([errores, errores2])

    errores.to_csv(f'results/errores_{name}.csv')

    # Tracking signal
    dfts = df_res_m.reset_index()
    for col in dfts.columns:
        if (col == name) or (col == 'fecha'):
            continue

        dfts[col+'_ts'] = 0
    
        for i in range(0,len(df)):
            errors = dfts[name][0:i+1] - dfts[col][0:i+1]
            bias = np.sum(errors)
            mad = np.mean(np.abs(errors))

            if mad == 0:
                ts = float('inf')
            else:
                ts = bias/mad

            dfts[col+'_ts'][i] = ts

    dfts.to_csv(f'results/ts_{name}.csv')