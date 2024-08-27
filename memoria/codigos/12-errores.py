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
