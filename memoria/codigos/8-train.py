# Entrenamiento
for name in names:
    reset_models()
    print(name)

    # Modelos sin covariates
    mod_blockrnn.fit(train[f'{name}'])
    mod_blockrnn.save(f'models/{name}/blockrnn')

    # Modelos con regresores pasados
    mod_blockrnn_multi.fit(series=train[name], past_covariates=train.drop_columns(names))
    mod_blockrnn_multi.save(f'models/{name}/blockrnn_multi')

    # Modelos con regresores pasados y futuros
    mod_dlinear_multi.fit(series=train[name], past_covariates=train[['tmed', 'prec', 'hrMedia', 'gasolina', 'diesel']], future_covariates=series['holidays'])
    mod_dlinear_multi.save(f'models/{name}/dlinear_multi')