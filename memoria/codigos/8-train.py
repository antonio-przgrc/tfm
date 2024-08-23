# Entrenamiento
for name in names:
    reset_models()
    print(name)

    # Modelos sin covariates
    mod_blockrnn.fit(train[f'{name}'])
    mod_blockrnn.save(f'models/{name}/blockrnn')

    mod_blocklstm.fit(train[f'{name}'])
    mod_blocklstm.save(f'models/{name}/blocklstm')

    mod_blockgru.fit(train[f'{name}'])
    mod_blockgru.save(f'models/{name}/blockgru')

    mod_prophet.fit(train[f'{name}'])
    mod_prophet.save(f'models/{name}/prophet')

    mod_nbeats.fit(train[f'{name}'])
    mod_nbeats.save(f'models/{name}/nbeats')

    mod_nhits.fit(train[f'{name}'])
    mod_nhits.save(f'models/{name}/nhits')

    mod_tcn.fit(train[f'{name}'])
    mod_tcn.save(f'models/{name}/tcn')

    mod_dlinear.fit(train[f'{name}'])
    mod_dlinear.save(f'models/{name}/dlinear')

    mod_nlinear.fit(train[f'{name}'])
    mod_nlinear.save(f'models/{name}/nlinear')

    mod_tide.fit(train[f'{name}'])
    mod_tide.save(f'models/{name}/tide')

    mod_tsmixer.fit(train[f'{name}'])
    mod_tsmixer.save(f'models/{name}/tsmixer')

    # Modelos con covariates
    mod_blockrnn_multi.fit(series=train[name], past_covariates=train.drop_columns(names))
    mod_blockrnn_multi.save(f'models/{name}/blockrnn_multi')

    mod_blocklstm_multi.fit(series=train[name], past_covariates=train.drop_columns(names))
    mod_blocklstm_multi.save(f'models/{name}/blocklstm_multi')

    mod_blockgru_multi.fit(series=train[name], past_covariates=train.drop_columns(names))
    mod_blockgru_multi.save(f'models/{name}/blockgru_multi')

    mod_nbeats_multi.fit(series=train[name], past_covariates=train.drop_columns(names))
    mod_nbeats_multi.save(f'models/{name}/nbeats_multi')

    mod_nhits_multi.fit(series=train[name], past_covariates=train.drop_columns(names))
    mod_nhits_multi.save(f'models/{name}/nhits_multi')

    mod_tcn_multi.fit(series=train[name], past_covariates=train.drop_columns(names))
    mod_tcn_multi.save(f'models/{name}/tcn_multi')

    mod_dlinear_multi.fit(series=train[name], past_covariates=train[['tmed', 'prec', 'hrMedia', 'gasolina', 'diesel']], future_covariates=series['holidays'])
    mod_dlinear_multi.save(f'models/{name}/dlinear_multi')

    mod_nlinear_multi.fit(series=train[name], past_covariates=train[['tmed', 'prec', 'hrMedia', 'gasolina', 'diesel']], future_covariates=series['holidays'])
    mod_nlinear_multi.save(f'models/{name}/nlinear_multi')

    mod_tide_multi.fit(series=train[name], past_covariates=train[['tmed', 'prec', 'hrMedia', 'gasolina', 'diesel']], future_covariates=series['holidays'])
    mod_tide_multi.save(f'models/{name}/tide_multi')
    
    mod_tsmixer_multi.fit(series=train[name], past_covariates=train[['tmed', 'prec', 'hrMedia', 'gasolina', 'diesel']], future_covariates=series['holidays'])
    mod_tsmixer_multi.save(f'models/{name}/tsmixer_multi')