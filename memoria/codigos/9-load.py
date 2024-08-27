# Carga de modelos, todos se cargan de forma identica
mod_blockrnn = BlockRNNModel.load(f'models/{name}/blockrnn')

# Agrupacion de modelos sin regresores
models_uni = [mod_blockrnn, mod_blocklstm, mod_blockgru, mod_nbeats, mod_nhits, mod_tcn, mod_dlinear, mod_nlinear, mod_tide, mod_tsmixer]

# Agrupacion de modelos con regresores
models_multi = [mod_blockrnn_multi, mod_blocklstm_multi, mod_blockgru_multi, mod_nbeats_multi, mod_nhits_multi, mod_tcn_multi, mod_dlinear_multi, mod_nlinear_multi, mod_tide_multi, mod_tsmixer_multi]
