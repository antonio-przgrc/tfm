mod_blockrnn = BlockRNNModel.load(f'models/{name}/blockrnn')
mod_blockrnn_multi = BlockRNNModel.load(f'models/{name}/blockrnn_multi')
mod_blocklstm = BlockRNNModel.load(f'models/{name}/blocklstm')
mod_blocklstm_multi = BlockRNNModel.load(f'models/{name}/blocklstm_multi')
mod_blockgru = BlockRNNModel.load(f'models/{name}/blockgru')
mod_blockgru_multi =  BlockRNNModel.load(f'models/{name}/blockgru_multi')
mod_prophet = Prophet.load(f'models/{name}/prophet')
mod_nbeats = NBEATSModel.load(f'models/{name}/nbeats')
mod_nbeats_multi = NBEATSModel.load(f'models/{name}/nbeats_multi')
mod_nhits = NHiTSModel.load(f'models/{name}/nhits')
mod_nhits_multi = NHiTSModel.load(f'models/{name}/nhits_multi')
mod_tcn = TCNModel.load(f'models/{name}/tcn')
mod_tcn_multi = TCNModel.load(f'models/{name}/tcn_multi')
mod_dlinear = DLinearModel.load(f'models/{name}/dlinear')
mod_dlinear_multi = DLinearModel.load(f'models/{name}/dlinear_multi')
mod_nlinear = NLinearModel.load(f'models/{name}/nlinear')
mod_nlinear_multi = NLinearModel.load(f'models/{name}/nlinear_multi')
mod_tide = TiDEModel.load(f'models/{name}/tide')
mod_tide_multi =  TiDEModel.load(f'models/{name}/tide_multi')
mod_tsmixer = TSMixerModel.load(f'models/{name}/tsmixer')
mod_tsmixer_multi = TSMixerModel.load(f'models/{name}/tsmixer_multi')

models_uni = [mod_blockrnn, mod_blocklstm, mod_blockgru, mod_nbeats, mod_nhits,
mod_tcn, mod_dlinear, mod_nlinear, mod_tide, mod_tsmixer]

models_multi = [mod_blockrnn_multi, mod_blocklstm_multi, mod_blockgru_multi, mod_nbeats_multi,
mod_nhits_multi, mod_tcn_multi, mod_dlinear_multi, mod_nlinear_multi, mod_tide_multi, mod_tsmixer_multi]
