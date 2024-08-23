# Definicion de modelos
EPOCHS = 200
BATCH = 256

mod_blockrnn = BlockRNNModel(
    model='RNN',
    input_chunk_length=260,
    output_chunk_length=130,
    hidden_dim=25,
    n_rnn_layers=2,
    dropout=0.2,
    n_epochs=EPOCHS,
    batch_size=BATCH,
    show_warnings=True,
    model_name='rnn',
    pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"},
    add_encoders={'cyclic': {'past': ['quarter','dayofyear']}}
)
mod_blockrnn_multi = cp(mod_blockrnn)

mod_blocklstm = BlockRNNModel(
    model='LSTM',
    input_chunk_length=260,
    output_chunk_length=130,
    hidden_dim=25,
    n_rnn_layers=2,
    dropout=0.2,
    n_epochs=EPOCHS,
    batch_size=BATCH,
    show_warnings=True,
    model_name='lstm',
    pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"},
    add_encoders={'cyclic': {'past': ['quarter','dayofyear']}}
)
mod_blocklstm_multi = cp(mod_blocklstm)

mod_blockgru = BlockRNNModel(
    model='GRU',
    input_chunk_length=260,
    output_chunk_length=130,
    hidden_dim=25,
    n_rnn_layers=2,
    dropout=0.2,
    n_epochs=EPOCHS,
    batch_size=BATCH,
    show_warnings=True,
    model_name='gru',
    pl_trainer_kwargs={"precision": '64', "accelerator": "cpu"},
    add_encoders={'cyclic': {'past': ['quarter','dayofyear']}}
)
mod_blockgru_multi = cp(mod_blockgru)

mod_prophet = Prophet(
    add_seasonalities=None, 
    country_holidays='ES', 
    suppress_stdout_stderror=True, 
    add_encoders=None, 
    cap=None, 
    floor=None
)

mod_nbeats = NBEATSModel(
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='nbeats',
    pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True},
    add_encoders={'cyclic': {'past': ['quarter','dayofyear']}}
)
mod_nbeats_multi = cp(mod_nbeats)

mod_nhits = NHiTSModel(
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='nhits',
    pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True},
    add_encoders={'cyclic': {'past': ['quarter','dayofyear']}}
)
mod_nhits_multi = cp(mod_nhits)

mod_tcn = TCNModel(
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='tcn',
    pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True},
    add_encoders={'cyclic': {'past': ['quarter','dayofyear']}}
)
mod_tcn_multi = cp(mod_tcn)

mod_dlinear = DLinearModel(
    input_chunk_length=260,
    output_chunk_length=130,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='dlinear',
    pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True},
    add_encoders={'cyclic': {'future': ['quarter','dayofyear']}}
)
mod_dlinear_multi = cp(mod_dlinear)

mod_nlinear = NLinearModel(
    input_chunk_length=260,
    output_chunk_length=130,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='nlinear',
    pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True},
    add_encoders={'cyclic': {'future': ['quarter','dayofyear']}}
)
mod_nlinear_multi = cp(mod_nlinear)

mod_tide = TiDEModel(
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='tide',
    pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True},
    add_encoders={'cyclic': {'future': ['quarter','dayofyear']}}
)
mod_tide_multi = cp(mod_tide)

mod_tsmixer = TSMixerModel(
    input_chunk_length=260,
    output_chunk_length=130,
    dropout=0.2,
    n_epochs=EPOCHS,
    show_warnings=True,
    batch_size=BATCH,
    model_name='tsmixer',
    pl_trainer_kwargs={"precision": '64', "accelerator": "gpu", "devices": -1, "auto_select_gpus": True},
    add_encoders={'cyclic': {'future': ['quarter','dayofyear']}}
)
mod_tsmixer_multi = cp(mod_tsmixer)
