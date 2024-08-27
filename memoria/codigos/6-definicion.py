# Definicion de modelos
EPOCHS = 200
BATCH = 256

# Definicion para modelo tipo RNN
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

# Definicion para modelo Prophet
mod_prophet = Prophet(
    add_seasonalities=None, 
    country_holidays='ES', 
    suppress_stdout_stderror=True, 
    add_encoders=None, 
    cap=None, 
    floor=None
)

# Definicion tipo para modelos NBEATS, NHiTS, TCN, DLinear, NLinear, TiDE y TSMixer
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