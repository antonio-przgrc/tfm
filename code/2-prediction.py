import warnings
import pandas as pd
import pickle
from darts.models import BlockRNNModel, NBEATSModel, NHiTSModel, TCNModel, TransformerModel, TFTModel, DLinearModel, NLinearModel, TiDEModel, TSMixerModel 
warnings.filterwarnings('ignore')

with open('results/test.pickle', 'rb') as f:
    test = pickle.load(f)

with open('results/names.pickle', 'rb') as f:
    names = pickle.load(f)

models = [
    BlockRNNModel.load_from_checkpoint(model_name="RNN", best=True),
    BlockRNNModel.load_from_checkpoint(model_name="LSTM", best=True),
    BlockRNNModel.load_from_checkpoint(model_name="GRU", best=True),
    NBEATSModel.load_from_checkpoint(model_name="NBEATS", best=True),
    NHiTSModel.load_from_checkpoint(model_name="NHiTS", best=True),
    TCNModel.load_from_checkpoint(model_name="TCN", best=True),
    TransformerModel.load_from_checkpoint(model_name="Transformer", best=True),
    TFTModel.load_from_checkpoint(model_name="TFT", best=True),
    DLinearModel.load_from_checkpoint(model_name="DLinear", best=True),
    NLinearModel.load_from_checkpoint(model_name="NLinear", best=True),
    TiDEModel.load_from_checkpoint(model_name="TiDE", best=True),
    TSMixerModel.load_from_checkpoint(model_name="TSMixer", best=True),
]

resultados = test[names].pd_dataframe()

print(names)

# Se realizan las predicciones de cada modelo
for model in models:
    print(model.model_name)
    pred = model.predict(n=130)
    prediccion = pred[names].pd_dataframe()
    col_name = (model.model_name + "_" + prediccion.columns.values).tolist()
    prediccion = prediccion.set_axis(labels=col_name, axis=1)
    resultados = pd.concat([resultados, prediccion], axis=1)


# Se separa por familia
for name in names:
    res = resultados.filter(regex=name)

    res.columns = res.columns.str.replace("_" + name, "")

    res.to_csv(f'results/resultados_{name}.csv')