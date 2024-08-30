# https://stackoverflow.com/questions/71239557/export-tensorboard-with-pytorch-data-into-csv-with-python

import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            wall_time = list(map(lambda x: x.wall_time, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step, "wall_time": wall_time}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

models = [
    "RNN",
    "LSTM",
    "GRU",
    "NBEATS",
    "NHiTS",
    "TCN",
    "Transformer",
    "TFT",
    "DLinear",
    "NLinear",
    "TiDE",
    "TSMixer"
]

for model in models:
    path=f"darts_logs/{model}/logs" #folderpath
    df=tflog2pandas(path)

    tr = df[df["metric"] == "train_loss"]
    val = df[df["metric"] == "val_loss"]

    tr['incremento'] = tr['wall_time'].diff().fillna(0).cumsum()
    val['incremento'] = val['wall_time'].diff().fillna(0).cumsum()

    tr.to_csv(f'results/train-logs/{model}_train_loss.csv')
    val.to_csv(f'results/train-logs/{model}_val_loss.csv')