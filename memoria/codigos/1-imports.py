from copy import copy as cp
import warnings
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error as rmse

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import Prophet, BlockRNNModel, NBEATSModel, NHiTSModel, TCNModel, DLinearModel, NLinearModel, TiDEModel, TSMixerModel

warnings.filterwarnings('ignore')
