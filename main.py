
#%%
from data import DataPross ,DataProvider
from model import lstm
import numpy as np
import torch




df = DataPross.Data('data/EURUSDDayli.csv')

df.clean()
df.normalize()
# df.visualize()
ddf = df.df


Train = DataProvider.TrendPredictionDataset(ddf, sequence_length=30, forecast_horizon=10,
                 target_col='Close', threshold=0.05)


