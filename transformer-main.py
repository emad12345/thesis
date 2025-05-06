import os
print(os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

from data import DataPross ,DataProvider
from model import transformer
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


#%%
data = DataPross.Data('data/EURUSDDayli.csv')

#%%
data.clean()
data.normalize()
#%%
#data.visualize()
#%%
dataa = data.df
#%%
train_df, val_df = train_test_split (dataa, test_size=0.3, shuffle=False)
val_df , test_df = train_test_split(val_df, test_size=0.5, shuffle=False)

#%%

train_dataset = DataProvider.TransformerFinanceDataset(train_df , sequence_length=30, forecast_horizon=1, target_cols=['Close'])
val_dataset = DataProvider.TransformerFinanceDataset(val_df , sequence_length=30, forecast_horizon=1, target_cols=['Close'])
test_dataset = DataProvider.TransformerFinanceDataset(test_df , sequence_length=30, forecast_horizon=1, target_cols=['Close'])
#%%

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64)

#%%
model = transformer.TransformerTimeSeriesModel(input_size=4, model_dim=128, num_heads=8, num_layers=4, dropout=0.2, output_size=1)

model.train_model( train_loader, val_loader, lr=1e-3, epochs=100, device='cuda')

#%%


criterion = nn.MSELoss()
writer = SummaryWriter(log_dir=os.path.join("runs", "Transformer_Finance_Experiment"))

model.test_model(test_loader , device='cuda' , log_tensorboard=True, writer= writer)