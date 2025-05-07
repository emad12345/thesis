#%% Imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import DataPross, DataProvider
from model import transformer

#%% Load and preprocess data
data = DataPross.Data('data/EURUSDDayli.csv')
data.clean()
data.normalize()

df = data.df

#%% Split data
train_df, val_df = train_test_split(df, test_size=0.3, shuffle=False)
val_df, test_df = train_test_split(val_df, test_size=0.5, shuffle=False)

#%% Create datasets
sequence_length = 30
forecast_horizon = 1
target_cols = ['Close']

train_dataset = DataProvider.TransformerFinanceDataset(train_df, sequence_length, forecast_horizon, target_cols)
val_dataset   = DataProvider.TransformerFinanceDataset(val_df, sequence_length, forecast_horizon, target_cols)
test_dataset  = DataProvider.TransformerFinanceDataset(test_df, sequence_length, forecast_horizon, target_cols)

#%% Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#%% Initialize model
model = transformer.TransformerTimeSeriesModel(
    input_size=4,
    model_dim=128,
    num_heads=8,
    num_layers=4,
    dropout=0.2,
    output_size=1,
    task='regression'
)

#%% Train model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.train_model(train_loader, val_loader, lr=1e-3, epochs=100, device=device)

#%% Test model
criterion = nn.MSELoss()
writer = SummaryWriter(log_dir=os.path.join("runs", "Transformer_Finance_Experiment"))

model.test_model(test_loader, device=device, log_tensorboard=True, writer=writer)
