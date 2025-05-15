#%% Imports
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data import DataPross, DataProvider
from model import transformer

#%% Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_dir = os.path.join("runs", "Transformer_Finance_Experiment")

#%% Load and preprocess data
data = DataPross.Data('data/EURUSD_Candlestick_1_M_BID_04.05.2023-03.05.2025.csv')
data.clean()
data.add_indicators()

data.normalize()
df = data.df.drop(columns=['Volume' , 'Gmt time'])  # Drop 'Volume' column

#%% Split data
train_df, val_df = train_test_split(df, test_size=0.3, shuffle=False)
val_df, test_df = train_test_split(val_df, test_size=0.5, shuffle=False)

#%% Create datasets
sequence_length = 30
forecast_horizon = 10
target_cols = 'Close'
threshold = 0.005


train_dataset = DataProvider.TrendPredictionDataset(train_df, sequence_length, forecast_horizon, target_cols , threshold)
val_dataset   = DataProvider.TrendPredictionDataset(val_df, sequence_length, forecast_horizon, target_cols, threshold)
test_dataset  = DataProvider.TrendPredictionDataset(test_df, sequence_length, forecast_horizon, target_cols, threshold)

#%% Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#%% Initialize model


model = transformer.TransformerTimeSeriesModel(
    input_size=13,
    model_dim=64,
    num_heads=8,
    num_layers=3,
    dropout=0.3,
    output_size=1,
    task= 'classification',
    num_classes=3
)

#%% Train model
model.train_model(train_loader, val_loader, lr=1e-4, epochs=10, device=device)

#%% Test model
criterion = nn.MSELoss()
writer = SummaryWriter(log_dir=log_dir)

model.test_model(test_loader, device=device, log_tensorboard=True, writer=writer)

