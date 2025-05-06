
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from data import DataPross ,DataProvider
from model import transformer
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
#%%

model = transformer.TransformerTimeSeriesModel(input_size=4, model_dim=64, num_heads=4, num_layers=2, dropout=0.1, output_size=1)

model.load_state_dict(torch.load("/home/rango/DataspellProjects/untitled/theseis/runs/transformer/best_transformer_model.pth"))

model.eval()

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







#%%
all_preds = []
all_closes = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        preds = model(x).squeeze().cpu().numpy()
        closes = y.squeeze().cpu().numpy()

        all_preds.extend(preds)
        all_closes.extend(closes)
