#%%
from data import DataPross ,DataProvider
from model import lstm
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
# data.visualize()
#%%
dataa = data.df
#%%



train_df, val_df = train_test_split (dataa, test_size=0.2, shuffle=False)

train_dataset = DataProvider.FinanceDataset(train_df, sequence_length=20)
val_dataset = DataProvider.FinanceDataset(val_df, sequence_length=20)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


model = lstm.LSTMModel(input_size=3 , hidden_size=64 , num_layers=2, dropout=0.2)

model.train_model( train_loader, val_loader, lr=1e-3, epochs=50, device='cuda')



