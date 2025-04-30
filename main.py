#%%
import os
print(os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os



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
#data.visualize()
#%%
dataa = data.df
#%%



train_df, val_df = train_test_split (dataa, test_size=0.3, shuffle=False)

val_df , test_df = train_test_split(val_df, test_size=0.5, shuffle=False)
len(test_df)

train_dataset = DataProvider.FinanceDataset(train_df, sequence_length=20)
#%%
test_dataset = DataProvider.FinanceDataset(test_df, sequence_length=20)
val_dataset = DataProvider.FinanceDataset(val_df, sequence_length=20)
#%%
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

val_loader = DataLoader(val_dataset, batch_size=64)

model = lstm.LSTMModel(input_size=4 , hidden_size=64 , num_layers=2, dropout=0.2)
#%%

model.train_model( train_loader, val_loader, lr=1e-3, epochs=50, device='cuda')



#%%
criterion = nn.MSELoss()
writer = SummaryWriter(log_dir=os.path.join("runs", "LSTM_Finance_Experiment"))

model.test_model(test_loader , device='cuda' , log_tensorboard=True, writer= writer)

#%%
