#%% md
# class for classification
# x = n days , y = up , down , no change for each batch
# 
# 
#%%
import torch
from torch.utils.data import Dataset

class FinanceDataset(Dataset):
    def __init__(self, data, sequence_length=20, target_col='Close'):
        self.sequence_length = sequence_length
        self.target_col = target_col

        self.data = data
        self.data.drop(columns=['date'] , inplace=True)
        self.features = data.drop(columns=[target_col]).values
        self.targets = data[target_col].values

        self.X, self.y = self.create_sequences()

    def create_sequences(self):
        X, y = [], []
        for i in range(len(self.data) - self.sequence_length):
            seq_x = self.features[i:i+self.sequence_length]
            seq_y = self.targets[i+self.sequence_length]
            X.append(seq_x)
            y.append(seq_y)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#%% md
# class for forcast
# x =  n days , y = m day for each batch
#%%

#%% md
# 