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





from torch.utils.data import Dataset
import torch

class TransformerFinanceDataset(Dataset):
    def __init__(self, data, sequence_length=30, forecast_horizon=1, target_cols=['Close']):

        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.target_cols = target_cols

        self.data = data.copy()
        self.data.drop(columns=['date'], inplace=True, errors='ignore')

        self.features = self.data.drop(columns=target_cols).values
        self.targets = self.data[target_cols].values  # چند ستون تارگت پشتیبانی می‌کنیم

        self.X, self.y = self._create_sequences()

    def _create_sequences(self):
        X, y = [], []
        max_index = len(self.features) - self.sequence_length - self.forecast_horizon + 1
        for i in range(max_index):
            seq_x = self.features[i : i + self.sequence_length]
            seq_y = self.targets[i + self.sequence_length + self.forecast_horizon - 1]
            X.append(seq_x)
            y.append(seq_y)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#%% md
# class for forcast
# x =  n days , y = m day for each batch
#%%

#%%
import torch
from torch.utils.data import Dataset

class TrendPredictionDataset(Dataset):
    def __init__(self, data, sequence_length=30, forecast_horizon=5,
                 target_col='Close', threshold=0.01):  # threshold به صورت درصد مثلاً 0.01 یعنی 1٪
        """
        data: دیتافریم ورودی
        sequence_length: تعداد روزهای ورودی
        forecast_horizon: چند روز آینده را بررسی کنیم
        target_col: ستونی که می‌خواهیم روند آن را پیش‌بینی کنیم
        threshold: حد تغییر برای تعریف روند صعودی یا نزولی
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.threshold = threshold
        self.target_col = target_col

        self.data = data.copy()
        self.data.drop(columns=['date'], inplace=True, errors='ignore')

        self.features = self.data.drop(columns=[target_col]).values
        self.targets = self.data[target_col].values

        self.X, self.y = self._create_sequences()

    def _create_sequences(self):
        X, y = [], []
        max_index = len(self.features) - self.sequence_length - self.forecast_horizon
        for i in range(max_index):
            seq_x = self.features[i: i + self.sequence_length]
            current_price = self.targets[i + self.sequence_length - 1]
            future_price = self.targets[i + self.sequence_length + self.forecast_horizon - 1]

            # درصد تغییر
            change = (future_price - current_price) / current_price

            if change > self.threshold:
                trend = 1
            elif change < -self.threshold:
                trend = -1
            else:
                trend = 0

            X.append(seq_x)
            y.append(trend)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
