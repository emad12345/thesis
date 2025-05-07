#%% md
# class for classification
# x = n days , y = up , down , no change for each batch
# 
# 
#%%
import torch
from sympy.physics.units import volume
from torch.utils.data import Dataset
from vectorbt.generic.plotting import Volume
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np
import plotly.graph_objects as go

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

class TrendPredictionDataset(Dataset):
    def __init__(self, data, sequence_length=30, forecast_horizon=5,
                 target_col='Close', threshold=0.01):
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

            change = (future_price - current_price) / current_price

            if change > self.threshold:
                trend = 2
            elif change < -self.threshold:
                trend = 0
            else:
                trend = 1

            X.append(seq_x)
            y.append(trend)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def visualize_labeled_trends(self, price_series, title="Labeled Trends on Price"):
        trend_labels = self.y.numpy()
        x = np.arange(self.sequence_length, self.sequence_length + len(trend_labels))
        price_series = np.array(price_series)

        # نمودار قیمت اصلی
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(len(price_series)),
            y=price_series,
            mode='lines',
            name='Price',
            line=dict(color='black', width=2)
        ))

        # نقاط روند
        colors = {0: 'red', 1: 'blue', 2: 'green'}
        symbols = {0: 'triangle-down', 1: 'circle', 2: 'triangle-up'}
        labels = {0: 'Down', 1: 'Neutral', 2: 'Up'}

        for trend in [0, 1, 2]:
            idxs = x[trend_labels == trend]
            fig.add_trace(go.Scatter(
                x=idxs,
                y=price_series[idxs],
                mode='markers',
                name=labels[trend],
                marker=dict(color=colors[trend], symbol=symbols[trend], size=10),
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Time Step',
            yaxis_title='Price',
            legend_title='Trend',
            template='plotly_white',
            height=600,
            width=1000
        )

        fig.show()
#%% md
# class for forcast
# x =  n days , y = m day for each batch
#%%

#%%



