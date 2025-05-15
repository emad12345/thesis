import numpy as np
from pypots.classification.timesnet import TimesNet
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
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight
from pypots.classification.timesnet import TimesNet
from sklearn.metrics import classification_report





data = DataPross.Data('data/EURUSD_Candlestick_1_M_BID_04.05.2023-03.05.2025.csv')
data.clean()
print(len(data.df))

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

#%%

def dataset_to_numpy(dataset):
    X_list, y_list = [], []
    for X, y in DataLoader(dataset, batch_size=1):
        X_list.append(X.squeeze(0).numpy())
        y_list.append(y.item())
    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    return X_arr, y_arr

# تبدیل train, val, test به numpy
X_train, y_train = dataset_to_numpy(train_dataset)
X_val, y_val     = dataset_to_numpy(val_dataset)
X_test, y_test   = dataset_to_numpy(test_dataset)

# محاسبه وزن کلاس‌ها




# بررسی ابعاد
print("Train shape:", X_train.shape)  # مثلا (1000, 30, 8)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)

#%% آماده‌سازی داده‌ها برای TimesNet
train_data = {"X": X_train, "y": y_train}
val_data   = {"X": X_val, "y": y_val}
test_data  = {"X": X_test}


#%%
import numpy as np
from collections import Counter

print("Train class distribution:", Counter(y_train))
print("Val class distribution:", Counter(y_val))
print("Test class distribution:", Counter(y_test))

#%% تعریف مدل TimesNet
n_steps = X_train.shape[1]        # تعداد تایم‌استپ‌ها، مثلا 30
n_features = X_train.shape[2]    # تعداد ویژگی‌ها، مثلا 8
n_classes = len(np.unique(y_train))  # تعداد کلاس‌ها، مثلا 3 (down, neutral, up)

model = TimesNet(
    n_steps=n_steps,
    n_features=n_features,
    n_classes=n_classes,
    n_layers=2,
    top_k=5,
    d_model=64,
    d_ffn=128,
    n_kernels=6,
    dropout=0.1,
    batch_size=64,
    epochs=50,
    patience=5,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True
)

#%% آموزش مدل
model.fit(train_data, val_set=val_data)

#%% پیش‌بینی روی داده‌ی تست
results = model.predict(test_data)
y_pred = results["classification"]

#%% ارزیابی مدل
from sklearn.metrics import classification_report, accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

print("\nClassification Report:\n", classification_report(y_test, y_pred))
probs = model.predict_proba(test_data)
print("Probabilities for first test sample:", probs[0])
