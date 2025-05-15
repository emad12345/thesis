import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data import DataPross, DataProvider
from pypots.classification.timesnet import TimesNet
from pypots.nn.modules.loss import Criterion

# تعریف کلاس loss سازگار با PyPOTS
class WeightedCrossEntropyLoss(Criterion):
    def __init__(self, weight_tensor):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    def forward(self, input, target):
        return self.loss_fn(input, target)

# ----- Load and prepare data -----
data = DataPross.Data('data/EURUSD_Candlestick_1_M_BID_04.05.2023-03.05.2025.csv')
data.clean()
data.normalize()
data.add_indicators()
df = data.df.drop(columns=['Volume', 'Gmt time'])

# Split
train_df, val_df = train_test_split(df, test_size=0.3, shuffle=False)
val_df, test_df = train_test_split(val_df, test_size=0.5, shuffle=False)

# Datasets
sequence_length = 30
forecast_horizon = 10
target_cols = 'Close'
threshold = 0.0038  # مقدار مناسب انتخاب‌شده

train_dataset = DataProvider.TrendPredictionDataset(train_df, sequence_length, forecast_horizon, target_cols, threshold)
val_dataset   = DataProvider.TrendPredictionDataset(val_df, sequence_length, forecast_horizon, target_cols, threshold)
test_dataset  = DataProvider.TrendPredictionDataset(test_df, sequence_length, forecast_horizon, target_cols, threshold)

# تبدیل دیتاست‌ها به numpy
def dataset_to_numpy(dataset):
    X_list, y_list = [], []
    for X, y in DataLoader(dataset, batch_size=1):
        X_list.append(X.squeeze(0).numpy())
        y_list.append(y.item())
    return np.array(X_list), np.array(y_list)

X_train, y_train = dataset_to_numpy(train_dataset)
X_val, y_val     = dataset_to_numpy(val_dataset)
X_test, y_test   = dataset_to_numpy(test_dataset)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)

# نمایش توزیع کلاس‌ها
print("Train class distribution:", Counter(y_train))
print("Val class distribution:", Counter(y_val))
print("Test class distribution:", Counter(y_test))


# آماده‌سازی داده برای مدل
train_data = {"X": X_train, "y": y_train}
val_data   = {"X": X_val, "y": y_val}
test_data  = {"X": X_test}

# ----- محاسبه وزن کلاس‌ها -----
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = WeightedCrossEntropyLoss(class_weights_tensor)

# ----- تعریف مدل -----
n_steps = X_train.shape[1]
n_features = X_train.shape[2]
n_classes = len(np.unique(y_train))

model = TimesNet(
    n_steps=n_steps,
    n_features=n_features,
    n_classes=n_classes,
    n_layers=2,
    top_k=5,
    d_model=64,
    d_ffn=128,
    n_kernels=6,
    dropout=0.5,
    batch_size=64,
    epochs=50,
    patience=40,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    training_loss=loss_fn,
    validation_metric=loss_fn,
    verbose=True
)

# ----- آموزش -----
model.fit(train_data, val_set=val_data)

# ----- پیش‌بینی -----
results = model.predict(test_data)
y_pred = results["classification"]

# ----- ارزیابی -----
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# نمایش احتمال اولین نمونه
probs = model.predict_proba(test_data)
print("Probabilities for first test sample:", probs[0])
