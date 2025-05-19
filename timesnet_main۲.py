import os
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt

from data import DataPross, DataProvider
from pypots.classification.timesnet import TimesNet
from pypots.nn.modules.loss import Criterion

# Weighted Loss for Imbalanced Classes
class WeightedCrossEntropyLoss(Criterion):
    def __init__(self, weight_tensor):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    def forward(self, input, target):
        return self.loss_fn(input, target)

# Save utility
def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

def dataset_to_numpy(dataset):
    X, y = [], []
    for x_, y_ in DataLoader(dataset, batch_size=1):
        X.append(x_.squeeze(0).numpy())
        y.append(y_.item())
    return np.array(X), np.array(y)

# ----- Configuration -----
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f"results/timesnet_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# ----- Data Load -----
data = DataPross.Data('data/EURUSD_Candlestick_1_M_BID_04.05.2023-03.05.2025.csv')
data.clean()
data.normalize()
data.add_indicators()
df = data.df.drop(columns=['Volume', 'Gmt time'])

train_df, val_df = train_test_split(df, test_size=0.3, shuffle=False)
val_df, test_df = train_test_split(val_df, test_size=0.5, shuffle=False)

sequence_length = 30
forecast_horizon = 10
threshold = 0.0038
target_col = "Close"

train_ds = DataProvider.TrendPredictionDataset(train_df, sequence_length, forecast_horizon, target_col, threshold)
val_ds   = DataProvider.TrendPredictionDataset(val_df, sequence_length, forecast_horizon, target_col, threshold)
test_ds  = DataProvider.TrendPredictionDataset(test_df, sequence_length, forecast_horizon, target_col, threshold)

X_train, y_train = dataset_to_numpy(train_ds)
X_val, y_val     = dataset_to_numpy(val_ds)
X_test, y_test   = dataset_to_numpy(test_ds)

# Log class distribution
from collections import Counter
for name, arr in [("train", y_train), ("val", y_val), ("test", y_test)]:
    dist = Counter(arr)
    for k, v in dist.items():
        writer.add_scalar(f"class_dist/{name}_class_{k}", v)

# ----- Class Weights -----
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = WeightedCrossEntropyLoss(class_weights_tensor)

# ----- Model -----
model = TimesNet(
    n_steps=X_train.shape[1],
    n_features=X_train.shape[2],
    n_classes=len(np.unique(y_train)),
    n_layers=2,
    top_k=5,
    d_model=64,
    d_ffn=128,
    n_kernels=6,
    dropout=0.5,
    batch_size=32,
    epochs=50,
    patience=10,
    training_loss=loss_fn,
    validation_metric=loss_fn,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    saving_path=os.path.join(log_dir, "model.pypots"),
    model_saving_strategy="best",
    verbose=True
)

# ----- Train -----
model.fit({"X": X_train, "y": y_train}, val_set={"X": X_val, "y": y_val})

# ----- Predict -----
results = model.predict({"X": X_test})
y_pred = results["classification"]
probs = model.predict_proba({"X": X_test})

# ----- Evaluation -----
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4, output_dict=True)
writer.add_scalar("test/accuracy", acc)

# Save classification report
save_json(report, os.path.join(log_dir, "classification_report.json"))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(log_dir, "confusion_matrix.png"))
writer.add_figure("ConfusionMatrix", plt.gcf())

# Save probabilities of first test sample
np.save(os.path.join(log_dir, "sample_probs.npy"), probs[0])

# Save final predictions
np.save(os.path.join(log_dir, "predictions.npy"), y_pred)
# ----- Save metadata (hyperparameters + class distributions) -----
metadata = {
    "hyperparameters": {
        "n_steps": X_train.shape[1],
        "n_features": X_train.shape[2],
        "n_classes": len(np.unique(y_train)),
        "n_layers": model.n_layers,
        "top_k": model.top_k,
        "d_model": model.d_model,
        "d_ffn": model.d_ffn,
        "n_kernels": model.n_kernels,
        "dropout": model.dropout,
        "batch_size": model.batch_size,
        "epochs": model.epochs,
        "patience": model.patience,
    },
    "class_distribution": {
        "train": dict(Counter(y_train)),
        "val": dict(Counter(y_val)),
        "test": dict(Counter(y_test)),
    }
}

save_json(metadata, os.path.join(log_dir, "metadata.json"))

writer.close()
