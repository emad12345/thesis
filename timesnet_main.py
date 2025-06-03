# ===== Imports =====
import os
import json
import datetime
import gc
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorboard.data.provider import DataProvider
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from data import DataPross, DataProvider
from pypots.classification.timesnet import TimesNet
from pypots.nn.modules.loss import Criterion
from utilits import io_utils , loss , data_utils


# ===== Configuration =====
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f"results/timesnet_{timestamp}"
model_dir = os.path.join(log_dir, "model.pypots")
tensorboard_dir = os.path.join(model_dir, "tensorboard")
os.makedirs(tensorboard_dir, exist_ok=True)

writer = SummaryWriter(log_dir)

# ===== Data Load & Processing =====
# data = DataPross.Data('data/EURUSD_Candlestick_1_M_BID_04.05.2023-03.05.2025.csv')
# data.clean()
# data.normalize()
# data.add_indicators()
# df = data.df.drop(columns=['Volume', 'Gmt time'])
#
# train_df, val_df = train_test_split(df, test_size=0.3, shuffle=False)
# val_df, test_df = train_test_split(val_df, test_size=0.5, shuffle=False)
#
# sequence_length = 30
# forecast_horizon = 10
# threshold = 0.0038
# target_col = "Close"
#
#
#
# # Define dataset
# train_ds = DataProvider.TrendPredictionDataset(
#     train_df,
#     sequence_length=sequence_length,
#     forecast_horizon=forecast_horizon,
#     threshold=threshold,
#     target_col=target_col
# )
#
# val_ds = DataProvider.TrendPredictionDataset(
#     val_df,
#     sequence_length=sequence_length,
#     forecast_horizon=forecast_horizon,
#     threshold=threshold,
#     target_col=target_col
# )
#
# test_ds = DataProvider.TrendPredictionDataset(
#     test_df,
#     sequence_length=sequence_length,
#     forecast_horizon=forecast_horizon,
#     threshold=threshold,
#     target_col=target_col
# )
#
# # Convert datasets to numpy arrays
# X_train, y_train = data_utils.dataset_to_numpy(train_ds)
# X_val, y_val = data_utils.dataset_to_numpy(val_ds)
# X_test, y_test = data_utils.dataset_to_numpy(test_ds)




save_path = "saved_data"
data = np.load(os.path.join(save_path, 'timesnet_data.npz'))
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val     = data['X_val'],   data['y_val']
X_test, y_test   = data['X_test'],  data['y_test']

balancer = DataProvider.BalancedDatasetBuilderSmartUndersampling(reduction_ratio=1)
X_train, y_train = balancer.balance(X_train, y_train)

# ===== Class Distribution Logging =====
for name, arr in [("train", y_train), ("val", y_val), ("test", y_test)]:
    dist = Counter(arr)
    for k, v in dist.items():
        writer.add_scalar(f"class_dist/{name}_class_{k}", v)

# ===== Class Weights & Loss Function =====
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
print("Class Weights:", class_weights)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
loss_fn = loss.WeightedCrossEntropyLoss(weight_tensor)

# ===== Model Definition =====
model = TimesNet(
    n_steps=X_train.shape[1],
    n_features=X_train.shape[2],
    n_classes=len(np.unique(y_train)),
    n_layers=2,
    top_k=5,
    d_model=64,
    d_ffn=128,
    n_kernels=6,
    dropout=0.3,
    batch_size=512,
    epochs=100,
    patience=10,
    # training_loss=loss_fn,
    # validation_metric=loss_fn,
    device=device,
    saving_path=model_dir,
    model_saving_strategy="best",
    verbose=True
)

# ===== Training =====
model.fit({"X": X_train, "y": y_train}, val_set={"X": X_val, "y": y_val})
gc.collect()
torch.cuda.empty_cache()
print('✅ Training complete.')

# ===== Prediction =====
results = model.predict({"X": X_test})
y_pred = results["classification"]
probs = model.predict_proba({"X": X_test})

gc.collect()
torch.cuda.empty_cache()

# ===== Evaluation =====
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4, output_dict=True)
writer.add_scalar("test/accuracy", acc)

# Save classification report
io_utils.save_json(report, os.path.join(log_dir, "classification_report.json"))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(log_dir, "confusion_matrix.png"))
writer.add_figure("ConfusionMatrix", plt.gcf())

# Save Outputs
np.save(os.path.join(log_dir, "sample_probs.npy"), probs[0])
np.save(os.path.join(log_dir, "predictions.npy"), y_pred)

# Save Metadata
def fix_keys(d):
    return {int(k): v for k, v in d.items()}
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
        "train": fix_keys(Counter(y_train)),
        "val": fix_keys(Counter(y_val)),
        "test": fix_keys(Counter(y_test)),
    }
}
io_utils.save_json(metadata, os.path.join(log_dir, "metadata.json"))

# Finalize
writer.flush()
writer.close()
