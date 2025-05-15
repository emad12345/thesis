# %% Imports
import numpy as np
import torch
from collections import Counter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data import DataPross, DataProvider
# from model import lstm  # فقط در صورت نیاز

# %% Load and preprocess data
df = DataPross.Data('data/EURUSD_Candlestick_1_M_BID_04.05.2023-03.05.2025.csv')
df.clean()
df.normalize()
ddf = df.df.drop(columns=['Gmt time'])

# %% Utility: Convert custom Dataset to numpy arrays
def dataset_to_numpy(dataset):
    X_list, y_list = [], []
    for X, y in DataLoader(dataset, batch_size=1):
        X_list.append(X.squeeze(0).numpy())
        y_list.append(y.item())
    return np.array(X_list), np.array(y_list)

# %% Threshold sweep: Analyze class distributions
# thresholds = np.linspace(0.001, 0.01, 20)
# for t in thresholds:
#     dataset = DataProvider.TrendPredictionDataset(
#         ddf,
#         sequence_length=30,
#         forecast_horizon=10,
#         target_col='Close',
#         threshold=t
#     )
#     _, y = dataset_to_numpy(dataset)
#     counts = Counter(y)
#     print(f"Threshold = {t:.4f} → Class counts: {counts}")


# %% Example: Create a specific dataset for visual inspection
Train = DataProvider.TrendPredictionDataset(
    ddf,
    sequence_length=30,
    forecast_horizon=10,
    target_col='Close',
    threshold=0.0038
)

print(f"Train samples: {len(Train.y)}")

# %% Print label distribution
labels = Train.y.numpy()
unique, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique, counts):
    print(f"Class {int(label)}: {count} samples")
exit()

# %% Visualization function: sanity check labeled trends
def sanity_check_samples(dataset, price_series, num_samples=10):
    sequence_length = dataset.sequence_length
    forecast_horizon = dataset.forecast_horizon

    for i in range(num_samples):
        x, label = dataset[i]
        start_idx = i
        end_idx = i + sequence_length + forecast_horizon
        prices_to_plot = price_series[start_idx:end_idx]
        x_axis = np.arange(start_idx, end_idx)

        plt.figure(figsize=(8, 3))
        plt.plot(x_axis, prices_to_plot, color='black', linewidth=2, label='Price')

        current_price = prices_to_plot[sequence_length - 1]
        future_price = prices_to_plot[-1]

        plt.scatter(x_axis[sequence_length - 1], current_price, color='orange', label='Current', zorder=5)
        plt.scatter(
            x_axis[-1],
            future_price,
            color='green' if label == 2 else 'red' if label == 0 else 'blue',
            label=f"Future (Label={label})",
            zorder=5
        )

        plt.title(f"Sample #{i} - Label: {label}")
        plt.xlabel("Time Step")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# %% Example usage (uncomment if needed)
# price_series = ddf['Close'].values
# sanity_check_samples(dataset=Train, price_series=price_series, num_samples=20)
