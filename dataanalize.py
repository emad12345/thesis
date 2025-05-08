#data analize
#%%
from data import DataPross ,DataProvider
from model import lstm
import numpy as np
import torch




df = DataPross.Data('data/EURUSDDayli.csv')

df.clean()
df.normalize()
# df.visualize()
ddf = df.df


Train = DataProvider.TrendPredictionDataset(ddf, sequence_length=30, forecast_horizon=10,
                 target_col='Close', threshold=0.007)
# print(len(Train.y))
# #
# close_prices = ddf['Close'].values
#
# Train.visualize_labeled_trends(close_prices)
#
#
# import numpy as np
# labels = Train.y.numpy()
# print(labels)
# unique, counts = np.unique(labels, return_counts=True)
#
# for label, count in zip(unique, counts):
#     print(f"Class {int(label)}: {count} samples")



import matplotlib.pyplot as plt
import numpy as np

def sanity_check_samples(dataset, price_series, num_samples=10):
    """
    dataset: آبجکت Dataset شما (مثلاً TrendPredictionDataset)
    price_series: آرایه قیمت اصلی (مثلاً df['Close'].values)
    num_samples: تعداد نمونه‌هایی که نمایش داده می‌شن
    """
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

        # موقعیت حال و آینده
        current_price = prices_to_plot[sequence_length - 1]
        future_price = prices_to_plot[-1]

        plt.scatter(x_axis[sequence_length - 1], current_price, color='orange', label='Current', zorder=5)
        plt.scatter(x_axis[-1], future_price,
                    color='green' if label == 2 else 'red' if label == 0 else 'blue',
                    label=f"Future (Label={label})", zorder=5)

        plt.title(f"Sample #{i} - Label: {label}")
        plt.xlabel("Time Step")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
# price_series = ddf['Close'].values
# sanity_check_samples(dataset=Train, price_series=price_series, num_samples=20)
