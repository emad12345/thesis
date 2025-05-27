import pandas as pd
from tensorboard.data.provider import DataProvider

from data import DataPross, DataProvider

# Load your financial data
df = DataPross.Data('data/EURUSD_Candlestick_1_M_BID_04.05.2023-03.05.2025.csv')
df.clean()
df.normalize()

# ddf = df.df.drop(columns=['Gmt time'])
# #
# # Create the dataset with triple hurdle method
# dataset = DataProvider.TripleHurdleDataset(ddf,
#                              price_threshold=0.0005,  # 0.05% instead of 1%
#                              volatility_threshold=0.0001,  # Much lower
#                              volume_threshold=0.02)  # 2% instead of 10%
# # View trend statistics
# stats = dataset.get_trend_statistics()
# print(f"Trend distribution: {stats['percentages']}")
#
# # Visualize the labeled trends
# fig = dataset.visualize_labeled_trends()
# fig.show()

