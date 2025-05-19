#%%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange


#%%
class Data():
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path).loc[:500000]
    def read_data(self):
        pass

    def addfeature(self):
        pass

    def clean(self):
        # Remove rows with missing or zero volume (e.g. weekends or holidays)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        # self.df.index = self.df['date']
        return self.df


    def anomaly_detection(self, z_thresh=3):
        self.df['z_score'] = zscore(self.df['daily_return'].fillna(0))
        self.df['anomaly'] = abs(self.df['z_score']) > z_thresh


    def normalize(self):
        if self.df is not None:
            scaler = MinMaxScaler()
            columns_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume']
            self.df[columns_to_normalize] = scaler.fit_transform(self.df[columns_to_normalize])


    def visualize(self):
        if self.df is None:
            print("Run read_data() first.")
            return

        fig = go.Figure()

        # Main price line
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=self.df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue')
        ))

        # Anomaly points
        if 'anomaly' in self.df.columns and self.df['anomaly'].any():
            anomalies = self.df[self.df['anomaly']]
            fig.add_trace(go.Scatter(
                x=anomalies['date'],
                y=anomalies['Close'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8, symbol='cross')
            ))

        fig.update_layout(
            title='Close Price Over Time (with Anomalies)',
            xaxis_title='Date',
            yaxis_title='Close Price',
            template='plotly_dark',
            hovermode='x unified',
            height=500
        )

        fig.show()

    def add_indicators(self):
        if self.df is None:
            print("Data not loaded.")
            return

        # Ensure required columns exist
        required_cols = ['Close', 'High', 'Low']
        for col in required_cols:
            if col not in self.df.columns:
                print(f"Missing column: {col}")
                return

        # Simple Moving Averages
        self.df['SMA_20'] = SMAIndicator(close=self.df['Close'], window=20).sma_indicator()
        self.df['SMA_50'] = SMAIndicator(close=self.df['Close'], window=50).sma_indicator()

        # Exponential Moving Average
        self.df['EMA_20'] = EMAIndicator(close=self.df['Close'], window=20).ema_indicator()

        # RSI
        self.df['RSI'] = RSIIndicator(close=self.df['Close'], window=14).rsi()

        # MACD
        macd = MACD(close=self.df['Close'], window_slow=26, window_fast=12, window_sign=9)
        self.df['MACD'] = macd.macd()
        self.df['MACD_Signal'] = macd.macd_signal()

        # Bollinger Bands
        bb = BollingerBands(close=self.df['Close'], window=20, window_dev=2)
        self.df['BB_upper'] = bb.bollinger_hband()
        self.df['BB_middle'] = bb.bollinger_mavg()
        self.df['BB_lower'] = bb.bollinger_lband()

        # Average True Range
        atr = AverageTrueRange(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], window=14)
        self.df['ATR'] = atr.average_true_range()
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)


        # Print number of features
        print(f"📊 تعداد کل ویژگی‌ها پس از افزودن اندیکاتورها: {self.df.shape[1]}")


# #%%
# data = Data('EURUSDDayli.csv')
# data.read_data()
# data.clean()
#
# #%%
# data.visualize()
# #%%
# data.normalize()
# #%%
# data
#
# #%%
# data.visualize()
#%%
# data = Data('EURUSDDayli.csv')
#
# print(data.df)
# daata = data.df
# import matplotlib.pyplot as plt
# plt.plot(daata['Close'])
# plt.title('Closing Prices')
# plt.show()
#
# # ساده ترین راه تشخیص آماری
#
# print('[[[[[[[[[[[[[[[[[[[[[[[[[[')
# from scipy import stats
# z_scores = np.abs(stats.zscore(daata['Close']))
# outliers = np.where(z_scores > 3)
# print(outliers)
#
#
# time_deltas = daata['date'].diff()
# print(time_deltas.value_counts())
#

