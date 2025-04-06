#%%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler


#%%
class Data():
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)

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
