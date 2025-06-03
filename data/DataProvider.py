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

            # change = (future_price - current_price) / current_price
            if current_price == 0:
                change = 0  # یا continue یا مقدار خاص
            else:
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
import torch
from torch.utils.data import Dataset
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TripleHurdleDataset(Dataset):
    """
    Dataset class for financial time series data with triple hurdle method for trend labeling.
    The triple hurdle method uses three criteria to determine trend direction:
    1. Price Change: Must exceed a threshold
    2. Volatility: Must confirm the trend direction
    3. Volume: Must support the trend (higher volume for stronger trends)
    """

    def __init__(self, data, sequence_length=30, forecast_horizon=5,
                 price_col='Close', volume_col='Volume',
                 price_threshold=0.01, volatility_threshold=0.005, volume_threshold=0.1):
        """
        Initialize the dataset with triple hurdle parameters.

        Args:
            data (DataFrame): Financial data including price and volume
            sequence_length (int): Length of input sequences
            forecast_horizon (int): How many steps ahead to predict
            price_col (str): Column name for price data
            volume_col (str): Column name for volume data
            price_threshold (float): Minimum percentage change to consider a trend
            volatility_threshold (float): Volatility threshold for confirmation
            volume_threshold (float): Volume change threshold for confirmation
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.price_threshold = price_threshold
        self.volatility_threshold = volatility_threshold
        self.volume_threshold = volume_threshold
        self.price_col = price_col
        self.volume_col = volume_col

        self.data = data.copy()
        self.data.drop(columns=['date'], inplace=True, errors='ignore')

        # Ensure required columns exist
        if volume_col not in self.data.columns:
            raise ValueError(f"Volume column '{volume_col}' not found in data")

        # Calculate additional features
        self._calculate_volatility()

        # Store feature and target data
        self.features = self.data.drop(columns=[price_col]).values
        self.prices = self.data[price_col].values
        self.volumes = self.data[volume_col].values
        self.volatility = self.data['volatility'].values

        # Create sequences
        self.X, self.y, self.hurdle_details = self._create_sequences()

    def _calculate_volatility(self, window=5):
        """Calculate rolling volatility."""
        self.data['volatility'] = self.data[self.price_col].pct_change().rolling(window).std()
        self.data['volatility'].fillna(0, inplace=True)

    def _create_sequences(self):
        """Create sequences with triple hurdle labeling."""
        X, y, hurdle_details = [], [], []

        max_index = len(self.features) - self.sequence_length - self.forecast_horizon

        for i in range(max_index):
            # Extract sequence features
            seq_x = self.features[i: i + self.sequence_length]

            # Current and future values
            current_price = self.prices[i + self.sequence_length - 1]
            future_price = self.prices[i + self.sequence_length + self.forecast_horizon - 1]

            # Future period values for additional hurdles
            future_vol = self.volatility[i + self.sequence_length: i + self.sequence_length + self.forecast_horizon]
            future_volume = self.volumes[i + self.sequence_length: i + self.sequence_length + self.forecast_horizon]
            current_volume = self.volumes[i + self.sequence_length - 5: i + self.sequence_length].mean()

            # Calculate metrics for each hurdle
            # Hurdle 1: Price Change
            if current_price > 0:
                price_change = (future_price - current_price) / current_price
            else:
                price_change = 0

            # Hurdle 2: Volatility
            avg_volatility = np.mean(future_vol)
            volatility_increasing = avg_volatility > self.volatility[i + self.sequence_length - 1]

            # Hurdle 3: Volume
            avg_future_volume = np.mean(future_volume)
            volume_change = (avg_future_volume - current_volume) / current_volume if current_volume > 0 else 0

            # Apply the triple hurdle method
            trend = self._apply_triple_hurdle(price_change, avg_volatility, volume_change, volatility_increasing)

            # Store details for visualization/analysis
            details = {
                'price_change': price_change,
                'volatility': avg_volatility,
                'volume_change': volume_change,
                'volatility_increasing': volatility_increasing
            }

            X.append(seq_x)
            y.append(trend)
            hurdle_details.append(details)

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.int64),
            hurdle_details
        )

    def _apply_triple_hurdle(self, price_change, volatility, volume_change, volatility_increasing):
        """
        Apply the triple hurdle method to determine trend.

        Returns:
            0: Downtrend
            1: Neutral
            2: Uptrend
        """
        # Preliminary trend based on price change
        if price_change > self.price_threshold:
            preliminary_trend = 2  # Up
        elif price_change < -self.price_threshold:
            preliminary_trend = 0  # Down
        else:
            return 1  # Neutral (fails first hurdle)

        # Hurdle 2: Volatility must confirm trend
        if preliminary_trend == 2:  # Uptrend
            if volatility < self.volatility_threshold and not volatility_increasing:
                return 1  # Fails second hurdle
        elif preliminary_trend == 0:  # Downtrend
            if volatility < self.volatility_threshold and volatility_increasing:
                return 1  # Fails second hurdle

        # Hurdle 3: Volume must support trend
        if preliminary_trend == 2:  # Uptrend
            if volume_change < self.volume_threshold:
                return 1  # Fails third hurdle
        elif preliminary_trend == 0:  # Downtrend
            if volume_change > -self.volume_threshold:
                return 1  # Fails third hurdle

        # All hurdles passed
        return preliminary_trend

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_hurdle_details(self, idx):
        """Get detailed information about the triple hurdle classification."""
        return self.hurdle_details[idx]

    def visualize_labeled_trends(self, price_series=None, volume_series=None, title="Triple Hurdle Labeled Trends"):
        """
        Visualize the labeled trends with all triple hurdle components.

        Args:
            price_series: Price data to plot (defaults to using target column data)
            volume_series: Volume data to plot
            title: Plot title
        """
        trend_labels = self.y.numpy()
        x = np.arange(self.sequence_length, self.sequence_length + len(trend_labels))

        if price_series is None:
            price_series = self.prices

        if volume_series is None and hasattr(self, 'volumes'):
            volume_series = self.volumes

        price_series = np.array(price_series)

        # Create subplots: price with trends, volume, volatility
        fig = make_subplots(rows=3, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=('Price with Trend Labels', 'Volume', 'Volatility'),
                            row_heights=[0.5, 0.25, 0.25])

        # Plot 1: Price with trend labels
        fig.add_trace(
            go.Scatter(x=np.arange(len(price_series)), y=price_series, mode='lines',
                       name='Price', line=dict(color='black', width=2)),
            row=1, col=1
        )

        # Add trend markers
        colors = {0: 'red', 1: 'gray', 2: 'green'}
        symbols = {0: 'triangle-down', 1: 'circle', 2: 'triangle-up'}
        labels = {0: 'Down', 1: 'Neutral', 2: 'Up'}

        for trend in [0, 1, 2]:
            idxs = x[trend_labels == trend]
            if len(idxs) > 0:  # Only add trace if there are points with this trend
                fig.add_trace(
                    go.Scatter(
                        x=idxs,
                        y=price_series[idxs],
                        mode='markers',
                        name=labels[trend],
                        marker=dict(color=colors[trend], symbol=symbols[trend], size=10),
                    ),
                    row=1, col=1
                )

        # Plot 2: Volume
        if volume_series is not None:
            fig.add_trace(
                go.Bar(x=np.arange(len(volume_series)), y=volume_series, name='Volume',
                       marker=dict(color='lightblue')),
                row=2, col=1
            )

        # Plot 3: Volatility
        if hasattr(self, 'volatility'):
            fig.add_trace(
                go.Scatter(x=np.arange(len(self.volatility)), y=self.volatility,
                           mode='lines', name='Volatility', line=dict(color='purple')),
                row=3, col=1
            )

            # Add volatility threshold line
            fig.add_trace(
                go.Scatter(x=[0, len(self.volatility)],
                           y=[self.volatility_threshold, self.volatility_threshold],
                           mode='lines', name='Volatility Threshold',
                           line=dict(color='purple', dash='dash')),
                row=3, col=1
            )

        fig.update_layout(
            title=title,
            xaxis3_title='Time Step',
            template='plotly_white',
            height=900,
            width=1000,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def get_trend_statistics(self):
        """Get statistics about the labeled trends."""
        trend_labels = self.y.numpy()

        # Count occurrences of each trend
        trend_counts = {
            'Down': np.sum(trend_labels == 0),
            'Neutral': np.sum(trend_labels == 1),
            'Up': np.sum(trend_labels == 2)
        }

        # Calculate percentages
        total = len(trend_labels)
        trend_percentages = {
            'Down': round(trend_counts['Down'] / total * 100, 2),
            'Neutral': round(trend_counts['Neutral'] / total * 100, 2),
            'Up': round(trend_counts['Up'] / total * 100, 2)
        }

        return {
            'counts': trend_counts,
            'percentages': trend_percentages,
            'total': total
        }

    def get_sequential_patterns(self, window=3):
        """Analyze sequential patterns in the labeled trends."""
        trend_labels = self.y.numpy()
        patterns = {}

        for i in range(len(trend_labels) - window + 1):
            pattern = tuple(trend_labels[i:i + window])
            patterns[pattern] = patterns.get(pattern, 0) + 1

        return sorted(patterns.items(), key=lambda x: x[1], reverse=True)

#%%
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from collections import defaultdict


class EnhancedTripleHurdleDataset(Dataset):
    """
    Advanced dataset class for financial time series with adaptive triple hurdle method.

    Features:
    - Adaptive thresholds based on market conditions
    - Market regime detection for context-aware labeling
    - Confidence scoring for trend predictions
    - Trading strategy simulation and backtesting
    - Sequential pattern analysis with profitability metrics
    - Cross-validation capabilities for label quality assessment
    """

    def __init__(self, data, sequence_length=30, forecast_horizon=5,
                 price_col='Close', high_col='High', low_col='Low', volume_col='Volume',
                 price_threshold=0.01, volatility_threshold=0.005, volume_threshold=0.1,
                 adaptive_thresholds=True, regime_detection=True, confidence_scoring=True,
                 hurdle_weights=(0.5, 0.3, 0.2)):
        """
        Initialize the enhanced dataset with multiple labeling options.

        Args:
            data (DataFrame): Financial data with price, volume, and optional columns
            sequence_length (int): Length of input sequences
            forecast_horizon (int): How many steps ahead to predict
            price_col (str): Column name for price data
            high_col (str): Column name for high price data
            low_col (str): Column name for low price data
            volume_col (str): Column name for volume data
            price_threshold (float): Base threshold for price changes
            volatility_threshold (float): Base threshold for volatility
            volume_threshold (float): Base threshold for volume changes
            adaptive_thresholds (bool): Whether to use adaptive thresholds
            regime_detection (bool): Whether to detect and use market regimes
            confidence_scoring (bool): Whether to add confidence scores to labels
            hurdle_weights (tuple): Weights for each hurdle (price, volatility, volume)
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.base_price_threshold = price_threshold
        self.base_volatility_threshold = volatility_threshold
        self.base_volume_threshold = volume_threshold
        self.price_col = price_col
        self.high_col = high_col
        self.low_col = low_col
        self.volume_col = volume_col
        self.adaptive_thresholds = adaptive_thresholds
        self.regime_detection = regime_detection
        self.confidence_scoring = confidence_scoring
        self.hurdle_weights = hurdle_weights

        # Validate and prepare data
        self.data = data.copy()
        self.data.drop(columns=['date'], inplace=True, errors='ignore')
        self._check_required_columns()

        # Calculate technical features and market context
        self._calculate_technical_features()
        if self.regime_detection:
            self._detect_market_regimes()
        if self.adaptive_thresholds:
            self._calculate_adaptive_thresholds()

        # Store feature and target data
        self.features = self.data.drop(columns=[price_col], errors='ignore').values
        self.prices = self.data[price_col].values
        self.volumes = self.data[volume_col].values if volume_col in self.data.columns else None

        # Create sequences with enhanced labeling
        self.X, self.y, self.confidences, self.hurdle_details, self.price_changes = self._create_sequences()

        # Log dataset statistics
        self._log_dataset_statistics()

    def _check_required_columns(self):
        """Verify that required columns exist in the dataset."""
        required_cols = [self.price_col]
        if self.regime_detection:
            required_cols.extend([self.high_col, self.low_col])

        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Handle missing volume column
        if self.volume_col not in self.data.columns:
            print(f"Warning: Volume column '{self.volume_col}' not found. Using price-based volume proxy.")
            self.data[self.volume_col] = np.abs(self.data[self.price_col].pct_change())
            self.volume_col_is_proxy = True
        else:
            self.volume_col_is_proxy = False

    def _calculate_technical_features(self):
        """Calculate technical indicators and features needed for labeling."""
        # Basic price-derived features
        self.data['return'] = self.data[self.price_col].pct_change()
        self.data['log_return'] = np.log(self.data[self.price_col]).diff()

        # Volatility measures
        window_sizes = [5, 14, 21]
        for window in window_sizes:
            self.data[f'volatility_{window}'] = self.data['return'].rolling(window).std()

        # Default volatility measure (14-day)
        self.data['volatility'] = self.data['volatility_14']

        # Volume features
        if not self.volume_col_is_proxy:
            self.data['volume_ma'] = self.data[self.volume_col].rolling(14).mean()
            self.data['relative_volume'] = self.data[self.volume_col] / self.data['volume_ma']

        # ATR for adaptive thresholds
        if self.high_col in self.data.columns and self.low_col in self.data.columns:
            atr_indicator = ta.volatility.AverageTrueRange(
                high=self.data[self.high_col],
                low=self.data[self.low_col],
                close=self.data[self.price_col],
                window=14
            )
            self.data['atr'] = atr_indicator.average_true_range()
            self.data['atr_pct'] = self.data['atr'] / self.data[self.price_col]

        # Fill NaN values
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(0, inplace=True)

    def _detect_market_regimes(self):
        """Detect market regimes (trending, ranging, volatile) for context-aware labeling."""
        # Calculate directional indicators
        if self.high_col in self.data.columns and self.low_col in self.data.columns:
            adx_indicator = ta.trend.ADXIndicator(
                high=self.data[self.high_col],
                low=self.data[self.low_col],
                close=self.data[self.price_col],
                window=14
            )
            self.data['adx'] = adx_indicator.adx()

            # Calculate price momentum
            self.data['rsi'] = ta.momentum.RSIIndicator(
                close=self.data[self.price_col], window=14
            ).rsi()

            # Define regimes based on ADX and volatility
            self.data['regime'] = np.where(
                self.data['adx'] > 25, 'trending',
                np.where(self.data['volatility'] > self.data['volatility'].rolling(30).mean() * 1.5,
                         'volatile', 'ranging')
            )

            # Define regime multipliers for thresholds
            self.regime_multipliers = {
                'trending': {'price': 0.8, 'vol': 1.2, 'volume': 1.0},
                'ranging': {'price': 1.2, 'vol': 0.8, 'volume': 1.2},
                'volatile': {'price': 1.5, 'vol': 0.7, 'volume': 1.3}
            }
        else:
            print("Warning: High/Low columns missing, using simplified regime detection")
            self.data['regime'] = np.where(
                self.data['volatility'] > self.data['volatility'].rolling(30).mean() * 1.5,
                'volatile', 'normal'
            )
            self.regime_multipliers = {
                'volatile': {'price': 1.5, 'vol': 0.7, 'volume': 1.3},
                'normal': {'price': 1.0, 'vol': 1.0, 'volume': 1.0}
            }

    def _calculate_adaptive_thresholds(self, lookback=50):
        """Calculate adaptive thresholds based on recent market conditions."""
        # Price threshold based on ATR
        if 'atr_pct' in self.data.columns:
            self.data['price_threshold'] = self.base_price_threshold * (
                    self.data['atr_pct'] / self.data['atr_pct'].rolling(lookback).mean()
            )
        else:
            # Fall back to volatility if ATR not available
            self.data['price_threshold'] = self.base_price_threshold * (
                    self.data['volatility'] / self.data['volatility'].rolling(lookback).mean()
            )

        # Volatility threshold
        self.data['volatility_threshold'] = self.base_volatility_threshold * (
                self.data['volatility'] / self.data['volatility'].rolling(lookback).mean()
        )

        # Volume threshold
        if not self.volume_col_is_proxy:
            vol_std = self.data[self.volume_col].rolling(lookback).std() / self.data['volume_ma']
            self.data['volume_threshold'] = self.base_volume_threshold * (
                    vol_std / vol_std.rolling(lookback).mean()
            )
        else:
            self.data['volume_threshold'] = self.base_volume_threshold

        # Ensure no NaN or extreme values
        threshold_cols = ['price_threshold', 'volatility_threshold', 'volume_threshold']
        for col in threshold_cols:
            # Replace NaNs
            self.data[col].fillna(getattr(self, f'base_{col}'), inplace=True)

            # Cap extreme values (2x base threshold)
            max_threshold = getattr(self, f'base_{col}') * 3
            min_threshold = getattr(self, f'base_{col}') * 0.5
            self.data[col] = np.clip(self.data[col], min_threshold, max_threshold)

    def _get_thresholds(self, idx):
        """Get the appropriate thresholds for a given index, considering adaptivity and regime."""
        if self.adaptive_thresholds:
            price_thresh = self.data.iloc[idx]['price_threshold']
            vol_thresh = self.data.iloc[idx]['volatility_threshold']
            vol_thresh = max(vol_thresh, 0.001)  # Ensure minimum volatility threshold

            if not self.volume_col_is_proxy:
                volume_thresh = self.data.iloc[idx][
                    'volume_threshold']  # ← اینجا باید volume_thresh باشه، نه vol_thresh
            else:
                volume_thresh = self.base_volume_threshold
        else:
            price_thresh = self.base_price_threshold
            vol_thresh = self.base_volatility_threshold
            volume_thresh = self.base_volume_threshold

        # Apply regime multipliers if regime detection is enabled
        if self.regime_detection and 'regime' in self.data.columns:
            regime = self.data.iloc[idx]['regime']
            multipliers = self.regime_multipliers.get(regime, {'price': 1.0, 'vol': 1.0, 'volume': 1.0})

            price_thresh *= multipliers['price']
            vol_thresh *= multipliers['vol']
            volume_thresh *= multipliers['volume']

        return price_thresh, vol_thresh, volume_thresh

    def _create_sequences(self):
        """Create sequences with enhanced triple hurdle labeling."""
        X, y, confidences, hurdle_details, price_changes = [], [], [], [], []

        max_index = len(self.features) - self.sequence_length - self.forecast_horizon

        for i in range(max_index):
            # Extract sequence features
            seq_x = self.features[i: i + self.sequence_length]

            # Current index for threshold and regime lookups
            current_idx = i + self.sequence_length - 1

            # Get current and future values
            current_price = self.prices[current_idx]
            future_price = self.prices[current_idx + self.forecast_horizon]

            # Future period values for additional hurdles
            future_vol = self.data['volatility'].iloc[current_idx:current_idx + self.forecast_horizon].mean()
            current_vol = self.data['volatility'].iloc[current_idx]

            # Volume metrics
            if not self.volume_col_is_proxy:
                future_volume = self.volumes[current_idx:current_idx + self.forecast_horizon].mean()
                current_volume = self.volumes[current_idx - 4:current_idx + 1].mean()
                if current_volume > 0:
                    volume_change = (future_volume - current_volume) / current_volume
                else:
                    volume_change = 0
            else:
                # Use price volatility as volume proxy
                volume_change = future_vol - current_vol

            # Calculate price change
            if current_price > 0:
                price_change = (future_price - current_price) / current_price
            else:
                price_change = 0

            # Get appropriate thresholds for this index
            price_threshold, vol_threshold, volume_threshold = self._get_thresholds(current_idx)

            # Check if volatility is increasing
            volatility_increasing = future_vol > current_vol

            # Apply the enhanced triple hurdle method
            if self.confidence_scoring:
                trend, confidence = self._apply_triple_hurdle_with_confidence(
                    price_change, future_vol, volume_change,
                    price_threshold, vol_threshold, volume_threshold,
                    volatility_increasing
                )
                confidences.append(confidence)
            else:
                trend = self._apply_triple_hurdle(
                    price_change, future_vol, volume_change,
                    price_threshold, vol_threshold, volume_threshold,
                    volatility_increasing
                )
                confidences.append(1.0)  # Default confidence

            # Store details for visualization/analysis
            details = {
                'price_change': price_change,
                'volatility': future_vol,
                'volume_change': volume_change,
                'volatility_increasing': volatility_increasing,
                'price_threshold': price_threshold,
                'vol_threshold': vol_threshold,
                'volume_threshold': volume_threshold,
                'regime': self.data.iloc[current_idx]['regime'] if 'regime' in self.data.columns else 'unknown'
            }

            X.append(seq_x)
            y.append(trend)
            hurdle_details.append(details)
            price_changes.append(price_change)

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.int64),
            torch.tensor(confidences, dtype=torch.float32),
            hurdle_details,
            np.array(price_changes)
        )

    def _apply_triple_hurdle(self, price_change, volatility, volume_change,
                             price_threshold, vol_threshold, volume_threshold,
                             volatility_increasing):
        """
        Apply the triple hurdle method with adaptive thresholds.

        Returns:
            0: Downtrend
            1: Neutral
            2: Uptrend
        """
        # Preliminary trend based on price change (Hurdle 1)
        if price_change > price_threshold:
            preliminary_trend = 2  # Up
        elif price_change < -price_threshold:
            preliminary_trend = 0  # Down
        else:
            return 1  # Neutral (fails first hurdle)

        # Hurdle 2: Volatility must confirm trend
        if preliminary_trend == 2:  # Uptrend
            if volatility < vol_threshold and not volatility_increasing:
                return 1  # Fails second hurdle
        elif preliminary_trend == 0:  # Downtrend
            if volatility < vol_threshold and volatility_increasing:
                return 1  # Fails second hurdle

        # Hurdle 3: Volume must support trend
        if preliminary_trend == 2:  # Uptrend
            if volume_change < volume_threshold:
                return 1  # Fails third hurdle
        elif preliminary_trend == 0:  # Downtrend
            if volume_change > -volume_threshold:
                return 1  # Fails third hurdle

        # All hurdles passed
        return preliminary_trend

    def _apply_triple_hurdle_with_confidence(self, price_change, volatility, volume_change,
                                             price_threshold, vol_threshold, volume_threshold,
                                             volatility_increasing):
        """
        Apply triple hurdle method with confidence scoring.

        Returns:
            (trend, confidence): Trend label (0,1,2) and confidence score (0-1)
        """
        # Calculate base confidence for each hurdle
        # Price hurdle confidence
        if abs(price_change) < price_threshold:
            price_conf = 0.0
            base_trend = 1  # Neutral
        else:
            price_conf = min(1.0, abs(price_change) / (price_threshold * 3))
            base_trend = 2 if price_change > 0 else 0

        # Volatility hurdle confidence
        vol_direction_match = (base_trend == 2 and volatility_increasing) or (
                    base_trend == 0 and not volatility_increasing)
        if volatility < vol_threshold:
            vol_conf = 0.0
        else:
            vol_conf = min(1.0, volatility / (vol_threshold * 3))
            # Increase confidence if volatility direction matches trend
            if vol_direction_match:
                vol_conf *= 1.2
                vol_conf = min(1.0, vol_conf)

        # Volume hurdle confidence
        if base_trend == 2:  # Uptrend
            volume_condition = volume_change > volume_threshold
        elif base_trend == 0:  # Downtrend
            volume_condition = volume_change < -volume_threshold
        else:  # Neutral
            volume_condition = abs(volume_change) < volume_threshold

        if volume_condition:
            volume_conf = min(1.0, abs(volume_change) / (volume_threshold * 3))
        else:
            volume_conf = 0.0

        # Calculate overall confidence
        weights = self.hurdle_weights
        overall_conf = (
                price_conf * weights[0] +
                vol_conf * weights[1] +
                volume_conf * weights[2]
        )

        # Determine trend based on hurdles and confidence
        if base_trend != 1 and overall_conf >= 0.6:
            return base_trend, overall_conf
        else:
            # Not enough confidence, return neutral
            return 1, overall_conf

    def _log_dataset_statistics(self):
        """Log basic statistics about the dataset and labels."""
        # Class distribution
        trend_labels = self.y.numpy()
        class_counts = {
            0: np.sum(trend_labels == 0),  # Down
            1: np.sum(trend_labels == 1),  # Neutral
            2: np.sum(trend_labels == 2)  # Up
        }

        total = len(trend_labels)
        class_percentages = {k: round(v / total * 100, 2) for k, v in class_counts.items()}

        print(f"\n=== TripleHurdle Dataset Statistics ===")
        print(f"Total samples: {total}")
        print(f"Sequence length: {self.sequence_length}, Forecast horizon: {self.forecast_horizon}")
        print(f"Class distribution:")
        print(f"  Down   (0): {class_counts[0]} samples ({class_percentages[0]}%)")
        print(f"  Neutral(1): {class_counts[1]} samples ({class_percentages[1]}%)")
        print(f"  Up     (2): {class_counts[2]} samples ({class_percentages[2]}%)")

        if self.confidence_scoring:
            conf_array = self.confidences.numpy()
            print(f"Confidence metrics:")
            print(f"  Average confidence: {np.mean(conf_array):.4f}")
            print(f"  Confidence by class:")
            for cls in [0, 1, 2]:
                cls_conf = conf_array[trend_labels == cls]
                if len(cls_conf) > 0:
                    print(f"    Class {cls}: {np.mean(cls_conf):.4f}")

        print("===================================\n")

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        if self.confidence_scoring:
            return self.X[idx], self.y[idx], self.confidences[idx]
        else:
            return self.X[idx], self.y[idx]

    def get_hurdle_details(self, idx):
        """Get detailed information about the triple hurdle classification."""
        return self.hurdle_details[idx]

    def get_trend_statistics(self):
        """Get comprehensive statistics about the labeled trends."""
        trend_labels = self.y.numpy()

        # Count occurrences of each trend
        trend_counts = {
            'Down': np.sum(trend_labels == 0),
            'Neutral': np.sum(trend_labels == 1),
            'Up': np.sum(trend_labels == 2)
        }

        # Calculate percentages
        total = len(trend_labels)
        trend_percentages = {k: round(v / total * 100, 2) for k, v in trend_counts.items()}

        # Calculate average price change by trend
        avg_change_by_trend = {}
        for trend_name, trend_val in [('Down', 0), ('Neutral', 1), ('Up', 2)]:
            mask = trend_labels == trend_val
            if np.any(mask):
                avg_change = np.mean(self.price_changes[mask])
                avg_change_by_trend[trend_name] = round(avg_change * 100, 2)
            else:
                avg_change_by_trend[trend_name] = 0

        # If confidence scoring is enabled, calculate average confidence by trend
        if self.confidence_scoring:
            conf_array = self.confidences.numpy()
            confidence_by_trend = {}
            for trend_name, trend_val in [('Down', 0), ('Neutral', 1), ('Up', 2)]:
                mask = trend_labels == trend_val
                if np.any(mask):
                    avg_conf = np.mean(conf_array[mask])
                    confidence_by_trend[trend_name] = round(avg_conf, 4)
                else:
                    confidence_by_trend[trend_name] = 0
        else:
            confidence_by_trend = None

        # Calculate streak statistics
        streaks = self._calculate_streaks()

        return {
            'counts': trend_counts,
            'percentages': trend_percentages,
            'avg_price_change_pct': avg_change_by_trend,
            'confidence': confidence_by_trend,
            'streaks': streaks,
            'total': total
        }

    def _calculate_streaks(self):
        """Calculate statistics about trend streaks."""
        trend_labels = self.y.numpy()

        streak_counts = {0: [], 1: [], 2: []}
        current_trend = trend_labels[0]
        current_streak = 1

        for i in range(1, len(trend_labels)):
            if trend_labels[i] == current_trend:
                current_streak += 1
            else:
                streak_counts[current_trend].append(current_streak)
                current_trend = trend_labels[i]
                current_streak = 1

        # Add the last streak
        streak_counts[current_trend].append(current_streak)

        # Calculate streak statistics
        streak_stats = {}
        for trend, streaks in streak_counts.items():
            if streaks:
                trend_name = {0: 'Down', 1: 'Neutral', 2: 'Up'}[trend]
                streak_stats[trend_name] = {
                    'max': max(streaks),
                    'avg': round(np.mean(streaks), 2),
                    'median': round(np.median(streaks), 2),
                    'count': len(streaks)
                }

        return streak_stats

    def get_sequential_patterns(self, window=3):
        """Analyze sequential patterns in the labeled trends with profitability metrics."""
        trend_labels = self.y.numpy()
        patterns = defaultdict(list)

        # Look for patterns and their future outcomes
        for i in range(len(trend_labels) - window - self.forecast_horizon):
            pattern = tuple(trend_labels[i:i + window])

            # Calculate future price change
            pattern_end_idx = i + window + self.sequence_length - 1
            future_idx = pattern_end_idx + self.forecast_horizon

            if future_idx < len(self.prices):
                current_price = self.prices[pattern_end_idx]
                future_price = self.prices[future_idx]

                if current_price > 0:
                    future_change = (future_price - current_price) / current_price
                else:
                    future_change = 0

                patterns[pattern].append(future_change)

        # Calculate statistics for each pattern
        pattern_stats = {}
        for pattern, changes in patterns.items():
            if len(changes) >= 5:  # Only include patterns with sufficient data
                pos_changes = [c for c in changes if c > 0]
                neg_changes = [c for c in changes if c < 0]

                pattern_stats[pattern] = {
                    'count': len(changes),
                    'mean_return': round(np.mean(changes) * 100, 2),
                    'median_return': round(np.median(changes) * 100, 2),
                    'win_rate': round(len(pos_changes) / len(changes) * 100, 2) if changes else 0,
                    'avg_win': round(np.mean(pos_changes) * 100, 2) if pos_changes else 0,
                    'avg_loss': round(np.mean(neg_changes) * 100, 2) if neg_changes else 0,
                    'profit_factor': round(
                        np.sum(pos_changes) / abs(np.sum(neg_changes)), 2
                    ) if neg_changes and pos_changes else float('inf')
                }

                # Add human-readable pattern description
                pattern_stats[pattern]['sequence'] = '-'.join(
                    ['Down' if t == 0 else 'Neutral' if t == 1 else 'Up' for t in pattern]
                )

        # Sort by profitability (profit factor * win rate)
        return sorted(
            pattern_stats.items(),
            key=lambda x: x[1]['profit_factor'] * x[1]['win_rate'],
            reverse=True
        )

    def cross_validate_labels(self, n_splits=5, model_class=None):
        """Validate label quality using predictive performance."""
        from sklearn.model_selection import TimeSeriesSplit

        X = self.X.numpy()
        y = self.y.numpy()

        # Reshape features for sklearn models
        X_reshaped = X.reshape(X.shape[0], -1)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = {'balanced_acc': [], 'f1': [], 'accuracy': []}

        # Use a simple model if none provided
        if model_class is None:
            from sklearn.ensemble import RandomForestClassifier
            model_class = RandomForestClassifier

        print(f"Performing {n_splits}-fold time series cross-validation...")

        for i, (train_idx, test_idx) in enumerate(tscv.split(X_reshaped)):
            X_train, X_test = X_reshaped[train_idx], X_reshaped[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train and evaluate
            model = model_class()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            scores['balanced_acc'].append(balanced_accuracy_score(y_test, y_pred))
            scores['f1'].append(f1_score(y_test, y_pred, average='weighted'))
            scores['accuracy'].append(accuracy_score(y_test, y_pred))

            print(f"Fold {i + 1}: Accuracy = {scores['accuracy'][-1]:.4f}, "
                  f"Balanced Acc = {scores['balanced_acc'][-1]:.4f}, "
                  f"F1 = {scores['f1'][-1]:.4f}")

        # Calculate average and std
        results = {}
        for metric, values in scores.items():
            results[metric] = {
                'mean': round(np.mean(values), 4),
                'std': round(np.std(values), 4),
                'values': values
            }

        return results

    def analyze_hurdle_importance(self):
        """Analyze which hurdles contribute most to successful predictions."""
        true_changes = self.price_changes
        trend_labels = self.y.numpy()

        # Determine actual trends using base thresholds
        actual_trends = []
        for change in true_changes:
            if change > self.base_price_threshold:
                actual_trends.append(2)  # Up
            elif change < -self.base_price_threshold:
                actual_trends.append(0)  # Down
            else:
                actual_trends.append(1)  # Neutral

        actual_trends = np.array(actual_trends)

        # Create predictions using different hurdle combinations
        hurdle_combinations = [
            ("Price Only", [True, False, False]),
            ("Price + Volatility", [True, True, False]),
            ("Price + Volume", [True, False, True]),
            ("All Hurdles", [True, True, True])
        ]

        results = {}

        for name, hurdles_used in hurdle_combinations:
            # Create predictions using only specified hurdles
            predictions = []

            for i, details in enumerate(self.hurdle_details):
                price_change = details['price_change']
                volatility = details['volatility']
                volume_change = details['volume_change']
                vol_increasing = details['volatility_increasing']

                price_threshold = details['price_threshold']
                vol_threshold = details['vol_threshold']
                volume_threshold = details['volume_threshold']

                # Initial trend based on price
                if price_change > price_threshold:
                    trend = 2
                elif price_change < -price_threshold:
                    trend = 0
                else:
                    trend = 1

                # Apply additional hurdles if specified
                if hurdles_used[1] and trend != 1:  # Volatility hurdle
                    if trend == 2:  # Uptrend
                        if volatility < vol_threshold and not vol_increasing:
                            trend = 1  # Fails volatility hurdle
                    else:  # Downtrend
                        if volatility < vol_threshold and vol_increasing:
                            trend = 1  # Fails volatility hurdle

                if hurdles_used[2] and trend != 1:  # Volume hurdle
                    if trend == 2:  # Uptrend
                        if volume_change < volume_threshold:
                            trend = 1  # Fails volume hurdle
                    else:  # Downtrend
                        if volume_change > -volume_threshold:
                            trend = 1  # Fails volume hurdle

                predictions.append(trend)

            predictions = np.array(predictions)

            # Calculate performance metrics
            accuracy = accuracy_score(actual_trends, predictions)
            balanced_acc = balanced_accuracy_score(actual_trends, predictions)
            f1 = f1_score(actual_trends, predictions, average='weighted')

            # Calculate class-specific metrics
            class_metrics = {}
            for cls in [0, 1, 2]:
                cls_mask = (actual_trends == cls)
                if np.any(cls_mask):
                    cls_acc = accuracy_score(actual_trends[cls_mask], predictions[cls_mask])
                    class_metrics[cls] = round(cls_acc, 4)

            results[name] = {
                'accuracy': round(accuracy, 4),
                'balanced_accuracy': round(balanced_acc, 4),
                'f1_score': round(f1, 4),
                'class_accuracy': class_metrics
            }

        return results

    def simulate_trading_strategy(self, starting_capital=10000, risk_per_trade=0.02,
                                  confidence_threshold=0.7, use_stops=True, stop_loss_pct=0.02):
        """Simulate a trading strategy based on the labels."""
        capital = starting_capital
        positions = []
        equity_curve = [capital]

        trend_labels = self.y.numpy()

        for i in range(len(trend_labels)):
            if i >= len(self.prices) - self.forecast_horizon:
                break

            # Current price at decision point
            entry_idx = i + self.sequence_length - 1
            entry_price = self.prices[entry_idx]

            # Future price at target horizon
            exit_idx = entry_idx + self.forecast_horizon
            exit_price = self.prices[exit_idx]

            # Skip if confidence is below threshold
            if self.confidence_scoring:
                confidence = self.confidences[i].item()
                if confidence < confidence_threshold:
                    continue

            # Trading logic based on predicted trend
            if trend_labels[i] == 2:  # Uptrend
                # Long position
                position_size = capital * risk_per_trade / entry_price

                # Simulate price path for stop loss
                if use_stops:
                    stop_price = entry_price * (1 - stop_loss_pct)
                    price_path = self.prices[entry_idx:exit_idx + 1]

                    # Check if stop loss was hit
                    if np.any(price_path < stop_price):
                        stop_idx = np.where(price_path < stop_price)[0][0]
                        exit_price = stop_price
                        stopped_out = True
                    else:
                        stopped_out = False
                else:
                    stopped_out = False

                profit_loss = position_size * (exit_price - entry_price)
                positions.append({
                    'type': 'LONG',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': position_size,
                    'profit_loss': profit_loss,
                    'pct_change': (exit_price - entry_price) / entry_price,
                    'stopped_out': stopped_out
                })
                capital += profit_loss

            elif trend_labels[i] == 0:  # Downtrend
                # Short position
                position_size = capital * risk_per_trade / entry_price

                # Simulate price path for stop loss
                if use_stops:
                    stop_price = entry_price * (1 + stop_loss_pct)
                    price_path = self.prices[entry_idx:exit_idx + 1]

                    # Check if stop loss was hit
                    if np.any(price_path > stop_price):
                        stop_idx = np.where(price_path > stop_price)[0][0]
                        exit_price = stop_price
                        stopped_out = True
                    else:
                        stopped_out = False
                else:
                    stopped_out = False

                profit_loss = position_size * (entry_price - exit_price)
                positions.append({
                    'type': 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': position_size,
                    'profit_loss': profit_loss,
                    'pct_change': (entry_price - exit_price) / entry_price,
                    'stopped_out': stopped_out
                })
                capital += profit_loss

            equity_curve.append(capital)

        # Calculate performance metrics
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            drawdowns = 1 - np.array(equity_curve) / np.maximum.accumulate(equity_curve)
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

            # Count wins and losses
            win_count = sum(1 for p in positions if p['profit_loss'] > 0)
            loss_count = sum(1 for p in positions if p['profit_loss'] <= 0)

            win_rate = win_count / len(positions) if positions else 0

            # Average win/loss
            avg_win = np.mean([p['profit_loss'] for p in positions if p['profit_loss'] > 0]) if win_count > 0 else 0
            avg_loss = np.mean([p['profit_loss'] for p in positions if p['profit_loss'] <= 0]) if loss_count > 0 else 0

            # Total P&L
            total_profit = sum(p['profit_loss'] for p in positions if p['profit_loss'] > 0)
            total_loss = sum(p['profit_loss'] for p in positions if p['profit_loss'] <= 0)

            # Profit factor
            profit_factor = -total_profit / total_loss if total_loss < 0 else float('inf')
        else:
            sharpe = 0
            max_drawdown = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return {
            'final_capital': capital,
            'total_return': (capital - starting_capital) / starting_capital,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(positions),
            'stopped_out_count': sum(1 for p in positions if p.get('stopped_out', False)),
            'equity_curve': equity_curve,
            'positions': positions
        }

    def visualize_labeled_trends(self, price_series=None, title="Triple Hurdle Labeled Trends"):
        """
        Visualize the labeled trends with all triple hurdle components.

        Args:
            price_series: Price data to plot (defaults to using target column data)
            title: Plot title

        Returns:
            Plotly figure object
        """
        trend_labels = self.y.numpy()
        x = np.arange(self.sequence_length, self.sequence_length + len(trend_labels))

        if price_series is None:
            price_series = self.prices

        price_series = np.array(price_series)

        # Create subplots: price with trends, volume, volatility, regime
        n_rows = 3 + (1 if self.regime_detection else 0) + (1 if self.confidence_scoring else 0)
        subplot_titles = ['Price with Trend Labels', 'Volume', 'Volatility']

        if self.regime_detection:
            subplot_titles.append('Market Regime')

        if self.confidence_scoring:
            subplot_titles.append('Prediction Confidence')

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=subplot_titles,
            row_heights=[0.5] + [0.25] * (n_rows - 1)
        )

        # Plot 1: Price with trend labels
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(price_series)),
                y=price_series,
                mode='lines',
                name='Price',
                line=dict(color='black', width=1.5)
            ),
            row=1, col=1
        )

        # Add trend markers
        colors = {0: 'red', 1: 'gray', 2: 'green'}
        symbols = {0: 'triangle-down', 1: 'circle', 2: 'triangle-up'}
        labels = {0: 'Down', 1: 'Neutral', 2: 'Up'}

        for trend in [0, 1, 2]:
            idxs = x[trend_labels == trend]
            if len(idxs) > 0:  # Only add trace if there are points with this trend
                fig.add_trace(
                    go.Scatter(
                        x=idxs,
                        y=price_series[idxs],
                        mode='markers',
                        name=labels[trend],
                        marker=dict(color=colors[trend], symbol=symbols[trend], size=8),
                    ),
                    row=1, col=1
                )

        # Plot 2: Volume
        if self.volume_col in self.data.columns and not self.volume_col_is_proxy:
            fig.add_trace(
                go.Bar(
                    x=np.arange(len(self.volumes)),
                    y=self.volumes,
                    name='Volume',
                    marker=dict(color='lightblue'),
                    opacity=0.7
                ),
                row=2, col=1
            )
        else:
            # Use a placeholder for volume
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 0],
                    mode='lines',
                    name='Volume (Not Available)',
                    line=dict(color='lightgray')
                ),
                row=2, col=1
            )

        # Plot 3: Volatility
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(self.data['volatility'])),
                y=self.data['volatility'],
                mode='lines',
                name='Volatility',
                line=dict(color='purple')
            ),
            row=3, col=1
        )

        # Add volatility threshold line
        fig.add_trace(
            go.Scatter(
                x=[0, len(self.data['volatility'])],
                y=[self.base_volatility_threshold, self.base_volatility_threshold],
                mode='lines',
                name='Base Vol Threshold',
                line=dict(color='purple', dash='dash')
            ),
            row=3, col=1
        )

        # Plot 4: Market Regime (if enabled)
        current_row = 4
        if self.regime_detection and 'regime' in self.data.columns:
            # Convert regime to numeric for plotting
            regime_map = {'trending': 2, 'volatile': 1, 'ranging': 0, 'normal': 0.5}
            regime_numeric = np.array([regime_map.get(r, 0) for r in self.data['regime']])

            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(regime_numeric)),
                    y=regime_numeric,
                    mode='lines',
                    name='Market Regime',
                    line=dict(color='orange', width=2)
                ),
                row=current_row, col=1
            )

            # Add colored background for regimes
            regime_colors = {'trending': 'rgba(0,255,0,0.1)', 'volatile': 'rgba(255,0,0,0.1)',
                             'ranging': 'rgba(0,0,255,0.1)', 'normal': 'rgba(200,200,200,0.1)'}

            for regime in np.unique(self.data['regime']):
                regime_periods = []
                in_period = False
                start_idx = None

                for i, r in enumerate(self.data['regime']):
                    if r == regime and not in_period:
                        in_period = True
                        start_idx = i
                    elif r != regime and in_period:
                        regime_periods.append((start_idx, i - 1))
                        in_period = False

                # Handle case if we end while in a period
                if in_period:
                    regime_periods.append((start_idx, len(self.data['regime']) - 1))

                # Add colored rectangles for each period
                for start, end in regime_periods:
                    fig.add_shape(
                        type="rect",
                        x0=start, y0=0,
                        x1=end, y1=2.5,
                        fillcolor=regime_colors.get(regime, 'rgba(200,200,200,0.1)'),
                        line=dict(width=0),
                        layer="below",
                        row=current_row, col=1
                    )

            fig.update_yaxes(
                tickvals=[0, 1, 2],
                ticktext=['Ranging', 'Volatile', 'Trending'],
                row=current_row, col=1
            )

            current_row += 1

        # Plot 5: Confidence (if enabled)
        if self.confidence_scoring:
            conf_array = self.confidences.numpy()

            # Colored confidence by trend
            for trend in [0, 1, 2]:
                trend_indices = np.where(trend_labels == trend)[0]
                if len(trend_indices) > 0:
                    trend_x = x[trend_indices]
                    trend_conf = conf_array[trend_indices]

                    fig.add_trace(
                        go.Bar(
                            x=trend_x,
                            y=trend_conf,
                            name=f'{labels[trend]} Confidence',
                            marker=dict(color=colors[trend]),
                            opacity=0.7
                        ),
                        row=current_row, col=1
                    )

            # Add threshold line
            fig.add_trace(
                go.Scatter(
                    x=[x[0], x[-1]],
                    y=[0.6, 0.6],
                    mode='lines',
                    name='Confidence Threshold',
                    line=dict(color='black', dash='dash')
                ),
                row=current_row, col=1
            )

            fig.update_yaxes(range=[0, 1], row=current_row, col=1)

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Time Step',
            template='plotly_white',
            height=250 * n_rows,
            width=1000,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    def visualize_trading_simulation(self, sim_results):
        """Visualize the results of a trading simulation."""
        equity_curve = sim_results['equity_curve']
        positions = sim_results['positions']

        # Create figure
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Equity Curve', 'Trade P&L'),
            row_heights=[0.7, 0.3]
        )

        # Plot equity curve
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(equity_curve)),
                y=equity_curve,
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        # Plot drawdown as filled area
        max_equity = np.maximum.accumulate(equity_curve)
        drawdown = 1 - np.array(equity_curve) / max_equity
        drawdown_pct = drawdown * 100

        # Add drawdown as a separate y-axis
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(drawdown_pct)),
                y=drawdown_pct,
                mode='lines',
                name='Drawdown %',
                line=dict(color='red'),
                fill='tozeroy',
                yaxis='y2'
            ),
            row=1, col=1
        )

        # Plot individual trade P&L
        trade_idx = np.arange(len(positions))
        pnl = [p['profit_loss'] for p in positions]
        colors = ['green' if p > 0 else 'red' for p in pnl]

        fig.add_trace(
            go.Bar(
                x=trade_idx,
                y=pnl,
                name='Trade P&L',
                marker=dict(color=colors),
                opacity=0.8
            ),
            row=2, col=1
        )

        # Add horizontal line at zero for P&L
        fig.add_trace(
            go.Scatter(
                x=[0, len(positions)],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=f"Trading Simulation Results (Return: {sim_results['total_return']:.2%}, Sharpe: {sim_results['sharpe_ratio']:.2f})",
            xaxis2_title='Trade #',
            yaxis_title='Equity',
            yaxis2=dict(
                title='Drawdown %',
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
                overlaying='y',
                side='right',
                range=[0, max(50, np.max(drawdown_pct) * 1.1)]  # Cap at 50% or actual max * 1.1
            ),
            template='plotly_white',
            height=800,
            width=1000,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Add metrics as annotations
        metrics_text = (
            f"<b>Return:</b> {sim_results['total_return']:.2%} | "
            f"<b>Sharpe:</b> {sim_results['sharpe_ratio']:.2f} | "
            f"<b>Max DD:</b> {sim_results['max_drawdown']:.2%} | "
            f"<b>Win Rate:</b> {sim_results['win_rate']:.2%} | "
            f"<b>Profit Factor:</b> {sim_results['profit_factor']:.2f} | "
            f"<b>Trades:</b> {sim_results['total_trades']}"
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            text=metrics_text,
            showarrow=False,
            font=dict(size=12),
            align="center"
        )

        return fig





from collections import Counter
import numpy as np

class BalancedDatasetBuilder:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def balance(self, X, y):
        np.random.seed(self.random_state)
        class_counts = Counter(y)
        max_count = max(class_counts.values())

        X_balanced = []
        y_balanced = []

        for cls in class_counts:
            idxs = np.where(y == cls)[0]
            needed = max_count - len(idxs)

            # Add original samples
            X_balanced.append(X[idxs])
            y_balanced.append(y[idxs])

            # Oversample if needed
            if needed > 0:
                sampled_idxs = np.random.choice(idxs, size=needed, replace=True)
                X_balanced.append(X[sampled_idxs])
                y_balanced.append(y[sampled_idxs])

        X_balanced = np.concatenate(X_balanced, axis=0)
        y_balanced = np.concatenate(y_balanced, axis=0)

        # Shuffle the result
        shuffled_idxs = np.random.permutation(len(y_balanced))
        return X_balanced[shuffled_idxs], y_balanced[shuffled_idxs]


from collections import Counter
import numpy as np

class BalancedDatasetUndersampling:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def balance(self, X, y):
        np.random.seed(self.random_state)
        class_counts = Counter(y)
        min_count = min(class_counts.values())

        X_balanced = []
        y_balanced = []

        for cls in class_counts:
            idxs = np.where(y == cls)[0]
            selected_idxs = np.random.choice(idxs, size=min_count, replace=False)

            X_balanced.append(X[selected_idxs])
            y_balanced.append(y[selected_idxs])

        X_balanced = np.concatenate(X_balanced, axis=0)
        y_balanced = np.concatenate(y_balanced, axis=0)

        # Shuffle the result
        shuffled_idxs = np.random.permutation(len(y_balanced))
        return X_balanced[shuffled_idxs], y_balanced[shuffled_idxs]


class BalancedDatasetBuilderSmartUndersampling:
    def __init__(self, random_state=42, reduction_ratio=1.0):
        """
        reduction_ratio: نسبت کاهش داده‌های کلاس‌های بزرگ نسبت به کلاس کوچک.
        1.0 یعنی همه رو برابر با کلاس کم می‌کنه (سخت‌ترین حالت).
        بالاتر از 1.0 یعنی داده بیشتری نگه می‌داره.
        """
        self.random_state = random_state
        self.reduction_ratio = reduction_ratio

    def balance(self, X, y):
        np.random.seed(self.random_state)
        class_counts = Counter(y)
        min_count = min(class_counts.values())
        target_count = int(min_count * self.reduction_ratio)

        X_balanced = []
        y_balanced = []

        for cls in class_counts:
            idxs = np.where(y == cls)[0]
            keep_count = min(len(idxs), target_count)
            selected_idxs = np.random.choice(idxs, size=keep_count, replace=False)

            X_balanced.append(X[selected_idxs])
            y_balanced.append(y[selected_idxs])

        X_balanced = np.concatenate(X_balanced, axis=0)
        y_balanced = np.concatenate(y_balanced, axis=0)

        # Shuffle
        shuffled_idxs = np.random.permutation(len(y_balanced))
        return X_balanced[shuffled_idxs], y_balanced[shuffled_idxs]

