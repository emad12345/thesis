import numpy as np
import pandas as pd
import vectorbt as vbt

# داده ساختگی EUR/USD
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=300, freq='D')
price = 1.1 + np.cumsum(np.random.randn(len(dates)) * 0.002)
price_series = pd.Series(price, index=dates, name="EURUSD_Close")

# پیش‌بینی مدل
predicted_return = np.random.randn(len(dates)) * 0.002
predicted_signal = np.where(predicted_return > 0.0005, 1, np.where(predicted_return < -0.0005, -1, 0))

# سیگنال‌های ورود و خروج
entries = predicted_signal == 1
exits = predicted_signal == -1

# بک‌تست با vectorbt
portfolio = vbt.Portfolio.from_signals(
    close=price_series,
    entries=entries,
    exits=exits,
    direction='both'
)

# آمار کلی
print(portfolio.stats())

# نمایش نمودار
portfolio.plot().show()
