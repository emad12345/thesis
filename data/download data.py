import os
import requests
import pandas as pd
import datetime
import lzma
import struct

def get_dukascopy_url(symbol, year, month, day, hour):
    # نمونه: https://datafeed.dukascopy.com/datafeed/EURUSD/2023/01/01/00h_ticks.bi5
    base_url = "https://datafeed.dukascopy.com/datafeed"
    return f"{base_url}/{symbol}/{year}/{month:02}/{day:02}/{hour:02}h_ticks.bi5"

def download_bi5_file(url):
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return lzma.decompress(resp.content)
        else:
            return None
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def parse_bi5(data):
    rows = []
    record_size = 20
    for i in range(0, len(data), record_size):
        chunk = data[i:i+record_size]
        if len(chunk) != record_size:
            continue
        millis, ask, bid, ask_vol, bid_vol = struct.unpack('>IIIff', chunk)
        timestamp = millis / 1000.0  # seconds
        rows.append((timestamp, ask / 100000.0, bid / 100000.0, ask_vol, bid_vol))
    return rows

def get_ticks_for_day(symbol, date):
    all_rows = []
    for hour in range(24):
        url = get_dukascopy_url(symbol, date.year, date.month, date.day, hour)
        data = download_bi5_file(url)
        if data:
            rows = parse_bi5(data)
            for row in rows:
                ts = datetime.datetime.combine(date, datetime.time(hour=0)) + datetime.timedelta(seconds=row[0])
                all_rows.append([ts] + list(row[1:]))
    df = pd.DataFrame(all_rows, columns=['datetime', 'ask', 'bid', 'ask_vol', 'bid_vol'])
    df.set_index('datetime', inplace=True)
    return df

# دانلود داده‌ها برای چند روز از EUR/USD
start_date = datetime.date(2023, 1, 1)
end_date = datetime.date(2023, 1, 3)  # می‌توانید تا سال 2010 یا پایین‌تر هم بروید

symbol = 'EURUSD'
all_data = []

current = start_date
while current <= end_date:
    print(f"Downloading data for {current}")
    df = get_ticks_for_day(symbol, current)
    all_data.append(df)
    current += datetime.timedelta(days=1)

df_all = pd.concat(all_data)
df_all.to_csv("eurusd_ticks.csv")
print("Saved all data to eurusd_ticks.csv")
