import model_gens.predictor as predictor

predictor

# import requests
# import pandas as pd
# from datetime import datetime, timedelta

# # Define the API endpoint and parameters
# base_url = 'https://api.pro.coinbase.com/products/BTC-USD/candles'
# granularity = 300  # 5 minutes in seconds
# end_time = datetime.utcnow()
# start_time = end_time - timedelta(days=730)

# # Function to fetch data in chunks
# def fetch_data(start, end, granularity):
#     url = f"{base_url}?start={start.isoformat()}&end={end.isoformat()}&granularity={granularity}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         raise Exception(f"Failed to fetch data: {response.status_code}")

# # Collect data in a loop
# data = []
# current_start = start_time

# while current_start < end_time:
#     current_end = min(current_start + timedelta(hours=6), end_time)
#     print(f"Fetching data from {current_start} to {current_end}")
#     chunk = fetch_data(current_start, current_end, granularity)
#     print(f"Received {len(chunk)} data points")
#     data.extend(chunk)
#     current_start = current_end

# # Convert to DataFrame and format
# columns = ['Date Time', 'Open', 'High', 'Low', 'Close', 'Volume']
# df = pd.DataFrame(data, columns=columns)
# df['Date Time'] = pd.to_datetime(df['Date Time'], unit='s')
# df = df.sort_values('Date Time').reset_index(drop=True)

# # Save to CSV or use as needed
# df.to_csv('Datas/BTCUSD/combined.csv', index=False)
# print(df.head())
