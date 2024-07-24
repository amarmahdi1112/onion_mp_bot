# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression

import os

# Set the base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Sample data
# data = {
#     'Open': [62848.78, 62848.56, 62848.62, 62845.02, 62847.48, 62848.53, 62676.54, 62716.32, 62753.82, 62791.53,
#              62842.37, 62842.35, 62816.73, 62902.17, 62906.84, 62904.43, 62919.73, 63499.85, 63499.84, 63498.47],
#     'High': [62849.0, 62848.92, 62848.75, 62848.45, 62848.71, 62848.68, 62716.32, 62753.82, 62791.54, 62842.48,
#              62842.39, 62842.43, 62902.43, 62906.84, 62913.27, 62919.73, 63499.86, 63499.85, 63499.86, 63547.51],
#     'Low': [62848.0, 62848.15, 62839.58, 62845.02, 62838.32, 62676.28, 62676.3, 62716.28, 62753.82, 62791.53,
#             62842.35, 62816.18, 62816.38, 62902.16, 62902.8, 62904.06, 62909.93, 63499.77, 63498.47, 63498.47],
#     'Close': [62848.67, 62848.62, 62845.02, 62847.48, 62848.53, 62676.54, 62716.32, 62753.82, 62791.53, 62842.37,
#               62842.35, 62816.73, 62902.18, 62906.84, 62904.43, 62919.73, 63499.85, 63499.84, 63498.47, 63547.51],
#     'Volume': [55, 53, 65, 59, 47, 63, 56, 59, 60, 57, 52, 54, 55, 56, 59, 57, 57, 61, 63, 58]

# }

# df = pd.DataFrame(data)

# # Calculate Pearson correlation coefficient
# correlation_open_high = pearsonr(df['Open'], df['High'])[0]
# print(f'Pearson Correlation (Open, High): {correlation_open_high}')

# # Perform Linear Regression
# X = df[['Open', 'Low', 'Close', 'Volume']]
# y = df['High']
# reg = LinearRegression().fit(X, y)
# print(f'Intercept: {reg.intercept_}')
# print(f'Coefficients: {reg.coef_}')

# # Make predictions
# predictions = reg.predict(X)

# # Plot results
# plt.scatter(df['Open'], df['High'], color='blue', label='Actual High Prices')
# plt.plot(df['Open'], predictions, color='red', label='Predicted High Prices')
# plt.xlabel('Open Prices')
# plt.ylabel('High Prices')
# plt.legend()
# plt.show()
