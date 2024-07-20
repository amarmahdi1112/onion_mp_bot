from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from tqdm import tqdm
from model_gens.utils.static.columns import Columns
import pickle

class PricePredictor:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.model = None
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.lags = range(1, 60)

    def prepare_features(self):
        potential_targets = [Columns.Open, Columns.High, Columns.Low, Columns.Close, Columns.Volume, Columns.Open_Close_Diff]
        # print(f"Preparing features for {self.target_column.name}")
        y = self.data[self.target_column.name].fillna(0)
        for potential_target in potential_targets:
            expected_str = f'expected_next_{potential_target.name.lower()}'
            true_diff_str = f'true_{potential_target.name.lower()}_diff'
            if expected_str in self.data.columns:
                self.data.drop(columns=[expected_str], inplace=True)
            if true_diff_str in self.data.columns:
                self.data.drop(columns=[true_diff_str], inplace=True)
            if Columns.Open_Close_Diff.name in self.data.columns:
                self.data.drop(columns=[Columns.Open_Close_Diff.name], inplace=True)
        X = self.data.fillna(0)
        return X, y

    def train_model(self):
        X, y = self.prepare_features()
        self.model = LinearRegression().fit(X, y)
        # self.save_model()

    def predict_value(self, features):
        return self.model.intercept_ + np.dot(self.model.coef_, features)

    def analyze_and_predict(self):
        target_feature = self.target_column.name
        self.data[f'expected_next_{target_feature.lower()}'] = np.nan
        self.data[f'true_{target_feature.lower()}_diff'] = np.nan

        for i in tqdm(range(max(self.lags), len(self.data)), desc=f"Analyzing and predicting {target_feature}"):
            next_X = self.data.iloc[i][self.features].values
            predicted_value = self.predict_value(next_X)

            self.data.at[self.data.index[i], f'expected_next_{target_feature.lower()}'] = predicted_value
            if i < len(self.data) - 1:
                true_next_value = self.data.iloc[i + 1][target_feature]
                self.data.at[self.data.index[i], f'true_{target_feature.lower()}_diff'] = true_next_value - predicted_value

        return self.data
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
