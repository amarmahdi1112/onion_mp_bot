from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import pandas as pd
import ta
from enum import Enum
import glob
import os
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from settings import BASE_DIR
from model_gens.utils.static.columns import Columns
from tqdm import tqdm
from model_gens.utils.static.processing_type import ProcessingType
from model_gens.utils.static.indicator import Indicator


class Preprocessor:
    def __init__(self, path=None, target_column=None, processing_type=ProcessingType.INITIAL):
        self.path = path
        self.currency = 'BTCUSDT'
        self.processing_type = processing_type
        self.target_column = target_column
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._data = pd.DataFrame()
        self._open_close_diff = pd.Series(dtype=float)
        self._load_data()

    # Data loading methods
    def _load_data(self):
        if self.processing_type == ProcessingType.INITIAL:
            if os.path.isfile(self.path):
                self._load_data_file()
            else:
                raise ValueError(f"Path {self.path} is not valid.")
        elif self.processing_type == ProcessingType.TRAINING:
            if os.path.isdir(self.path):
                if not self.target_column:
                    raise ValueError("Target column not specified.")
                self._load_data_file()

    def _load_data_file(self):
        try:
            print(f"Reading data from {self.path}...")
            if self.processing_type == ProcessingType.INITIAL:
                data = pd.read_csv(self.path, parse_dates=True, index_col=0)
                self._post_process_data(data)
            elif self.processing_type == ProcessingType.TRAINING:
                if not os.path.isdir(self.path):
                    raise ValueError(f"Path {self.path} is not a directory.")
                if self.target_column == Columns.Open:
                    data = pd.read_csv(
                        f'{self.path}/open/open_prediction_preprocessed_data.csv', parse_dates=True, index_col=0)
                elif self.target_column == Columns.Close:
                    data = pd.read_csv(
                        f'{self.path}/close/close_prediction_preprocessed_data.csv', parse_dates=True, index_col=0)
                elif self.target_column == Columns.High:
                    print(f"Reading high data... {
                          self.path}/high/high_prediction_preprocessed_data.csv")
                    data = pd.read_csv(
                        f'{self.path}/high/high_prediction_preprocessed_data.csv', parse_dates=True, index_col=0)
                elif self.target_column == Columns.Low:
                    data = pd.read_csv(
                        f'{self.path}/low/low_prediction_preprocessed_data.csv', parse_dates=True, index_col=0)
                self._post_process_data(data)
        except Exception as e:
            raise ValueError(f"Error: {e}")

    def _post_process_data(self, data=None):
        print("Post processing data...")
        if data is None:
            data = self._data
        print('Dropping unnecessary columns...')
        self._drop_unnecessary_columns(data)
        print('Filling missing values...')
        self._fill_missing_values(data)
        print('Ensuring required columns...')
        self._ensure_required_columns(data)
        self._data = data
        print('Setting attributes...')
        self._set_attributes()
        print('Data post processing complete.')

    def _ensure_required_columns(self, data):
        if self.processing_type == ProcessingType.INITIAL:
            required_columns = [Columns.Open.name, Columns.High.name,
                                Columns.Low.name, Columns.Close.name, Columns.Volume.name]
            if not all(col in data.columns for col in required_columns):
                missing = set(required_columns) - set(data.columns)
                raise ValueError(f"Missing required columns: {missing}")
        elif self.processing_type == ProcessingType.TRAINING:
            if self.target_column == Columns.Open:
                required_columns = [Columns.Open.name, Columns.High.name,
                                    Columns.Low.name, Columns.Close.name, Columns.Open_Close_Diff.name]
            elif self.target_column == Columns.Close:
                required_columns = [Columns.Open.name, Columns.High.name,
                                    Columns.Low.name, Columns.Close.name, 'expected_next_close', 'true_close_diff']
            elif self.target_column == Columns.High:
                required_columns = [Columns.Open.name, Columns.High.name,
                                    Columns.Low.name, Columns.Close.name, 'expected_next_high', 'true_high_diff']
            elif self.target_column == Columns.Low:
                required_columns = [Columns.Open.name, Columns.High.name,
                                    Columns.Low.name, Columns.Close.name, 'expected_next_low', 'true_low_diff']
            if not all(col in data.columns for col in required_columns):
                missing = set(required_columns) - set(data.columns)
                raise ValueError(f"Missing required columns: {missing}")

    def _drop_unnecessary_columns(self, data):
        columns_to_drop = ["Change", 'Open Interest']
        data.drop(
            columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

    def add_all_props(self, new_data):
        self._data = pd.concat([self._data, new_data]).sort_index()
        self._post_process_data()

    def _fill_missing_values(self, data):
        if data.isna().any().any():
            data.ffill().bfill()
            data.fillna(0, inplace=True)
            if data.isna().any().any():
                raise ValueError(
                    "Data contains NaN values even after forward and backward filling.")

    def _set_attributes(self):
        for col in [Columns.Open, Columns.High, Columns.Low, Columns.Close, Columns.Volume]:
            setattr(self, f"_{col.name.lower()}", self._data[col.name])
        self._open_close_diff = self._data.get(
            Columns.Open_Close_Diff.name, pd.Series(dtype=float))

    # Processing methods
    def preprocess_data_for_initial(self):
        columns_to_train = [Columns.High, Columns.Low, Columns.Close]

        X_train, X_test, y_train, y_test = None, None, None, None

        for column in tqdm(columns_to_train, desc="Processing data for initial training"):
            self.target_column = column
            X, y = self.prepare_dataset()

            X_train, X_test, y_train, y_test = self.split_train_test(X, y)
            self.save_last_steps(filename=f'/home/amar/Documents/Projects/MarketPredictor/Datas/BTCUSDT/preprocessed_data/{
                                 column.name}/test/{column.name}_last_steps.csv')

        return X_train, X_test, y_train, y_test

    def preprocess_data_for_training(self):
        return self.prepare_dataset()

    def scale_data(self, data=None):
        if data is None:
            data = self._data
        self.scaler.fit(data)
        return self.scaler.transform(data)

    def calculate_open_close_difference(self):
        if self._data.empty:
            raise ValueError("Data is empty.")
        self._data['Open_Close_Diff'] = self._data['Close'].shift(
            1) - self._data['Open']
        self._data['Open_Close_Diff'].fillna(0, inplace=True)
        self._data['Open_Close_Diff'] = self._data['Open_Close_Diff'].shift(-1)
        self._data.iloc[-1, self._data.columns.get_loc('Open_Close_Diff')] = 0

    def add_columns(self, column_data):
        for column_name, column_values in column_data.items():
            self._data[column_name] = column_values
        return self._data

    def prepare_features(self):
        # print('here')
        if self.target_column is None:
            raise ValueError("Target column not specified.")

        # Separate the target column
        y = self._data[self.target_column.name]
        y.fillna(0, inplace=True)
        
        # Create the feature matrix X by dropping the target column
        expected_str = f'expected_next_{self.target_column.name.lower()}'
        true_diff_str = f'true_{self.target_column.name.lower()}_diff'
        X = self._data.drop(columns=[expected_str, true_diff_str, Columns.Open_Close_Diff.name])
        
        X.fillna(0, inplace=True)
        return X, y

    # def prepare_features(self, data, features, lags):
    #     print(data)
    #     # Create the feature matrix X with lagged values
    #     X = pd.concat([data[feature].shift(lag)
    #                   for feature in features for lag in lags], axis=1)

    #     print(X)

    #     # Forward-fill and backward-fill to handle NaN values
    #     X.ffill(inplace=True)
    #     X.bfill(inplace=True)

    #     # Create the target vector y
    #     y = data[self.target_column.name]

    #     # Ensure X and y have the same length
    #     if y is not None:
    #         X = X.iloc[max(lags):-1]
    #         y = y.iloc[max(lags):-1]

    #     return X, y

    def train_regression_model(self, X, y):
        model = LinearRegression().fit(X, y)
        return model

    def predict_value(self, model, features):
        return model.intercept_ + np.dot(model.coef_, features)

    def analyze_and_predict(self, lags=range(1, 60)):
        target_feature = self.target_column.name
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self._data[f'expected_next_{target_feature.lower()}'] = np.nan
        self._data[f'true_{target_feature.lower()}_diff'] = np.nan

        for i in tqdm(range(len(self._data) - max(lags)), desc=f"Analyzing and predicting {target_feature}"):
            X, y = self.prepare_features()

            model = self.train_regression_model(X, y)

            next_X = self._data.iloc[i][features].values
            predicted_value = self.predict_value(model, next_X)

            self._data.at[self._data.index[i], f'expected_next_{
                target_feature.lower()}'] = predicted_value
            if i < len(self._data) - 1:
                true_next_value = self._data.iloc[i + 1][target_feature]
                self._data.at[self._data.index[i], f'true_{
                    target_feature.lower()}_diff'] = true_next_value - predicted_value

        self.save_predictions(
            self._data.iloc[max(lags):], target_feature.lower())
        return self._data.iloc[max(lags):]

    def analyze_and_predict_low(self, lags=range(1, 60)):
        self.target_column = Columns.Low
        return self.analyze_and_predict(lags)

    def analyze_and_predict_high(self, lags=range(1, 60)):
        self.target_column = Columns.High
        return self.analyze_and_predict(lags)

    def analyze_and_predict_close(self, lags=range(1, 60)):
        self.target_column = Columns.Close
        return self.analyze_and_predict(lags)

    def save_predictions(self, data, target_feature):
        directory = os.path.join(
            BASE_DIR, 'Datas', self.currency, 'preprocessed_data', target_feature)
        os.makedirs(directory, exist_ok=True)

        data.to_csv(
            f'{directory}/{target_feature}_prediction_preprocessed_data.csv', index=True)

    def _calculate_indicator(self, indicator):
        method_name = f"_calculate_{indicator.name.lower()}"
        method = getattr(self, method_name, None)
        if method:
            return method()
        return {}

    def add_technical_indicators(self, data, indicators):
        column_data = {}
        attr = getattr(self, f"_{self.target_column.name.lower()}")

        for indicator in tqdm(indicators, desc="Calculating technical indicators"):
            if indicator in [Indicator.ATR, Indicator.STOCHASTIC_OSCILLATOR, Indicator.DONCHIAN_CHANNEL,
                             Indicator.ADDITIONAL_INDICATORS, Indicator.LAG_FEATURES, Indicator.ROLLING_STATISTICS,
                             Indicator.INTERACTION_FEATURES, Indicator.DIFFERENCING, Indicator.RANGE_FEATURES,
                             Indicator.PREVIOUS_VALUES, Indicator.VOLATILITY] and self.target_column.name in ['High', 'Low']:
                continue
            column_data.update(self._calculate_indicator(indicator))

        data = self.add_columns(column_data)
        print(f"Technical indicators added for {self.target_column.name}.")
        return data

    def add_columns(self, column_data):
        for column_name, column_values in column_data.items():
            self._data[column_name] = column_values
        return self._data.ffill().bfill()

    def add_indicators_to_data(self):
        indicators = [
            Indicator.SMA, Indicator.EMA, Indicator.RSI, Indicator.MACD,
            Indicator.BOLLINGER_BANDS, Indicator.ATR, Indicator.STOCHASTIC_OSCILLATOR,
            Indicator.DONCHIAN_CHANNEL, Indicator.ADDITIONAL_INDICATORS, Indicator.LAG_FEATURES,
            Indicator.ROLLING_STATISTICS, Indicator.INTERACTION_FEATURES, Indicator.DIFFERENCING,
            Indicator.RANGE_FEATURES, Indicator.PREVIOUS_VALUES, Indicator.VOLATILITY
        ]
        self._data = self.add_technical_indicators(self._data, indicators)
        self._data = self._data.ffill().bfill()

    def open_close_diff_features(self):
        y = self._data[Columns.Open_Close_Diff.name]
        X = self._data.drop(columns=[Columns.Open_Close_Diff.name])
        X = X.ffill().bfill()
        y = y.ffill().bfill()
        self._validate_data(X, y)
        return X, y

    def close_price_data_processor(self):
        data = self._data[~self._data.index.duplicated(keep='first')]
        indicators = [
            Indicator.SMA, Indicator.EMA, Indicator.RSI, Indicator.MACD, Indicator.BOLLINGER_BANDS,
            Indicator.ATR, Indicator.STOCHASTIC_OSCILLATOR, Indicator.DONCHIAN_CHANNEL, Indicator.ADDITIONAL_INDICATORS,
            Indicator.LAG_FEATURES, Indicator.ROLLING_STATISTICS, Indicator.INTERACTION_FEATURES, Indicator.DIFFERENCING,
            Indicator.RANGE_FEATURES, Indicator.PREVIOUS_VALUES, Indicator.VOLATILITY
        ]
        data = self.add_technical_indicators(data, indicators)
        y = data[Columns.Close.name]
        data = self.remove_columns(data, [Columns.Close.name])
        data = data.ffill().bfill()
        self._validate_data(data, y)
        return data, y

    def remove_columns(self, data, columns):
        return data.drop(columns=columns, errors='ignore')

    def process_data(self):
        indicators = [
            Indicator.SMA, Indicator.EMA, Indicator.RSI, Indicator.MACD, Indicator.BOLLINGER_BANDS,
            Indicator.ATR, Indicator.STOCHASTIC_OSCILLATOR, Indicator.DONCHIAN_CHANNEL, Indicator.ADDITIONAL_INDICATORS,
            Indicator.LAG_FEATURES, Indicator.ROLLING_STATISTICS, Indicator.INTERACTION_FEATURES, Indicator.DIFFERENCING,
            Indicator.RANGE_FEATURES, Indicator.PREVIOUS_VALUES, Indicator.VOLATILITY
        ]
        data = self._data[~self._data.index.duplicated(keep='first')]
        data = self.add_technical_indicators(data, indicators)
        y = data.pop(
            f'true_{self.target_column.name.lower()}_diff').ffill().bfill()
        data = data.drop(columns=[
                         Columns.Close.name, Columns.Open_Close_Diff.name], errors='ignore').ffill().bfill()
        self._validate_data(data, y)
        return data, y

    def high_low_price_features(self):
        data = self._data[~self._data.index.duplicated(keep='first')]
        data = self.analyze_and_predict()
        return self.process_data(data)

    def split_data_for_high_low(self, data):
        return self.process_data(data)

    def prepare_dataset(self, time_step=60):
        if self.target_column == Columns.Open:
            X, y = self.open_close_diff_features()
        elif self.target_column == Columns.Close:
            X, y = self.close_price_data_processor()
        elif self.target_column in [Columns.High, Columns.Low]:
            X, y = self.high_low_price_features()
        else:
            raise ValueError("Invalid target column specified.")

        if self.processing_type == ProcessingType.TRAINING:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
            y = self.scaler.transform(y.values.reshape(-1, 1)).flatten()

        X_seq, y_seq = self.create_sequences(
            pd.DataFrame(X), pd.Series(y), time_step)

        return (self.split_train_test(X_seq, y_seq) if self.processing_type == ProcessingType.TRAINING else (X_seq, y_seq))

    def create_sequences(self, data_X, data_y, time_step=60):
        X, y = [], []
        data_X_values = data_X.values
        data_y_values = data_y.values
        for i in tqdm(range(len(data_X_values) - time_step), desc="Creating sequences"):
            X.append(data_X_values[i:i + time_step])
            y.append(data_y_values[i + time_step - 1])
        if not any(np.array_equal(data_X_values[0], seq[0]) for seq in X):
            print(
                "Top of the data is not included in the sequences due to insufficient length.")
        print(f"Successfully created sequences. X shape: {
              np.array(X).shape}, y shape: {np.array(y).shape}")
        return np.array(X), np.array(y)

    def _validate_data(self, X, y):
        assert not X.isna().any().any(), "Features contain NaN values after preprocessing"
        assert not y.isna().any(), "Target contains NaN values after preprocessing"
        assert len(X) == len(y), "Mismatch in length between X and y"
        if X.shape[0] < 60:
            raise ValueError(
                "Not enough data after preprocessing to create sequences.")

    def split_train_test(self, X, y, split_ratio=0.8):
        split = int(split_ratio * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        return X_train, X_test, y_train, y_test

    def save_last_steps(self, steps=120, filename='last_steps.csv'):
        data = self._data[-steps:]
        data.to_csv(filename, index=True)

    def output_decoder(self, predictions, data, column_name):
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        print(f"Column name: {column_name}")
        print(f"Predictions (scaled): {predictions}")
        placeholder = np.zeros((predictions.shape[0], data.shape[1]))
        placeholder[:, -1] = predictions.flatten()
        try:
            predicted_prices = self.scaler.inverse_transform(placeholder)[
                :, -1]
            return predicted_prices
        except Exception as e:
            print(f"Error during inverse transformation: {e}")
            return None

    def save_scaler(self, scaler_filename):
        with open(scaler_filename, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_scaler(self, scaler_filename):
        with open(scaler_filename, 'rb') as f:
            self.scaler = pickle.load(f)

    def _calculate_sma(self):
        attr = getattr(self, f"_{self.target_column.name.lower()}")
        return {
            f'SMA_20_{self.target_column.name}': attr.rolling(window=20).mean(),
            f'SMA_50_{self.target_column.name}': attr.rolling(window=50).mean()
        }

    def _calculate_ema(self):
        attr = getattr(self, f"_{self.target_column.name.lower()}")
        return {
            f'EMA_20_{self.target_column.name}': attr.ewm(span=20, adjust=False).mean(),
            f'EMA_50_{self.target_column.name}': attr.ewm(span=50, adjust=False).mean()
        }

    def _calculate_rsi(self):
        attr = getattr(self, f"_{self.target_column.name.lower()}")
        return {
            f'RSI_{self.target_column.name}': ta.momentum.RSIIndicator(attr, window=14).rsi()
        }

    def _calculate_macd(self):
        attr = getattr(self, f"_{self.target_column.name.lower()}")
        macd = ta.trend.MACD(attr)
        return {
            f'MACD_{self.target_column.name}': macd.macd(),
            f'MACD_signal_{self.target_column.name}': macd.macd_signal(),
            f'MACD_diff_{self.target_column.name}': macd.macd_diff()
        }

    def _calculate_bollinger_bands(self):
        attr = getattr(self, f"_{self.target_column.name.lower()}")
        bbands = ta.volatility.BollingerBands(attr)
        return {
            f'BB_upper_{self.target_column.name}': bbands.bollinger_hband(),
            f'BB_middle_{self.target_column.name}': bbands.bollinger_mavg(),
            f'BB_lower_{self.target_column.name}': bbands.bollinger_lband()
        }

    def _calculate_atr(self):
        high = getattr(self, f"_{Columns.High.name.lower()}")
        low = getattr(self, f"_{Columns.Low.name.lower()}")
        close = getattr(self, f"_{Columns.Close.name.lower()}")
        return {
            f'ATR_{self.target_column.name}': ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        }

    def _calculate_stochastic_oscillator(self):
        high = getattr(self, f"_{Columns.High.name.lower()}")
        low = getattr(self, f"_{Columns.Low.name.lower()}")
        close = getattr(self, f"_{Columns.Close.name.lower()}")
        stoch = ta.momentum.StochasticOscillator(
            high, low, close, window=14, smooth_window=3)
        return {
            f'Stoch_{self.target_column.name}': stoch.stoch(),
            f'Stoch_signal_{self.target_column.name}': stoch.stoch_signal()
        }

    def _calculate_donchian_channel(self):
        high = getattr(self, f"_{Columns.High.name.lower()}")
        low = getattr(self, f"_{Columns.Low.name.lower()}")
        close = getattr(self, f"_{Columns.Close.name.lower()}")
        donchian = ta.volatility.DonchianChannel(high, low, close, window=20)
        return {
            f'Donchian_upper_{self.target_column.name}': donchian.donchian_channel_hband(),
            f'Donchian_lower_{self.target_column.name}': donchian.donchian_channel_lband()
        }

    def _calculate_additional_indicators(self):
        high = getattr(self, f"_{Columns.High.name.lower()}")
        low = getattr(self, f"_{Columns.Low.name.lower()}")
        close = getattr(self, f"_{Columns.Close.name.lower()}")
        volume = getattr(self, f"_{Columns.Volume.name.lower()}")
        return {
            f'Williams_%R_{self.target_column.name}': ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r(),
            f'CMF_{self.target_column.name}': ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow(),
            f'MFI_{self.target_column.name}': ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index(),
            f'OBV_{self.target_column.name}': ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        }

    def _calculate_lag_features(self):
        attr = getattr(self, f"_{self.target_column.name.lower()}")
        return {f'Close_lag_{lag}_{self.target_column.name}': attr.shift(lag) for lag in range(1, 6)}

    def _calculate_rolling_statistics(self):
        attr = getattr(self, f"_{self.target_column.name.lower()}")
        return {
            f'Rolling_mean_20_{self.target_column.name}': attr.rolling(window=20).mean(),
            f'Rolling_std_20_{self.target_column.name}': attr.rolling(window=20).std()
        }

    def _calculate_interaction_features(self):
        volume = getattr(self, f"_{Columns.Volume.name.lower()}")
        close = getattr(self, f"_{Columns.Close.name.lower()}")
        return {f'Volume_Close_{self.target_column.name}': volume * close}

    def _calculate_differencing(self):
        attr = getattr(self, f"_{self.target_column.name.lower()}")
        return {f'Close_diff_{self.target_column.name}': attr.diff()}

    def _calculate_range_features(self):
        high = getattr(self, f"_{Columns.High.name.lower()}")
        low = getattr(self, f"_{Columns.Low.name.lower()}")
        open_ = getattr(self, f"_{Columns.Open.name.lower()}")
        close = getattr(self, f"_{Columns.Close.name.lower()}")
        return {
            f'High_Low_Range_{self.target_column.name}': high - low,
            f'Open_Close_Range_{self.target_column.name}': open_ - close
        }

    def _calculate_previous_values(self):
        high = getattr(self, f"_{Columns.High.name.lower()}")
        low = getattr(self, f"_{Columns.Low.name.lower()}")
        attr = getattr(self, f"_{self.target_column.name.lower()}")
        return {
            f'Previous_High_{self.target_column.name}': high.shift(1),
            f'Previous_Low_{self.target_column.name}': low.shift(1),
            f'Previous_Close_{self.target_column.name}': attr.shift(1)
        }

    def _calculate_volatility(self):
        attr = getattr(self, f"_{self.target_column.name.lower()}")
        return {f'Volatility_{self.target_column.name}': attr.rolling(window=max(2, int(len(attr) // 10))).std()}
