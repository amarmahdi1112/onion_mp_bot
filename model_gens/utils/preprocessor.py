from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import pandas as pd
import ta # type: ignore
from enum import Enum
import glob
import os
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from settings import BASE_DIR
from model_gens.utils.static.columns import Columns
from tqdm import tqdm # type: ignore
from model_gens.utils.static.processing_type import ProcessingType
from model_gens.utils.static.indicator import Indicator
from model_gens.utils.price_predictor import PricePredictor


class Preprocessor:
    def __init__(self, path=None, target_column=None, processing_type=ProcessingType.INITIAL, load_csv=True):
        self.path = path
        self.currency = 'BTCUSDT'
        self.processing_type = processing_type
        self.target_column = target_column
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self._data = pd.DataFrame()
        self._open_close_diff = pd.Series(dtype=float)
        self._load_csv = load_csv
        if load_csv:
            self._load_data()

    def _load_data(self):
        print(f"Loading data for {self.processing_type.name.lower()}...")
        # Validate path first
        self._validate_path()
        # Load data based on processing type
        if self.processing_type == ProcessingType.INITIAL:
            self._load_data_file()
        elif self.processing_type == ProcessingType.TRAINING:
            if not self.target_column:
                raise ValueError("Target column not specified.")
            self._load_data_file()

    def _validate_path(self):
        if self.processing_type == ProcessingType.INITIAL and not os.path.isfile(self.path):
            raise ValueError(f"Path {self.path} is not a valid file.")
        elif self.processing_type == ProcessingType.TRAINING and not os.path.isdir(self.path):
            raise ValueError(f"Path {self.path} is not a valid directory.")
        
    def _load_data_file(self):
        print(f"Reading data from {self.path}...")
        try:
            data = None
            if self.processing_type == ProcessingType.INITIAL:
                data = pd.read_csv(self.path, parse_dates=True, index_col=0)
            elif self.processing_type == ProcessingType.TRAINING:
                # Consolidate file path construction and reading
                if self.target_column in [Columns.Open, Columns.Close, Columns.High, Columns.Low]:
                    file_path = f'{self.path}/{self.target_column.name.lower()}/{self.target_column.name.lower()}_prediction_preprocessed_data.csv'
                    print(f"Reading {self.target_column.name.lower()} data... {file_path}")
                    data = pd.read_csv(file_path, parse_dates=True, index_col=0)
                else:
                    raise ValueError(f"Unsupported target column: {self.target_column}")
            else:
                raise ValueError(f"Unsupported processing type: {self.processing_type}")

            if data is not None:
                self._post_process_data(data)
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

    def _post_process_data(self, data=None):
        if data is None:
            data = self._data

        self._drop_unnecessary_columns(data)
        self._fill_missing_values(data)
        self._ensure_required_columns(data)
        self._data = data
        self._set_attributes()

    def _ensure_required_columns(self, data):
        required_columns = self._get_required_columns()
        missing = set(required_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _drop_unnecessary_columns(self, data):
        columns_to_drop = ["Change", 'Open Interest']
        data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

    def _fill_missing_values(self, data):
        # Apply fill methods in a chain and ensure they are applied by assigning the result
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        data.fillna(0, inplace=True)
        if data.isna().any().any():
            raise ValueError("Data contains NaN values even after filling.")

    def _set_attributes(self):
        # Assuming this method sets class attributes based on column names
        for col in [Columns.Open, Columns.High, Columns.Low, Columns.Close, Columns.Volume]:
            setattr(self, f"_{col.name.lower()}", self._data[col.name].copy())
        self._open_close_diff = self._data.get(Columns.Open_Close_Diff.name, pd.Series(dtype=float)).copy()

    def _get_required_columns(self):
        # This method abstracts away the logic for determining required columns
        base_required_columns = [
            Columns.Open.name, 
            Columns.High.name, 
            Columns.Low.name, 
            Columns.Close.name, 
            Columns.Volume.name
        ]
        additional_columns = {
            Columns.Open: [Columns.Open_Close_Diff.name],
            Columns.Close: ['expected_next_close', 'true_close_diff'],
            Columns.High: ['expected_next_high', 'true_high_diff'],
            Columns.Low: ['expected_next_low', 'true_low_diff'],
        }
        if self.processing_type == ProcessingType.INITIAL:
            return base_required_columns
        elif self.processing_type == ProcessingType.TRAINING:
            specific_columns = additional_columns.get(self.target_column, [])
            return base_required_columns + specific_columns
        
    # Processing methods
    def preprocess_data_for_initial(self):
        columns_to_train = [Columns.High, Columns.Low, Columns.Close, Columns.Volume]

        for column in columns_to_train:
            self.target_column = column
            predictor = PricePredictor(self._data, column)
            predictor.train_model()
            predictor.analyze_and_predict()
            self.save_predictions(predictor.data, self.target_column.name.lower())
            predictor.save_model(
                f'{BASE_DIR}/Datas/BTCUSDT/preprocessed_data/{column.name}/{column.name}_predictor.pkl')
            
            self.save_last_steps(filename=f'{BASE_DIR}/Datas/BTCUSDT/preprocessed_data/{column.name}/test/{column.name}_last_steps.csv')

    def preprocess_data_for_training(self):
        return self.prepare_dataset()

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

        true_diff_str = f'true_{self.target_column.name.lower()}_diff'
        # Separate the target column
        y = self._data[true_diff_str]
        y.fillna(0, inplace=True)
        
        X = self._data.drop(columns=[true_diff_str, Columns.Open_Close_Diff.name])
        
        X.fillna(0, inplace=True)
        return X, y

    def train_regression_model(self, X, y):
        model = LinearRegression().fit(X, y)
        return model

    def predict_value(self, model, features):
        return model.intercept_ + np.dot(model.coef_, features)

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
        X = data.ffill().bfill()
        self._validate_data(X, y)
        return X, y

    def price_features(self):
        return self.process_data()

    def split_data_for_high_low(self, data):
        return self.process_data()
    
    def prepare_dataset(self, time_step=60):
        if self.target_column == Columns.Open:
            X, y = self.open_close_diff_features()
        elif self.target_column in [Columns.High, Columns.Low, Columns.Close]:
            X, y = self.price_features()
            print(f"X columns: {X.columns}, y shape: {y.shape}")
        else:
            raise ValueError("Invalid target column specified.")

        if self.processing_type == ProcessingType.TRAINING:
            self.scaler_X.fit(X)
            X_scaled = self.scaler_X.transform(X)
            self.scaler_y.fit(y.values.reshape(-1, 1))
            y_scaled = self.scaler_y.transform(y.values.reshape(-1, 1)).flatten()
        else:
            X_scaled = self.scaler_X.transform(X)
            y_scaled = self.scaler_y.transform(y.values.reshape(-1, 1)).flatten()

        X_seq, y_seq = self.create_sequences(pd.DataFrame(X_scaled), pd.Series(y_scaled), time_step)
        return self.split_train_test(X_seq, y_seq) if self.processing_type == ProcessingType.TRAINING else (X_seq, y_seq)


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
        print(f"Successfully created sequences. X shape: {np.array(X).shape}, y shape: {np.array(y).shape}")
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
        # Extract directory from filename
        directory = os.path.dirname(filename)

        # Create directory if it doesn't exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save the data
        data = self._data[-steps:]
        data.to_csv(filename, index=True)
        
    def output_decoder(self, predictions, data, column_name):
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        
        print(f"Column name: {column_name}")
        print(f"Predictions (scaled): {predictions}")
        
        # Use the scaler_y to inverse transform the predictions
        try:
            predicted_prices = self.scaler_y.inverse_transform(predictions).flatten()
            return predicted_prices
        except Exception as e:
            print(f"Error during inverse transformation: {e}")
            return None

    def save_scaler(self, scaler_filename):
        with open(scaler_filename, 'wb') as f:
            pickle.dump(self.scaler_X, f)
            pickle.dump(self.scaler_y, f)

    def load_scaler(self, scaler_filename):
        with open(scaler_filename, 'rb') as f:
            self.scaler_X = pickle.load(f)
            self.scaler_y = pickle.load(f)

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
