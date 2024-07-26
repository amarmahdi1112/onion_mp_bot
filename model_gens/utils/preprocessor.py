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
from retrying import retry
import logging

class ValidationError(Exception):
    pass
class DataProcessingError(Exception):
    """Custom exception for errors during data processing."""
    pass
class InvalidProcessingTypeError(Exception):
    """Exception raised when an invalid processing type is encountered."""
    pass

class InvalidTargetColumnError(Exception):
    """Exception raised when an invalid target column is encountered."""
    pass

class UnexpectedError(Exception):
    """Exception raised for any unexpected errors."""
    pass
class DataLoadingError(Exception):
    """Exception raised when there is an error loading the data."""
    pass

class ModelTrainingError(Exception):
    """Exception raised during model training."""
    pass

class Preprocessor:
    def __init__(self, path=None, target_column=None, processing_type=ProcessingType.INITIAL, load_csv=True):
        self.path = path
        self.currency = 'BTCUSD'
        self.processing_type = processing_type
        self.target_column = target_column
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self._data = pd.DataFrame()
        self._open_close_diff = pd.Series(dtype=float)
        self._load_csv = load_csv
        self.added_columns = []  # List to keep track of added columns

        if load_csv:
            self._load_data()
            
    def _load_data(self):
            logging.info(f"Loading data for {self.processing_type.name.lower()}...")
            try:
                print('hello')
                self._validate_path()
                if self.processing_type == ProcessingType.INITIAL:
                    self._load_data_file()
                elif self.processing_type == ProcessingType.TRAINING or self.processing_type == ProcessingType.PREDICTION:
                    if not self.target_column:
                        raise ValidationError("Target column not specified.")
                    self._load_data_file()
            except ValidationError as e:
                logging.error(f"Validation error: {e}")
            except FileNotFoundError as e:
                logging.error(f"File not found: {e}")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")

    def _validate_path(self):
        if self.processing_type == ProcessingType.INITIAL and not os.path.isfile(self.path):
            raise ValidationError(f"Path {self.path} is not a valid file.")
        elif self.processing_type == ProcessingType.TRAINING and not os.path.isdir(self.path):
            raise ValidationError(f"Path {self.path} is not a valid directory.")

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def _load_data_file(self):
        print(f"Reading data from {self.path}...")
        try:
            data = None
            if self.processing_type == ProcessingType.INITIAL:
                data = pd.read_csv(self.path, parse_dates=True, index_col=0)
            elif self.processing_type == ProcessingType.TRAINING:
                # Consolidate file path construction and reading
                if self.target_column in [Columns.Close, Columns.High, Columns.Low, Columns.Volume]:
                    file_path = f'{self.path}/{self.target_column.name.lower()}/{self.target_column.name.lower()}_prediction_preprocessed_data.csv'
                    print(f"Reading {self.target_column.name.lower()} data... {file_path}")
                    data = pd.read_csv(file_path, parse_dates=True, index_col=0)
                else:
                    raise ValueError(f"Unsupported target column: {self.target_column}")
            elif self.processing_type == ProcessingType.PREDICTION:
                data = pd.read_csv(self.path, parse_dates=True, index_col=0)
            else:
                raise ValueError(f"Unsupported processing type: {self.processing_type}")

            if data is not None:
                self._post_process_data(data)
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

    def _post_process_data(self, data=None):
        if data is None:
            data = self._data

        try:
            self._drop_unnecessary_columns(data)
        except Exception as e:
            self._log_error("_drop_unnecessary_columns", e)
            raise DataProcessingError("Error in dropping unnecessary columns") from e

        try:
            self._fill_missing_values(data)
        except Exception as e:
            self._log_error("_fill_missing_values", e)
            raise DataProcessingError("Error in filling missing values") from e

        try:
            self._ensure_required_columns(data)
        except Exception as e:
            self._log_error("_ensure_required_columns", e)
            raise DataProcessingError("Error in ensuring required columns") from e

        self._data = data
        self._set_attributes()

    def _log_error(self, step, exception):
        # Implement logging mechanism here
        # Example: logging.error(f"Error during {step}: {exception}")
        print(f"Error during {step}: {exception}")
        
    def _ensure_required_columns(self, data):
        if data is None:
            raise ValueError("Data is empty.")
        required_columns = self._get_required_columns()
        missing = set(required_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _drop_unnecessary_columns(self, data):
        if data is None:
            raise ValueError("Data is empty.")
        columns_to_drop = ["Change", 'Open Interest']
        data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

    def _fill_missing_values(self, data):
        if data is None:
            raise ValueError("Data is empty.")
        # Apply fill methods in a chain and ensure they are applied by assigning the result
        # data.fillna(method='ffill', inplace=True)
        # data.fillna(method='bfill', inplace=True)
        # data.fillna(0, inplace=True)
        # use ffill and bfill to fill missing values
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        # Ensure there are no NaN values after filling
        if data.isna().any().any():
            raise ValueError("Data contains NaN values even after filling.")

    def _set_attributes(self):
        if self._data.empty:
            raise ValueError("Data is empty.")
        # Assuming this method sets class attributes based on column names
        for col in [Columns.Open, Columns.High, Columns.Low, Columns.Close, Columns.Volume]:
            setattr(self, f"_{col.name.lower()}", self._data[col.name].copy())
        self._open_close_diff = self._data.get(Columns.Open_Close_Diff.name, pd.Series(dtype=float)).copy()

    def _get_required_columns(self):
        try:
            # Validate processing_type
            if not isinstance(self.processing_type, ProcessingType):
                raise InvalidProcessingTypeError(f"Invalid processing type: {self.processing_type}")

            # Base required columns
            base_required_columns = [
                Columns.Open.name, 
                Columns.High.name, 
                Columns.Low.name, 
                Columns.Close.name, 
                Columns.Volume.name
            ]

            # Additional columns based on the target column
            additional_columns = {
                Columns.Open: [Columns.Open_Close_Diff.name],
                Columns.Close: ['expected_next_close', 'true_close_diff'],
                Columns.High: ['expected_next_high', 'true_high_diff'],
                Columns.Low: ['expected_next_low', 'true_low_diff'],
            }

            # Determine required columns based on processing type
            if self.processing_type == ProcessingType.INITIAL:
                return base_required_columns
            elif self.processing_type == ProcessingType.TRAINING or self.processing_type == ProcessingType.PREDICTION:
                # Validate target_column
                if self.target_column not in Columns:
                    raise InvalidTargetColumnError(f"Invalid target column: {self.target_column}")
                specific_columns = additional_columns.get(self.target_column, [])
                return base_required_columns + specific_columns
        except (InvalidProcessingTypeError, InvalidTargetColumnError) as e:
            # Log and re-raise specific known errors
            print(f"Error: {e}")  # Replace with actual logging
            raise
        except Exception as e:
            # Log and re-raise unexpected errors
            print(f"Unexpected error: {e}")  # Replace with actual logging
            raise UnexpectedError("An unexpected error occurred") from e
        
    def prepare_multi_sequence_datasets(self):
        fibonacci_steps = [7, 21, 49]
        X_train_all, X_test_all = [], []
        y_train_list, y_test_list = [], []
        X, y = None, None

        for steps in fibonacci_steps:
            if self.processing_type == ProcessingType.TRAINING:
                X_train, X_test, y_train, y_test = self.prepare_dataset(time_step=steps)
                print(f"Steps: {steps}, X_train: {len(X_train)}, X_test: {len(X_test)}, y_train: {len(y_train)}, y_test: {len(y_test)}")
                X_train_all.append(X_train)
                X_test_all.append(X_test)
                y_train_list.append(y_train)
                y_test_list.append(y_test)
            else:
                X, y = self.prepare_dataset(time_step=steps)
                print(f"Steps: {steps}, X: {len(X)}, y: {len(y)}")
                X_train_all.append(X)
                y_train_list.append(y)

        if self.processing_type == ProcessingType.TRAINING:
            # Ensure all sequences have the same length by checking minimum length
            min_train_length = min(len(y) for y in y_train_list)
            min_test_length = min(len(y) for y in y_test_list)

            print(f"Minimum training length: {min_train_length}, Minimum testing length: {min_test_length}")

            # Adjust all sequences to the minimum length without trimming targets
            X_train_all = [X[-min_train_length:] for X in X_train_all]
            X_test_all = [X[-min_test_length:] for X in X_test_all]
            y_train = y_train_list[0][-min_train_length:]
            y_test = y_test_list[0][-min_test_length:]

            return X_train_all, X_test_all, y_train, y_test
        else:
            # For prediction, ensure the sequences have the same length
            min_length = min(len(y) for y in y_train_list)
            X_train_all = [X[-min_length:] for X in X_train_all]
            y_train = y_train_list[0][-min_length:]

            return X_train_all, y_train

    def preprocess_data_for_initial(self):
        columns_to_train = [Columns.High, Columns.Low, Columns.Close, Columns.Volume]

        for column in columns_to_train:
            try:
                self.target_column = column
                predictor = PricePredictor(self._data, column)
                predictor.train_model()
                predictor.analyze_and_predict()
                self.save_predictions(predictor.data, self.target_column.name.lower())
                predictor.save_model(
                    f'{BASE_DIR}/Datas/BTCUSD/preprocessed_data/{column.name}/{column.name}_predictor.pkl')
                
                self.save_last_steps(filename=f'{BASE_DIR}/Datas/BTCUSD/preprocessed_data/{column.name}/test/{column.name}_last_steps.csv')
            except DataLoadingError as e:
                # Handle data loading errors
                print(f"Failed to load data for {column.name}: {e}")
                # Optionally, log the error and continue with the next column
                continue
            except ModelTrainingError as e:
                # Handle model training errors
                print(f"Failed to train model for {column.name}: {e}")
                # Optionally, log the error and continue with the next column
                continue
            except Exception as e:
                # Handle other unexpected errors
                print(f"An unexpected error occurred for {column.name}: {e}")
                # Depending on the severity, you might want to halt further processing or just log and continue
                continue
    def save_predictions(self, data, target_feature):
        directory = os.path.join(BASE_DIR, 'Datas', 'BTCUSD', 'preprocessed_data', target_feature)
        
        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the data
        file_path = os.path.join(directory, f'{target_feature}_prediction_preprocessed_data.csv')
        data.to_csv(file_path, index=True)

        print(f"Predictions saved to {file_path}")

    def preprocess_data_for_training(self):
        return self.prepare_dataset()
    
    def calculate_open_close_difference(self):
        try:
            if self._data.empty:
                raise ValueError("Data is empty.")
            self._data['Open_Close_Diff'] = self._data['Close'].shift(1) - self._data['Open']
            self._data['Open_Close_Diff'].fillna(0, inplace=True)
            self._data['Open_Close_Diff'] = self._data['Open_Close_Diff'].shift(-1)
            self._data.iloc[-1, self._data.columns.get_loc('Open_Close_Diff')] = 0
        except KeyError as e:
            # Handle missing 'Close' or 'Open' column
            raise KeyError(f"Missing column in data: {e}")
        except TypeError as e:
            # Handle non-numeric types in 'Close' or 'Open' columns
            raise TypeError(f"Non-numeric types in 'Close' or 'Open' columns: {e}")
        except IndexError as e:
            # This is unlikely due to the initial empty check, but included for completeness
            raise IndexError(f"Index error encountered: {e}")
        except Exception as e:
            # Catch-all for any other exceptions not explicitly handled above
            raise Exception(f"An unexpected error occurred: {e}")
        
    def add_columns(self, column_data):
        for column_name, column_values in column_data.items():
            self._data[column_name] = column_values
        return self._data
    
    def prepare_features(self):
        try:
            if self.target_column is None:
                raise ValueError("Target column not specified.")

            true_diff_str = f'true_{self.target_column.name.lower()}_diff'
            # Separate the target column
            y = self._data[true_diff_str]
            y.fillna(0, inplace=True)
            
            # Assuming Columns.Open_Close_Diff.name is a valid column name, if not, KeyError will be raised
            X = self._data.drop(columns=[true_diff_str, Columns.Open_Close_Diff.name])
            
            X.fillna(0, inplace=True)
            return X, y
        except AttributeError as e:
            # Handle missing 'name' attribute or None target_column
            raise AttributeError(f"Attribute error encountered: {e}")
        except KeyError as e:
            # Handle missing columns in self._data
            raise KeyError(f"Missing column in data: {e}")
        except Exception as e:
            # Catch-all for any other exceptions not explicitly handled above
            raise Exception(f"An unexpected error occurred: {e}")
        
    def train_regression_model(self, X, y):
        model = LinearRegression().fit(X, y)
        return model

    def predict_value(self, model, features):
        return model.intercept_ + np.dot(model.coef_, features)
    
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
            if indicator in [Indicator.STOCHASTIC_OSCILLATOR, Indicator.ADDITIONAL_INDICATORS, 
                             Indicator.LAG_FEATURES, Indicator.ROLLING_STATISTICS, Indicator.INTERACTION_FEATURES, 
                             Indicator.DIFFERENCING, Indicator.RANGE_FEATURES, Indicator.PREVIOUS_VALUES] and self.target_column.name in ['High', 'Low', 'Close', 'Volume']:
                continue
            
            column_data.update(self._calculate_indicator(indicator))

        data = self.add_columns(column_data)
        print(f"Technical indicators added for {self.target_column.name}.")
        return data

    def add_columns(self, column_data):
        for column_name, column_values in column_data.items():
            self._data[column_name] = column_values
            self.added_columns.append(column_name)
        return self._data.ffill().bfill()

    def remove_added_columns(self):
        # Remove columns listed in self.added_columns from self._data
        self._data.drop(columns=self.added_columns, inplace=True)
        # Clear the added_columns list
        self.added_columns = []


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
            Indicator.RANGE_FEATURES,  Indicator.VOLATILITY
        ]
        data = self.add_technical_indicators(data, indicators)
        target_column_name = 'true_close_diff'
        y = data[target_column_name]
        data = self.remove_columns(data, [target_column_name])
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

    def prepare_dataset(self, time_step=60):
        if self.target_column == Columns.Open:
            X, y = self.open_close_diff_features()
        elif self.target_column in [Columns.High, Columns.Low, Columns.Close, Columns.Volume]:
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

    def save_last_steps(self, steps=61, filename='last_steps.csv'):
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

    # def save_scaler(self, scaler_filename):
    #     with open(scaler_filename, 'wb') as f:
    #         pickle.dump(self.scaler_X, f)
    #         pickle.dump(self.scaler_y, f)

    # save the scalers separately on an separate files
    def save_scaler(self, scaler_filename):
        # Extract directory from filename
        directory = os.path.dirname(scaler_filename)
        
        directory_X = os.path.join(directory, '_X')
        directory_y = os.path.join(directory, '_y')
        
        # Create directory if it doesn't exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        if directory_X and not os.path.exists(directory_X):
            os.makedirs(directory_X)
        
        if directory_y and not os.path.exists(directory_y):
            os.makedirs(directory_y)
        
        # Save the scalers
        with open(os.path.join(directory_X, 'scaler_X.pkl'), 'wb') as f:
            pickle.dump(self.scaler_X, f)
        
        with open(os.path.join(directory_y, 'scaler_y.pkl'), 'wb') as f:
            pickle.dump(self.scaler_y, f)

    def load_scaler(self, scaler_filename):
        # Extract directory from filename
        directory = os.path.dirname(scaler_filename)
        
        directory_X = os.path.join(directory, '_X')
        directory_y = os.path.join(directory, '_y')
        
        # Load the scalers
        with open(os.path.join(directory_X, 'scaler_X.pkl'), 'rb') as f:
            self.scaler_X = pickle.load(f)
        
        with open(os.path.join(directory_y, 'scaler_y.pkl'), 'rb') as f:
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
