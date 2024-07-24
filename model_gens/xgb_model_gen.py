import xgboost as xgb # type: ignore
import numpy as np
import logging
import pickle
from model_gens.market_predictor_base import MarketPredictorBase  # Make sure the path is correct
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class XGBModel(MarketPredictorBase):
    def __init__(self):
        """Initializes the XGB model class, inheriting from MarketPredictorBase."""
        super().__init__()
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_shape = None

    @staticmethod
    def create_model():
        """Creates an XGBoost model.

        Returns:
            XGBRegressor: The created XGBoost model.
        """
        model = xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=10000, learning_rate=0.01)
        return model

    def train_model(self, X_train, y_train, X_val, y_val):
        """Trains the XGBoost model.

        Args:
            X_train (array): The training input data.
            y_train (array): The training target data.
            X_val (array): The validation input data.
            y_val (array): The validation target data.
        """
        # Save the feature shape after reshaping
        self.feature_shape = X_train.shape[1]

        # Creating and training the model
        self.model = self.create_model()
        self.model.fit(X_train, y_train, early_stopping_rounds=1000,
                       eval_set=[(X_val, y_val)], verbose=True)

        logging.info('XGBoost Model trained.')
        return self.model

    def make_predictions(self, X_test):
        print("Original X_test shape:", X_test.shape)  # For debugging

        # If the model was trained on flattened data
        if len(X_test.shape) == 3:
            X_test = X_test.reshape((X_test.shape[0], -1))

        # Applying scaler transform if necessary
        X_test = self.scaler.transform(X_test)

        predictions = self.model.predict(X_test)
        return self.output_decoder(predictions)

    def prepare_latest_data_and_predict(self, latest_data_path, n_steps=10, eval=False):
        """
        Prepares the latest data and makes future predictions.

        Args:
            latest_data_path (str): The path to the CSV file with the latest data.
            past_data_path (str): The path to the CSV file with the past data.
            n_steps (int): Number of steps to predict into the future.
            eval (bool): If True, evaluate the model on the latest data.

        Returns:
            tuple: Latest preprocessed data and DataFrame with predictions.
        """
        latest_data = self.load_data(latest_data_path)
        latest_data = self.preprocess_data(latest_data)

        if eval:
            X, y = self.prepare_xgb_dataset(latest_data)
            self.evaluate_model(X, y)
            return latest_data, None

        scaled_data = self.scaler.transform(latest_data)

        # remove the Close price column
        scaled_data = np.delete(scaled_data, 3, axis=1)

        initial_input = scaled_data[-60:]

        predictions = self.model.predict(initial_input.reshape(
            initial_input.shape[0], initial_input.shape[1]))

        predictions = np.array(predictions).reshape(-1, 1)
        decoded_predictions = self.output_decoder(predictions, latest_data)

        last_index = pd.to_datetime(latest_data.index[-1])
        predicted_dates = pd.date_range(
            start=last_index + pd.Timedelta(minutes=5), periods=len(decoded_predictions), freq='5T')
        predicted_df = pd.DataFrame(
            decoded_predictions, index=predicted_dates, columns=['Predicted Close'])

        return predicted_df, latest_data

    def prepare_xgb_dataset(self, data, target_index=3):
        """Prepares the dataset for training by scaling and separating features and target.

        Args:
            data (DataFrame): The preprocessed market data.
            target_index (int): The index of the target variable in the dataset.

        Returns:
            tuple: Input features (X) and target values (y).
        """
        # print("Original data shape:", data)  # For debugging
        scaled_data = self.scaler.fit_transform(data)
        X = scaled_data[:, np.arange(scaled_data.shape[1]) != target_index]
        y = scaled_data[:, target_index]
        return X, y

    def output_decoder(self, predictions, data):
        """Decodes the predictions using the inverse transform of the scaler.

        Args:
            predictions (array): Predicted values to be decoded.

        Returns:
            array: Decoded predicted values.
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been initialized.")
        if self.feature_shape is None:
            raise ValueError("Feature shape has not been set.")

        placeholder = np.zeros((predictions.shape[0], data.shape[1]))
        placeholder[:, 3] = predictions.flatten()

        predicted_prices = self.scaler.inverse_transform(placeholder)[:, 3]
        return predicted_prices

    def load_xgb_model(self, filename='xgb_model.json'):
        """Loads a model from a file.

        Args:
            filename (str): The filename to load the model from.
        """
        self.model = xgb.XGBRegressor()
        self.model.load_model(filename)
        logging.info(f'Model loaded from {filename}.')

    def save_xgb_model(self, filename='models/XAUUSD_model_xgb'):
        """Saves the trained model to a file.

        Args:
            filename (str): The filename to save the model.
        """
        self.model.save_model(f'{filename}.json')
        logging.info(f'Model saved to {filename}.')

    def save_scaler_and_shape(self, scaler_filename='scaler.pkl', shape_filename='feature_shape.pkl'):
        """Saves the scaler and feature shape to files.

        Args:
            scaler_filename (str): The filename to save the scaler.
            shape_filename (str): The filename to save the feature shape.
        """
        with open(scaler_filename, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(shape_filename, 'wb') as f:
            pickle.dump(self.feature_shape, f)
        logging.info(f'Scaler and feature shape saved to {scaler_filename} and {shape_filename}.')

    def load_scaler_and_shape(self, scaler_filename='scaler.pkl', shape_filename='feature_shape.pkl'):
        """Loads the scaler and feature shape from files.

        Args:
            scaler_filename (str): The filename to load the scaler from.
            shape_filename (str): The filename to load the feature shape from.
        """
        with open(scaler_filename, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(shape_filename, 'rb') as f:
            self.feature_shape = pickle.load(f)
        logging.info(f'Scaler and feature shape loaded from {scaler_filename} and {shape_filename}.')
