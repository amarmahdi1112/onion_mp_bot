import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import logging
from sklearn.preprocessing import MinMaxScaler
import pickle


class MetaModel:
    def __init__(self):
        """Initializes the MetaModel class."""
        self.meta_model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_shape = None
        self.lstm_model = None
        self.gru_model = None

    def create_model(self):
        """Creates a Gradient Boosting Regressor model.

        Returns:
            GradientBoostingRegressor: The created Gradient Boosting Regressor model.
        """
        self.meta_model = GradientBoostingRegressor(
            n_estimators=1000, learning_rate=0.01)
        return self.meta_model

    def train_model(self, X_meta_train, y_meta_train):
        """Trains the meta-model.

        Args:
            X_meta_train (array): Meta features for training.
            y_meta_train (array): Meta targets for training.
        """
        self.meta_model = self.create_model()
        self.meta_model.fit(X_meta_train, y_meta_train)
        logging.info('Meta Model trained.')
        return self.meta_model

    def generate_meta_features(self, lstm_model, gru_model, X, gru_scaler=None, lstm_scaler=None):
        """
        Generates meta-features from the predictions of LSTM and GRU models.

        Args:
            lstm_model: The trained LSTM model.
            gru_model: The trained GRU model.
            X (array): The input data.

        Returns:
            array: Combined predictions from LSTM and GRU models.
        """
        lstm_preds = lstm_model.predict(X)
        # self.output_decoder(lstm_preds, X)
        # print(self.output_decoder(
        #     predictions=lstm_preds, scaler=lstm_scaler, data=X))
        gru_preds = gru_model.predict(X)
        # print(self.output_decoder(predictions=gru_preds, scaler=gru_scaler, data=X))
        # print(gru_preds)
        meta_features = np.column_stack((lstm_preds, gru_preds))
        return meta_features

    def make_predictions(self, scaler, X_test):
        """Makes predictions using the trained meta-model.

        Args:
            X_test (array): The test input data.

        Returns:
            array: The predicted values.
        """
        # Generate meta-features using the LSTM and GRU models
        meta_features = self.generate_meta_features(
            self.lstm_model, self.gru_model, X_test)
        # Use the meta-model to predict based on the meta-features
        predictions = self.meta_model.predict(meta_features)
        # Decode the predictions to get the actual 'Close' prices
        decoded_predictions = self.output_decoder(
            predictions=predictions, scaler=scaler, data=X_test)
        return decoded_predictions

    def output_decoder(self, scaler, predictions, data):
        """Decodes the predictions using the inverse transform of the scaler.

        Args:
            predictions (array): Predicted values to be decoded.

        Returns:
            array: Decoded predicted values.
        """
        if scaler is None:
            raise ValueError("Scaler has not been initialized.")

        # Create a placeholder array to match the scaler's expected input shape
        placeholder = np.zeros((predictions.shape[0], data.shape[1]))
        # Assuming Close price is at index 3
        placeholder[:, 3] = predictions.flatten()

        # Perform inverse transform and extract the Close prices
        predicted_prices = scaler.inverse_transform(placeholder)[:, 3]
        return predicted_prices

    def load_meta_model(self, filename='meta_model.pkl'):
        """Loads a meta-model from a file.

        Args:
            filename (str): The filename to load the meta-model from.
        """
        with open(filename, 'rb') as f:
            self.meta_model = pickle.load(f)
        logging.info(f'Meta model loaded from {filename}.')

    def save_meta_model(self, filename='meta_model.pkl'):
        """Saves the meta-model to a file.

        Args:
            filename (str): The filename to save the meta-model.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.meta_model, f)
        logging.info(f'Meta model saved to {filename}.')

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
        logging.info(f'Scaler and feature shape saved to {
                     scaler_filename} and {shape_filename}.')

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
        logging.info(f'Scaler and feature shape loaded from {
                     scaler_filename} and {shape_filename}.')
