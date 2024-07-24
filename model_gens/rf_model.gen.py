from sklearn.ensemble import RandomForestRegressor
import numpy as np
import logging
from model_gens.market_predictor_base import MarketPredictorBase  # Make sure the path is correct


class RFModel(MarketPredictorBase):
    def __init__(self):
        """Initializes the Random Forest model class, inheriting from MarketPredictorBase."""
        super().__init__()
        self.model = None

    @staticmethod
    def create_model():
        """Creates a Random Forest model.

        Returns:
            RandomForestRegressor: The created Random Forest model.
        """
        model = RandomForestRegressor(n_estimators=1000, random_state=42)
        return model

    def train_model(self, X_train, y_train, X_val, y_val):
        """Trains the Random Forest model.

        Args:
            X_train (array): The training input data.
            y_train (array): The training target data.
            X_val (array): The validation input data.
            y_val (array): The validation target data.
        """
        X_train = X_train.reshape(
            (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        X_val = X_val.reshape(
            (X_val.shape[0], X_val.shape[1] * X_val.shape[2]))

        self.model = self.create_model()
        self.model.fit(X_train, y_train)

        # Validate the model
        val_predictions = self.model.predict(X_val)
        val_loss = np.mean((val_predictions - y_val) ** 2)
        logging.info(f'Validation Loss: {val_loss}')

        logging.info('Random Forest Model trained.')
        return self.model

    def make_predictions(self, X_test):
        """Makes predictions using the trained Random Forest model.

        Args:
            X_test (array): The test input data.

        Returns:
            array: The predicted values.
        """
        X_test = X_test.reshape(
            (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
        predictions = self.model.predict(X_test)
        return self.output_decoder(predictions, self.data)
