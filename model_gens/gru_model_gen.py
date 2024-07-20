import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional  # type: ignore
from model_gens.market_predictor_base import MarketPredictorBase
from model_gens.utils.static.columns import Columns
from settings import BASE_DIR
from model_gens.utils.model_training_tracker import ModelType
from keras.models import load_model  # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from model_gens.utils.static.processing_type import ProcessingType


class GRUModel(MarketPredictorBase):
    def __init__(
        self, 
        base_data_path, 
        new_data_path=None, 
        processing_type=ProcessingType.TRAINING,
        load_csv=False
    ):
        super().__init__(
            base_data_path=base_data_path,
            new_data_path=new_data_path,
            processing_type=processing_type,
            load_csv=load_csv
        )
        self.model = None

    @staticmethod
    def create_model(input_shape):
        """Creates a GRU model.

        Args:
            input_shape (tuple): The shape of the input data.

        Returns:
            Model: The created GRU model.
        """
        model = Sequential()
        model.add(Bidirectional(
            GRU(128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Bidirectional(GRU(64, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(GRU(32)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))  # Predict only the Close price
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, X_train, y_train, X_val, y_val):
        """Trains the GRU model.

        Args:
            X_train (array): The training input data.
            y_train (array): The training target data.
            X_val (array): The validation input data.
            y_val (array): The validation target data.
        """
        # Check if data needs to be reshaped
        if len(X_train.shape) == 2:
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        if len(X_val.shape) == 2:
            X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

        # Ensure that there are enough time steps for MaxPooling
        if X_train.shape[1] < 2 or X_val.shape[1] < 2:
            raise ValueError(
                "Input data must have at least 2 time steps for MaxPooling1D.")

        MarketPredictorBase.clear_gpu_memory()
        self.model = self.create_model((X_train.shape[1], X_train.shape[2]))

        # Define early stopping and learning rate reduction callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        # Compile the model with a Lower initial learning rate
        self.model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4), loss='mean_squared_error')

        history = self.model.fit(
            X_train, y_train,
            epochs=10000,
            batch_size=64,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr]
        )
        logging.info('GRU Model trained.')
        return self.model

    def make_predictions(self, X_test):
        """Makes predictions using the trained GRU model.

        Args:
            X_test (array): The test input data.

        Returns:
            array: The predicted values.
        """
        predictions = self.model.predict(X_test)
        return self.output_decoder(predictions)

    def load_model(self, column):
        """Loads the LSTM models for Open, High, Low, and Close prices."""
        # for column in Columns:
        model_path, scaler_path, shape_path = self.model_history.get_model_directory(
            model_table=ModelType.GRU, column_name=column.name
        )
        if model_path:
            model_full_path = os.path.join(BASE_DIR, model_path)
            scaler_full_path = os.path.join(BASE_DIR, scaler_path)
            # shape_full_path = os.path.join(BASE_DIR, shape_path)

            # Load the model, scaler, and shape
            model = load_model(model_full_path)
            self.scaler = self.preprocessor.load_scaler(scaler_full_path)
            # self.feature_shape = self.preprocessor.load_shape(shape_full_path)

            self.model = model
            logging.info(f'{column.name} model loaded from {
                model_full_path}')
        else:
            logging.warning(f'No trained model found for {column.name}')
