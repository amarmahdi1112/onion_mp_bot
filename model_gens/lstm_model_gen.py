import os
import logging
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional  # type: ignore
from model_gens.market_predictor_base import MarketPredictorBase
from model_gens.utils.static.columns import Columns
from settings import BASE_DIR
from model_gens.utils.model_training_tracker import ModelType
from keras.models import load_model  # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from model_gens.utils.static.processing_type import ProcessingType 

class LSTMModel(MarketPredictorBase):
    def __init__(
        self, 
        base_data_path: str, 
        new_data_path: str = None, 
        processing_type=ProcessingType.TRAINING,
        load_csv: bool = False
    ):
        """Initializes the LSTM model class, inheriting from MarketPredictorBase."""
        super().__init__(
            base_data_path=base_data_path,
            new_data_path=new_data_path,
            processing_type=processing_type,
            load_csv=load_csv
        )
        self.model = None
        self.scaler = None

    @staticmethod
    def create_model(input_shape):
        """Creates an LSTM model.

        Args:
            input_shape (tuple): The shape of the input data.

        Returns:
            Model: The created LSTM model.
        """
        model = Sequential()
        model.add(Bidirectional(
            LSTM(128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))  # Predict only the Close price
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, X_train, y_train, X_val, y_val):
        """Trains the LSTM model.

        Args:
            X_train (array): The training input data.
            y_train (array): The training target data.
            X_val (array): The validation input data.
            y_val (array): The validation target data.
        """
        MarketPredictorBase.clear_gpu_memory()
        self.model = self.create_model((X_train.shape[1], X_train.shape[2]))

        # Define early stopping and learning rate reduction callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

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
        logging.info('LSTM Model trained.')
        # Save the model weights
        # self.model.save_weights(f'{BASE_DIR}/models/xauusd/lstm/lstm_weights.h5')
        return self.model

    def make_predictions(self, X_test):
        """Makes predictions using the trained LSTM model.

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
            model_table=ModelType.LSTM, column_name=column.name
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
            logging.info(f'{column.name} model loaded from {model_full_path}')
        else:
            logging.warning(f'No trained model found for {column.name}')
