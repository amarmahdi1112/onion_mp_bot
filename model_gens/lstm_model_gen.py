import os
import logging
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Model, load_model  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Concatenate, Masking  # type: ignore
from model_gens.market_predictor_base import MarketPredictorBase
from model_gens.utils.static.columns import Columns
from settings import BASE_DIR
from model_gens.utils.model_training_tracker import ModelType
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from model_gens.utils.static.processing_type import ProcessingType 
class LSTMModel(MarketPredictorBase):
    def __init__(self, base_data_path, new_data_path=None, processing_type=ProcessingType.TRAINING, load_csv=False):
        super().__init__(base_data_path=base_data_path, new_data_path=new_data_path, processing_type=processing_type, load_csv=load_csv)
        self.model = None
        self.scaler = None

    @staticmethod
    def create_model(input_shapes):
        """Creates an LSTM model with multiple inputs.

        Args:
            input_shapes (list of tuples): The shapes of the input data.

        Returns:
            Model: The created LSTM model.
        """
        inputs = []
        lstm_layers = []

        for shape in input_shapes:
            input_layer = Input(shape=shape)
            masked_layer = Masking(mask_value=-999)(input_layer)
            lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(masked_layer)
            lstm_layer = Dropout(0.2)(lstm_layer)
            lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(lstm_layer)
            lstm_layer = Dropout(0.2)(lstm_layer)
            lstm_layer = Bidirectional(LSTM(32))(lstm_layer)
            lstm_layer = Dropout(0.2)(lstm_layer)
            inputs.append(input_layer)
            lstm_layers.append(lstm_layer)

        merged = Concatenate()(lstm_layers)
        dense_layer = Dense(64, activation='relu')(merged)
        dense_layer = Dropout(0.2)(dense_layer)
        dense_layer = Dense(32, activation='relu')(dense_layer)
        output_layer = Dense(1)(dense_layer)  # Predict only the Close price

        model = Model(inputs=inputs, outputs=output_layer)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, X_train_list, y_train, X_val_list, y_val):
        """Trains the LSTM model.

        Args:
            X_train_list (list of arrays): The training input data.
            y_train (array): The training target data.
            X_val_list (list of arrays): The validation input data.
            y_val (array): The validation target data.
        """
        MarketPredictorBase.clear_gpu_memory()
        self.model = self.create_model([x.shape[1:] for x in X_train_list])

        # Define early stopping and learning rate reduction callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

        # Compile the model with a Lower initial learning rate
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error')

        history = self.model.fit(X_train_list, y_train, epochs=10000, batch_size=64, validation_data=(X_val_list, y_val), callbacks=[early_stopping, reduce_lr])
        logging.info('LSTM Model trained.')
        return self.model

    def make_predictions(self, X_test_list):
        """Makes predictions using the trained LSTM model.

        Args:
            X_test_list (list of arrays): The test input data.

        Returns:
            array: The predicted values.
        """
        predictions = self.model.predict(X_test_list)
        return self.output_decoder(predictions)

    def load_model(self, column):
        """Loads the LSTM models for Open, High, Low, and Close prices."""
        model_path, scaler_path, shape_path = self.model_history.get_model_directory(model_table=ModelType.LSTM, column_name=column.name)
        if model_path:
            model_full_path = os.path.join(BASE_DIR, model_path)
            scaler_full_path = os.path.join(BASE_DIR, scaler_path)

            # Load the model, scaler, and shape
            model = load_model(model_full_path)
            self.scaler = self.preprocessor.load_scaler(scaler_full_path)

            self.model = model
            logging.info(f'{column.name} model loaded from {model_full_path}')
        else:
            logging.warning(f'No trained model found for {column.name}')
