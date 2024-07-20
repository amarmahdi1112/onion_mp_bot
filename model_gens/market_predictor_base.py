import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from scipy.stats import gaussian_kde
import gc
from keras import backend as K
from abc import ABC, abstractmethod
import logging
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import ta
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model  # type: ignore
import pickle
from model_gens.utils.model_training_tracker import ModelTrainingTracker
from settings import BASE_DIR
from model_gens.utils.static.columns import Columns
from statsmodels.tsa.arima.model import ARIMA
import joblib
from tensorflow.keras.optimizers import Adam  # type: ignore
import os
from model_gens.utils.preprocessor import Preprocessor
from model_gens.utils.static.processing_type import ProcessingType


class MarketPredictorBase(ABC):
    def __init__(
        self, 
        base_data_path, 
        new_data_path, 
        db_path=f'{BASE_DIR}/model_gens/db/model_training.db', 
        processing_type=ProcessingType.INITIAL,
        load_csv=True
    ):
        """Initializes the base class with common attributes for the market predictor."""
        self.model_history = ModelTrainingTracker(db_name=db_path)
        self.processing_type = processing_type
        print(f"Processing type: {self.processing_type}")
        self.load_csv = load_csv
        self.preprocessor = Preprocessor(path=base_data_path, processing_type=self.processing_type, load_csv=self.load_csv)
        self.new_preprocessor = None
        self.predicted_prices = None
        if new_data_path:
            self.new_preprocessor = Preprocessor(path=new_data_path)

    @staticmethod
    def clear_gpu_memory():
        """Clears GPU memory to avoid memory leaks and ensures a fresh state for tensorflow."""
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

    @staticmethod
    @abstractmethod
    def create_model(input_shape):
        """Creates a machine learning model.

        Args:
            input_shape (tuple): The shape of the input data.

        Returns:
            Model: A machine learning model.
        """
        pass

    @abstractmethod
    def train_model(self, X_train, y_train, X_val, y_val):
        """Trains the machine learning model.

        Args:
            X_train (array): The training input data.
            y_train (array): The training target data.
            X_val (array): The validation input data.
            y_val (array): The validation target data.
        """
        pass

    @abstractmethod
    def make_predictions(self, X_test):
        """Makes predictions using the trained model.

        Args:
            X_test (array): The test input data.

        Returns:
            array: The predicted values.
        """
        pass

    @staticmethod
    def save_models(model, model_name='model'):
        """Saves the trained model to disk.

        Args:
            model (Model): The trained machine learning model.
            model_name (str): The name of the file to save the model.
        """
        model.save(f'{model_name}')
        logging.info('Model saved.')

    def prepare_latest_data_and_predict_open(self):
        preprocess_last_trained_data = self.preprocessor

        preprocess_last_trained_data.add_all_props(self.new_preprocessor._data)

        # Prepare the dataset for the Open column
        try:
            X_seq, y_seq = preprocess_last_trained_data.prepare_dataset(
                target=Columns.Open, training=False)

        except Exception as e:
            print(f"Error preparing dataset: {e}")
            return None

        # Ensure the data shapes are consistent
        print(f"Feature set shape: {X_seq.shape}")
        print(f"Target shape: {y_seq.shape}")

        try:
            next_pred = self.model.predict(X_seq)
            next_pred_value = max(next_pred[0][0], 0)
            predictions = np.array([[next_pred_value]])

            decoded_predictions = preprocess_last_trained_data.output_decoder(
                predictions, preprocess_last_trained_data._data, Columns.Open.name
            )
            # Round the decoded predictions to 2 decimal places
            rounded_predictions = np.round(decoded_predictions, 2)

            return rounded_predictions
        except Exception as e:
            print("Error during prediction:", e)
            return None

    def prepare_latest_data_and_predict_low(self, new_predicted_open_price):
        preprocess_last_trained_data = self.preprocessor

        preprocess_last_trained_data.add_all_props(self.new_preprocessor._data)

        if new_predicted_open_price is None:
            raise ValueError(
                "The new predicted open price is required for predicting the low price.")

        # Create a new row with the predicted open price and append it to the data
        last_index = preprocess_last_trained_data._data.index[-1]
        new_index = last_index + pd.Timedelta(minutes=5)
        new_row = {
            Columns.Open.name: new_predicted_open_price,
            Columns.High.name: np.nan,
            Columns.Low.name: np.nan,
            Columns.Close.name: np.nan,
            Columns.Volume.name: np.nan,
            Columns.Open_Close_Diff.name: np.nan
        }
        new_data_df = pd.DataFrame(new_row, index=[new_index])
        preprocess_last_trained_data.add_all_props(new_data_df)

        # Prepare the dataset for the Low column
        try:
            X_seq, y_seq = preprocess_last_trained_data.prepare_dataset(
                target=Columns.Low, training=False)
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            return None

        # Ensure the data shapes are consistent
        print(f"Feature set shape: {X_seq.shape}")
        print(f"Target shape: {y_seq.shape}")

        try:
            next_pred = self.model.predict(X_seq)
            next_pred_value = max(next_pred[0][0], 0)
            predictions = np.array([[next_pred_value]])

            decoded_predictions = preprocess_last_trained_data.output_decoder(
                predictions, preprocess_last_trained_data._data, Columns.Low.name
            )
            # Round the decoded predictions to 2 decimal places
            rounded_predictions = np.round(decoded_predictions, 2)

            return rounded_predictions
        except Exception as e:
            print("Error during prediction:", e)
            return None

    def prepare_latest_data_and_predict_high(self, steps=12):
        preprocess_last_trained_data = self.preprocessor

        preprocess_last_trained_data.add_all_props(self.new_preprocessor._data)

        # print(preprocess_last_trained_data._data)

        try:
            X_seq, y_seq = preprocess_last_trained_data.prepare_dataset(
                target=Columns.High, training=False)
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            return None

        # Ensure the data shapes are consistent
        print(f"Feature set shape: {X_seq.shape}")
        print(f"Target shape: {y_seq.shape}")

        try:
            next_pred = self.model.predict(X_seq)
            next_pred_value = max(next_pred[0][0], 0)
            predictions = np.array([[next_pred_value]])

            decoded_predictions = preprocess_last_trained_data.output_decoder(
                predictions, preprocess_last_trained_data._data, Columns.High.name
            )
            # Round the decoded predictions to 2 decimal places
            rounded_predictions = np.round(decoded_predictions, 2)

            return rounded_predictions
        except Exception as e:
            print("Error during prediction:", e)
            return None
    # def prepare_latest_data_and_predict(self, model_name: Columns, eval=False):
    #     """Prepares the latest data and makes future predictions.

    #     Args:
    #         model_name (Columns): The name of the model to use for prediction ('Open', 'High', 'Low', 'Close').
    #         eval (bool): Whether to evaluate the model on the latest data.

    #     Returns:
    #         DataFrame: DataFrame with predictions.
    #     """
    #     preprocess_last_trained_data = Preprocessor(
    #         path=f'{BASE_DIR}/last_60_steps.csv'
    #     )
    #     preprocess_last_trained_data.add_all_props(
    #         self.preprocessor.data
    #     )

    #     last_Open = preprocess_last_trained_data.Open[-1]

        # # X, y = preprocess_last_trained_data.high_low_price_features(
        # #     target_column=model_name, training=False)
        # if model_name != Columns.Close:
        #     X, y = preprocess_last_trained_data.high_low_price_features(
        #         target_column=model_name, training=False
        #     )
        # else:
        #     X, y = preprocess_last_trained_data.Close_price_data_processor(
        #         training=False
        #     )

        # scaled_data = preprocess_last_trained_data.scale_data(X)

        # feature, y = preprocess_last_trained_data.create_sequences(
        #     scaled_data, y, time_step=60
        # )

        # next_pred = self.model.predict(feature)
        # next_pred_value = max(next_pred[0][0], 0)
        # predictions = np.array([[next_pred_value]])
        # decoded_predictions = preprocess_last_trained_data.output_decoder(
        #     predictions, preprocess_last_trained_data.data, model_name.value
        # )

        # last_index = pd.to_datetime(self.preprocessor.data.index[-1])

        # predicted_date = last_index + pd.Timedelta(minutes=5)
        # predicted_df = pd.DataFrame(decoded_predictions, index=[predicted_date], columns=[
        #                             f'Predicted {model_name.name.capitalize()}'])
        # return predicted_df

    #     # remove the first row of the combined data and save the new data
    #     preprocess_last_trained_data.data = preprocess_last_trained_data.data.iloc[1:]
    #     features = preprocess_last_trained_data.high_low_price_features()

    #     scale_data = preprocess_last_trained_data.scale_data(features)
    #     model = getattr(self, f'model_{model_name.name.lower()}')
    #     next_pred = model.predict(scale_data)
    #     next_pred_value = max(next_pred[0][0], 0)
    #     predictions = np.array([[next_pred_value]])
    #     decoded_predictions = self.output_decoder(
    #         predictions, scale_data, model_name.value)
    #     last_index = pd.to_datetime(scale_data.index[-1])
    #     predicted_date = last_index + pd.Timedelta(minutes=5)
    #     predicted_df = pd.DataFrame(decoded_predictions, index=[predicted_date], columns=[
    #                                 f'Predicted {model_name.name.capitalize()}'])
    #     return scale_data, predicted_df

        # # Check and adjust date if needed
        # combined_data = self.check_and_adjust_date(latest_data, saved_data)
        # preprocessed_data = self.preprocess_data(combined_data)
        # if model_name == Columns.High:
        #     # Remove High and Close columns
        #     if 'High' in combined_data.index and 'Close' in combined_data.index:
        #         combined_data = combined_data.drop(['High', 'Close'])
        # elif model_name == Columns.Low:
        #     # Remove Low and Close columns
        #     if 'Low' in combined_data.index and 'Close' in combined_data.index:
        #         combined_data = combined_data.drop(['Low', 'Close'])
        # elif model_name == Columns.Close:
        #     # Remove Close column
        #     if 'Close' in combined_data.index:
        #         combined_data = combined_data.drop(['Close'])
        # combined_data = self.prepare_dataset_for_prediction(
        #     preprocessed_data, target=model_name)
        # print(f"{preprocessed_data}")

        # # Preprocess the combined data
        # combined_data = self.preprocess_data(combined_data)
        # print(f"Shape of combined data: {combined_data}")
        # # Get the features from the last row
        # last_row = combined_data.iloc[-1]

        # # Create a DataFrame for the last row to match the structure
        # last_row_df = pd.DataFrame(last_row).transpose()

        # # Add this row to the combined data for prediction
        # combined_data = pd.concat([combined_data, last_row_df])
        # print(f"Shape of combined data: {combined_data}")
        # if eval:
        #     X, y = self.prepare_dataset(combined_data, target=model_name.value)
        #     self.evaluate_model(X, y)
        #     return combined_data, None

        # if combined_data.shape[0] < 60:
        #     raise ValueError(
        #         "Not enough data after preprocessing to create sequences.")

        # X = self.prepare_dataset_for_prediction(
        #     combined_data, target=model_name, time_step=60)

        # if X.size == 0:
        #     raise ValueError(
        #         "Prepared dataset for prediction is empty. Check your preprocessing steps.")

        # initial_input = X[-1]

        # model = getattr(self, f'model_{model_name.name.lower()}')
        # next_pred = model.predict(initial_input[np.newaxis, :, :])
        # next_pred_value = max(next_pred[0][0], 0)

        # predictions = np.array([[next_pred_value]])
        # decoded_predictions = self.output_decoder(
        #     predictions, combined_data, model_name.value)

        # last_index = pd.to_datetime(combined_data.index[-1])
        # predicted_date = last_index + pd.Timedelta(minutes=5)
        # predicted_df = pd.DataFrame(decoded_predictions, index=[predicted_date], columns=[
        #                             f'Predicted {model_name.name.capitalize()}'])

        # return combined_data, predicted_df

    # def check_and_adjust_date(self, current_data, saved_data):
    #     """Checks and adjusts the date of the current data based on the saved data.

    #     Args:
    #         current_data (DataFrame): The current data as a DataFrame.
    #         saved_data (DataFrame): The saved data as a DataFrame.

    #         Returns:
    #         DataFrame: Combined data with the adjusted current data.
    #     """
    #     # Get the last date from the saved data
    #     last_saved_date = saved_data.index[-1]

    #     # Check if the current data date is the next date after the last saved date
    #     current_data_date = current_data.index[0]

    #     # Combine the current data with the saved data
    #     combined_data = pd.concat([saved_data, current_data])

    #     if current_data_date != last_saved_date:
    #         # remove the first row of the combined data and save the new data
    #         combined_data = combined_data.iloc[1:]

    #     return combined_data

    # def output_decoder(self, predictions, data, column_index):
    #     """Decodes the predictions from the LSTM model.

    #     Args:
    #         predictions (array): The predictions made by the LSTM model.
    #         data (DataFrame): The original data used for scaling.
    #         column_index (int): The index of the column to decode.

    #     Returns:
    #         array: The decoded predictions.
    #     """
    #     placeholder = np.zeros((predictions.shape[0], data.shape[1]))
    #     placeholder[:, column_index] = predictions.flatten()
    #     predicted_prices = self.scaler.inverse_transform(placeholder)[
    #         :, column_index]
    #     return predicted_prices

    # def compute_advantages(self, rewards):
    #     """Computes the advantages based on the rewards.

    #     Args:
    #         rewards (tensor): The reward values.

    #     Returns:
    #         tensor: The computed advantages.
    #     """
    #     mean_rewards = tf.reduce_mean(rewards)
    #     advantages = rewards - mean_rewards
    #     return advantages

    # def loss_fn_spot(self, states, actions, advantages, density_values):
    #     """Calculates the loss for SPOT training.

    #     Args:
    #         states (array): The state values.
    #         actions (array): The action values.
    #         advantages (array): The advantage values.
    #         density_values (array): The density values.

    #     Returns:
    #         float: The computed loss.
    #     """
    #     action_probs = self.model(states)
    #     # Ensure actions have the same shape as action_probs
    #     actions = tf.expand_dims(actions, axis=-1)

    #     # Convert actions to float32
    #     actions = tf.cast(actions, dtype=tf.float32)
    #     # Convert advantages to float32
    #     advantages = tf.cast(advantages, dtype=tf.float32)
    #     # Convert density_values to float32
    #     density_values = tf.convert_to_tensor(density_values, dtype=tf.float32)

    #     # Ensure the shapes are compatible for multiplication
    #     action_probs = tf.expand_dims(action_probs, axis=-1)
    #     action_log_probs = tf.math.log(
    #         tf.reduce_sum(action_probs * actions, axis=1))
    #     density_penalty = -tf.reduce_mean(tf.math.log(density_values))
    #     loss = -tf.reduce_mean(action_log_probs * advantages) + density_penalty
    #     return loss

    # def Closed_form_policy_improvement(self, states, actions, rewards):
    #     """Performs Closed-form policy improvement.

    #     Args:
    #         states (array): The state values.
    #         actions (array): The action values.
    #         rewards (array): The reward values.

    #     Returns:
    #         array: The updated policy values.
    #     """
    #     action_probs = self.model(states)
    #     advantages = self.compute_advantages(rewards)
    #     # Ensure both action_probs and advantages are of the same type (float32)
    #     advantages = tf.cast(advantages, dtype=tf.float32)
    #     action_probs = tf.cast(action_probs, dtype=tf.float32)
    #     # Expand the dimensions of advantages to match action_probs
    #     advantages = tf.expand_dims(advantages, axis=-1)
    #     updated_policy = action_probs * advantages
    #     return updated_policy

    # def loss_fn_gcpc(self, states, goals):
    #     """Calculates the loss for GCPC training.

    #     Args:
    #         states (array): The state values.
    #         goals (array): The goal values.

    #     Returns:
    #         float: The computed loss.
    #     """
    #     predictions = self.model(states)
    #     goal_predictions = predictions * goals
    #     loss = tf.reduce_mean(tf.square(goal_predictions - states[:, 0]))
    #     return loss

    # def loss_fn_sswnp(self, states, targets):
    #     """Calculates the loss for SSWNP training.

    #     Args:
    #         states (array): The state values.
    #         targets (array): The target values.

    #     Returns:
    #         float: The computed loss.
    #     """
    #     predictions = self.model(states)
    #     loss = tf.reduce_mean(tf.square(predictions - targets))
    #     return loss

    # def loss_fn(self, states, targets):
    #     """Calculates the loss for standard training.

    #     Args:
    #         states (array): The state values.
    #         targets (array): The target values.

    #     Returns:
    #         float: The computed loss.
    #     """
    #     predictions = self.model(states)
    #     loss = tf.reduce_mean(tf.square(predictions - targets))
    #     return loss

    # def train_with_spot(self, X_train, y_train, X_val, y_val, model=None, batch_size=64, epochs=100, patience=10):
    #     """Trains the model using the SPOT method with early stopping.

    #     Args:
    #         X_train (array): The training input data.
    #         y_train (array): The training target data.
    #         X_val (array): The validation input data.
    #         y_val (array): The validation target data.
    #         model (tf.keras.Model): The pre-trained model to continue training.
    #         batch_size (int): The size of the batches for training.
    #         epochs (int): The number of epochs to train the model.
    #         patience (int): Number of epochs with no improvement after which training will be stopped.
    #     """
    #     # Ensure data is in the correct format and normalized
    #     X_train = np.array(X_train)
    #     y_train = np.array(y_train)

    #     # Flatten the input for density estimation
    #     X_train_flat = X_train.reshape(X_train.shape[0], -1)

    #     # Estimate density
    #     density_values = self.estimate_density(X_train_flat)
    #     if np.all(density_values == 0):
    #         raise ValueError(
    #             "Density values are all zero, check the density estimation process.")

    #     # Add epsilon to avoid log(0)
    #     epsilon = 1e-10
    #     density_values = np.clip(density_values, epsilon, None)

    #     # Convert density_values to float32
    #     density_values = density_values.astype(np.float32)

    #     # Prepare tensorflow dataset
    #     train_dataset = tf.data.Dataset.from_tensor_slices(
    #         (X_train, y_train, density_values))
    #     train_dataset = train_dataset.shuffle(buffer_size=1024).batch(
    #         batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #     val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(
    #         batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    #     self.optimizer = tf.keras.optimizers.Adam(
    #         learning_rate=1e-5)  # Further reduced learning rate

    #     @ tf.function
    #     def train_step(X, y, densities):
    #         with tf.GradientTape() as tape:
    #             advantages = self.compute_advantages(y)
    #             loss = self.loss_fn_spot(X, y, advantages, densities)
    #         grads = tape.gradient(loss, self.model.trainable_variables)
    #         grads = [tf.clip_by_value(grad, -0.5, 0.5)
    #                  for grad in grads]  # Adjust clipping range
    #         self.optimizer.apply_gradients(
    #             zip(grads, self.model.trainable_variables))
    #         return loss

    #     # Use the provided model or create a new one
    #     if model is None:
    #         self.model = self.create_model(
    #             (X_train.shape[1], X_train.shape[2]))
    #     else:
    #         self.model = model

    #     best_val_loss = float('inf')
    #     epochs_no_improve = 0

    #     # Training loop
    #     for epoch in range(epochs):
    #         print(f"Epoch {epoch+1}/{epochs}")
    #         for step, (X_batch, y_batch, densities_batch) in enumerate(train_dataset):
    #             loss = train_step(X_batch, y_batch, densities_batch)
    #             if step % 100 == 0:
    #                 print(f"Step {step}, Loss: {loss.numpy()}")

    #         # Validation step
    #         val_loss = 0
    #         for X_batch, y_batch in val_dataset:
    #             val_loss += self.model.evaluate(X_batch, y_batch, verbose=0)
    #         val_loss /= len(val_dataset)
    #         print(f"Validation Loss: {val_loss}")

    #         # Check for early stopping
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             epochs_no_improve = 0
    #             self.model.save('spot_trained_model.h5')
    #             logging.info('Model improved and saved.')
    #         else:
    #             epochs_no_improve += 1

    #         if epochs_no_improve >= patience:
    #             print(f"Early stopping triggered after {
    #                   patience} epochs of no improvement.")
    #             break

    #     logging.info('Model trained with SPOT method.')

    #     return self.model

    # def evaluate_model(self, X_test, y_test):
    #     """Evaluates the trained model on the test set."""
    #     predictions = self.model.predict(X_test)
    #     test_loss = self.loss_fn(X_test, y_test).numpy()
    #     print(f"Test Loss: {test_loss}")

    #     # Plot the true values vs predictions
    #     plt.plot(y_test, label='True Values')
    #     plt.plot(predictions, label='Predictions')
    #     plt.xlabel('Sample')
    #     plt.ylabel('Value')
    #     plt.title('True Values vs Predictions')
    #     plt.legend()
    #     plt.show()

    # def train_with_cfpi(self, X_train, y_train, X_val, y_val, model=None, batch_size=64, epochs=100, patience=10):
    #     """Trains the model using the Closed-Form Policy Improvement (CFPI) method.

    #     Args:
    #         X_train (array): The training input data.
    #         y_train (array): The training target data.
    #         X_val (array): The validation input data.
    #         y_val (array): The validation target data.
    #         model (tf.keras.Model): The pre-trained model to continue training.
    #         batch_size (int): The size of the batches for training.
    #         epochs (int): The number of epochs to train the model.
    #         patience (int): Number of epochs with no improvement after which training will be stopped.
    #     """
    #     # Ensure data is in the correct format and normalized
    #     X_train = np.array(X_train, dtype=np.float32)
    #     y_train = np.array(y_train, dtype=np.float32)
    #     X_val = np.array(X_val, dtype=np.float32)
    #     y_val = np.array(y_val, dtype=np.float32)

    #     self.optimizer = keras.optimizers.Adam(learning_rate=1e-5)

    #     # Use the provided model or create a new one
    #     if model is None:
    #         self.model = self.create_model(
    #             (X_train.shape[1], X_train.shape[2]))
    #     else:
    #         self.model = model

    #     best_val_loss = float('inf')
    #     epochs_no_improve = 0
    #     val_losses = []

    #     @ tf.function
    #     def train_step(states, targets):
    #         with tf.GradientTape() as tape:
    #             updated_policy = self.Closed_form_policy_improvement(
    #                 states, targets, targets)
    #             loss = self.cfpi_loss_fn(updated_policy, targets)
    #         grads = tape.gradient(loss, self.model.trainable_variables)
    #         self.optimizer.apply_gradients(
    #             zip(grads, self.model.trainable_variables))
    #         return loss

    #     train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(
    #         buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #     val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(
    #         batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    #     # Training loop
    #     for epoch in range(epochs):
    #         print(f"Epoch {epoch+1}/{epochs}")
    #         for step, (states_batch, targets_batch) in enumerate(train_dataset):
    #             loss = train_step(states_batch, targets_batch)
    #             if step % 100 == 0:
    #                 print(f"Step {step}, Loss: {loss.numpy()}")

    #         # Validation step
    #         val_loss = 0
    #         for states_batch, targets_batch in val_dataset:
    #             val_loss += self.cfpi_loss_fn(states_batch,
    #                                           targets_batch).numpy()
    #         val_loss /= len(val_dataset)
    #         val_losses.append(val_loss)
    #         print(f"Validation Loss: {val_loss}")

    #         # Check for early stopping
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             epochs_no_improve = 0
    #             self.model.save('cfpi_trained_model.h5')
    #             logging.info('Model improved and saved.')
    #         else:
    #             epochs_no_improve += 1

    #         if epochs_no_improve >= patience:
    #             print(f"Early stopping triggered after {
    #                   patience} epochs of no improvement.")
    #             break

    #     # Plot validation loss over episodes
    #     plt.plot(range(epochs), val_losses, label='Validation Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title('Validation Loss Over Epochs')
    #     plt.legend()
    #     plt.show()

    #     logging.info('Model trained with CFPI method.')

    #     return self.model

    # def cfpi_loss_fn(self, states, targets):
    #     """Calculates the loss for CFPI training.

    #     Args:
    #         states (array): The state values.
    #         targets (array): The target values.

    #     Returns:
    #         float: The computed loss.
    #     """
    #     predictions = self.model(states)
    #     loss = tf.reduce_mean(tf.square(predictions - targets))
    #     return loss

    # def train_with_gcpc(self, X_train, y_train, X_val, y_val, batch_size=64, epochs=100, patience=10):
    #     """Trains the model using the Goal-Constrained Policy Optimization (GCPC) method.

    #     Args:
    #         X_train (array): The training input data.
    #         y_train (array): The training target data.
    #         X_val (array): The validation input data.
    #         y_val (array): The validation target data.
    #         batch_size (int): The size of the batches for training.
    #         epochs (int): The number of epochs to train the model.
    #         patience (int): Number of epochs with no improvement after which training will be stopped.
    #     """
    #     X_train = np.array(X_train, dtype=np.float32)
    #     y_train = np.array(y_train, dtype=np.float32)
    #     X_val = np.array(X_val, dtype=np.float32)
    #     y_val = np.array(y_val, dtype=np.float32)

    #     self.optimizer = keras.optimizers.Adam(learning_rate=1e-5)

    #     @ tf.function
    #     def train_step(states, goals):
    #         with tf.GradientTape() as tape:
    #             loss = self.loss_fn_gcpc(states, goals)
    #         grads = tape.gradient(loss, self.model.trainable_variables)
    #         grads = [tf.clip_by_value(grad, -0.5, 0.5)
    #                  for grad in grads]  # Adjust clipping range
    #         self.optimizer.apply_gradients(
    #             zip(grads, self.model.trainable_variables))
    #         return loss

    #     train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    #     train_dataset = train_dataset.shuffle(buffer_size=1024).batch(
    #         batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    #     best_val_loss = float('inf')
    #     epochs_no_improve = 0

    #     for epoch in range(epochs):
    #         print(f"Epoch {epoch+1}/{epochs}")
    #         for step, (states_batch, goals_batch) in enumerate(train_dataset):
    #             loss = train_step(states_batch, goals_batch)
    #             if step % 100 == 0:
    #                 print(f"Step {step}, Loss: {loss.numpy()}")

    #         # Check for early stopping
    #         # Use the new evaluation method
    #         val_loss = self.gcpc_eval_model(X_val, y_val)
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             epochs_no_improve = 0
    #             self.model.save('gcpc_trained_model.h5')
    #             logging.info('Model improved and saved.')
    #         else:
    #             epochs_no_improve += 1

    #         if epochs_no_improve >= patience:
    #             print(f"Early stopping triggered after {
    #                   patience} epochs of no improvement.")
    #             break

    #     logging.info('Model trained with GCPC method.')

    #     return self.model

    # def gcpc_eval_model(self, X_val, y_val):
    #     """Evaluates the trained model on the validation set for GCPC.

    #     Args:
    #         X_val (array): The validation input data.
    #         y_val (array): The validation target data.

    #     Returns:
    #         float: The computed validation loss.
    #     """
    #     X_val = np.array(X_val, dtype=np.float32)
    #     y_val = np.array(y_val, dtype=np.float32)

    #     val_predictions = self.model.predict(X_val)
    #     val_loss = self.loss_fn_gcpc(X_val, y_val).numpy()
    #     print(f"Validation Loss: {val_loss}")
    #     return val_loss

    # def loss_fn_gcpc(self, states, goals):
    #     """Calculates the loss for GCPC training.

    #     Args:
    #         states (array): The state values.
    #         goals (array): The goal values.

    #     Returns:
    #         float: The computed loss.
    #     """
    #     predictions = self.model(states)
    #     # Ensure both predictions and goals are of the same type (float32)
    #     goals = tf.cast(goals, dtype=tf.float32)
    #     predictions = tf.cast(predictions, dtype=tf.float32)
    #     # Ensure states[:, 3] is of type float32 (assuming Close price is at index 3)
    #     states_Close = tf.cast(states[:, 3], dtype=tf.float32)
    #     goal_predictions = predictions * goals
    #     # Reshape states_Close to match the dimensions of goal_predictions
    #     states_Close = tf.reshape(states_Close, [-1, 1])
    #     # Broadcast states_Close to match the shape of goal_predictions
    #     states_Close = tf.broadcast_to(
    #         states_Close, tf.shape(goal_predictions))
    #     loss = tf.reduce_mean(tf.square(goal_predictions - states_Close))
    #     return loss

    # def train_with_sswnp(self, X_train, y_train, X_val, y_val):
    #     """Trains the model using the Sample-Weighted Non-Parametric (SSWNP) method.

    #     Args:
    #         X_train (array): The training input data.
    #         y_train (array): The training target data.
    #         X_val (array): The validation input data.
    #         y_val (array): The validation target data.
    #     """
    #     num_episodes = 100  # Example number of episodes
    #     self.optimizer = keras.optimizers.Adam()

    #     for episode in range(num_episodes):
    #         with tf.GradientTape() as tape:
    #             loss = self.loss_fn_sswnp(X_train, y_train)
    #         grads = tape.gradient(loss, self.model.trainable_variables)
    #         self.optimizer.apply_gradients(
    #             zip(grads, self.model.trainable_variables))

    # def train_with_adversarial_examples(self, X_train, y_train, X_val, y_val):
    #     """Trains the model using adversarial examples.

    #     Args:
    #         X_train (array): The training input data.
    #         y_train (array): The training target data.
    #         X_val (array): The validation input data.
    #         y_val (array): The validation target data.
    #     """
    #     num_episodes = 100  # Example number of episodes
    #     self.optimizer = keras.optimizers.Adam()

    #     for episode in range(num_episodes):
    #         adversarial_states = self.generate_adversarial_examples(
    #             self.model, X_train)
    #         with tf.GradientTape() as tape:
    #             loss = self.loss_fn(adversarial_states, y_train)
    #         grads = tape.gradient(loss, self.model.trainable_variables)
    #         self.optimizer.apply_gradients(
    #             zip(grads, self.model.trainable_variables))

    # @ staticmethod
    # def split_data_into_chunks(data, days=60):
    #     """Splits the data into chunks.

    #     Args:
    #         data (DataFrame): The input data.
    #         days (int): The number of days per chunk.

    #     Returns:
    #         list: A list of data chunks.
    #     """
    #     chunks = []
    #     start_idx = 0
    #     while start_idx < len(data):
    #         end_idx = start_idx + (days * 288)
    #         chunks.append(data[start_idx:end_idx])
    #         start_idx = end_idx
    #     return chunks

    # @staticmethod
    # def estimate_density(samples):
    #     """Estimates the density of the samples using Gaussian KDE."""
    #     # Ensure samples are in the correct shape
    #     samples = np.array(samples)
    #     if samples.ndim > 2:
    #         samples = samples.reshape(samples.shape[0], -1)
    #     # print(samples.shape)

    #     print(f"Shape of samples for density estimation: {samples.shape}")

    #     # # Adding a small amount of noise to the dataset to avoid singular matrix
    #     noise = np.random.normal(0, 1e-6, samples.shape)
    #     noisy_samples = samples + noise

    #     print(f"Noisy samples range: {
    #           noisy_samples.min()} - {noisy_samples.max()}")

    #     try:
    #         kde = gaussian_kde(noisy_samples.T)
    #         density_values = kde(noisy_samples.T)
    #         print(f"Density estimation range: {
    #               density_values.min()} - {density_values.max()}")
    #     except np.linalg.LinAlgError as e:
    #         print(f"LinAlgError: {
    #               e}. Attempting PCA-based density estimation.")
    #         # Limiting components to avoid singular matrix
    #         pca = PCA(n_components=min(samples.shape[0] - 1, 50))
    #         reduced_samples = pca.fit_transform(noisy_samples)
    #         kde = gaussian_kde(reduced_samples.T)
    #         density_values = kde(reduced_samples.T)
    #     except ValueError as e:
    #         print(f"ValueError: {e}. Attempting PCA-based density estimation.")
    #         # Limiting components to avoid singular matrix
    #         pca = PCA(n_components=min(samples.shape[0] - 1, 50))
    #         reduced_samples = pca.fit_transform(noisy_samples)
    #         kde = gaussian_kde(reduced_samples.T)
    #         density_values = kde(reduced_samples.T)

    #     print(f"Density estimation range: {
    #           density_values.min()} - {density_values.max()}")
    #     return density_values
