import tensorflow as tf  # type: ignore
from datetime import datetime
from settings import BASE_DIR
import os
from model_gens.utils.static.columns import Columns
from model_gens.utils.static.model_type import ModelType
from model_gens.market_predictor_base import MarketPredictorBase
from model_gens import lstm_model_gen, gru_model_gen
from typing import Union
import pandas as pd


class ModelTrainer:
    def __init__(self, model_class, model_type, data_path, asset_name='XAUUSD'):
        self.model_class = model_class
        self.model_type: ModelType = model_type
        self.asset_name = asset_name
        self.market_predictor: Union[
            MarketPredictorBase, lstm_model_gen.LSTMModel, gru_model_gen.GRUModel
        ] = model_class(base_data_path=data_path)
        
    def train(self, skip_existing=False):
        self.market_predictor.preprocessor.save_last_steps(60)
        # Remove duplicates and maintain order if necessary
        columns = list(set([Columns.Low, Columns.Close]))

        for column in columns:
            if self._skip_column_training(column, skip_existing):
                continue

            print(f"Processing {
                  column.name} - {columns.index(column) + 1}/{len(columns)}")
            self._train_and_save_model(column)

    def _skip_column_training(self, column, skip_existing):
        history = self.market_predictor.model_history.get_last_training_date(
            model_table=self.model_type, model_type=column.name.upper())
        last_data_date, _ = history
        last_data_date = pd.to_datetime(last_data_date) if isinstance(
            last_data_date, str) else last_data_date
        latest_data_date = self.market_predictor.preprocessor._data.index[-1]

        if last_data_date and last_data_date >= latest_data_date:
            print("Model has already been trained on the latest data.")
            return True

        model_path, _, _, _, _ = self._generate_paths(column)
        if skip_existing and os.path.exists(f'{model_path}.h5'):
            print(f"Skipping {column.name} as model already exists.")
            return True

        return False

    def _train_and_save_model(self, column):
        X_train, X_test, y_train, y_test = self.market_predictor.preprocessor.prepare_dataset(
            target=column, processed=True)
        model = self.market_predictor.train_model(
            X_train, y_train, X_test, y_test)

        model_path, scaler_path, shape_path, scaler_name, shape_name = self._generate_paths(
            column)
        self.market_predictor.save_models(model, model_path)
        self.market_predictor.preprocessor.save_scaler(
            scaler_filename=scaler_path)

        latest_data_date = self.market_predictor.preprocessor._data.index[-1]
        self.market_predictor.model_history.insert_new_training_data(
            model_table=self.model_type,
            model_name=f'{self.asset_name}_{self.model_type.value.lower()}_{
                column.name.lower()}',
            model_type=column.name,
            data_date=str(latest_data_date),
            model_date=str(datetime.now()),
            model_path=os.path.relpath(model_path, BASE_DIR),
            scaler_path=os.path.relpath(scaler_path, BASE_DIR),
            shape_path=os.path.relpath(shape_path, BASE_DIR),
            scaler_name=scaler_name,
            shape_name=shape_name,
            notes='Training completed'
        )

    def _generate_paths(self, column):
        model_subdir, scaler_subdir, shape_subdir = self._get_subdirectories(
            column)

        self._ensure_directory_exists(model_subdir)
        self._ensure_directory_exists(scaler_subdir)
        self._ensure_directory_exists(shape_subdir)

        model_filename = f'{self.asset_name}_{self.model_type.value.lower()}_{
            column.name.lower()}.h5'
        scaler_filename = f'{self.asset_name}_{self.model_type.value.lower()}_{
            column.name.lower()}_scaler.pkl'
        shape_filename = f'{self.asset_name}_{self.model_type.value.lower()}_{
            column.name.lower()}_shape.pkl'

        model_path = os.path.join(model_subdir, model_filename)
        scaler_path = os.path.join(scaler_subdir, scaler_filename)
        shape_path = os.path.join(shape_subdir, shape_filename)

        return model_path, scaler_path, shape_path, scaler_filename, shape_filename

    def _get_subdirectories(self, column):
        base_model_path = f'models/{self.asset_name.lower()
                                    }/{self.model_type.value.lower()}/{column.name.lower()}'
        base_scaler_path = f'models/{self.asset_name.lower()}/scalers/{
            self.model_type.value.lower()}/{column.name.lower()}'
        base_shape_path = f'models/{self.asset_name.lower()}/shapes/{
            self.model_type.value.lower()}/{column.name.lower()}'

        model_subdir = os.path.join(BASE_DIR, base_model_path)
        scaler_subdir = os.path.join(BASE_DIR, base_scaler_path)
        shape_subdir = os.path.join(BASE_DIR, base_shape_path)

        return model_subdir, scaler_subdir, shape_subdir

    @staticmethod
    def _ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'Created directory: {directory}')
        else:
            print(f'Directory already exists: {directory}')
