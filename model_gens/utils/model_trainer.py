import tensorflow as tf  # type: ignore
from datetime import datetime
from settings import BASE_DIR
import os
from model_gens.utils.static.columns import Columns
from model_gens.utils.static.model_type import ModelType
from model_gens.market_predictor_base import MarketPredictorBase
from model_gens import lstm_model_gen, gru_model_gen
from model_gens.utils.static.processing_type import ProcessingType
from typing import Union
import pandas as pd


class ModelTrainer:
    def __init__(
        self, 
        model_class, 
        model_type, 
        data_path, 
        asset_name='XAUUSD', 
        processing_type=ProcessingType.TRAINING
    ):
        self.model_class = model_class
        self.model_type: ModelType = model_type
        self.asset_name = asset_name
        self.target_column = None
        self.processing_type = processing_type
        self.market_predictor: Union[
            MarketPredictorBase, lstm_model_gen.LSTMModel, gru_model_gen.GRUModel
        ] = model_class(base_data_path=data_path)
        
    def train(self, skip_existing=False):
        # Remove duplicates and maintain order if necessary
        columns = [Columns.High, Columns.Low, Columns.Close, Columns.Volume]

        for column in columns:
            self.target_column = column
            self.market_predictor.preprocessor.target_column = column
            self.market_predictor.preprocessor._load_data()
            if self.market_predictor.preprocessor._data.empty:
                print(f"No data found for {column.name}")
                continue
            if self._skip_column_training(skip_existing):
                continue

            print(f"Processing {column.name} - {columns.index(column) + 1}/{len(columns)}")
            self._train_and_save_model()

    def _skip_column_training(self, skip_existing):
        history = self.market_predictor.model_history.get_last_training_date(
            model_table=self.model_type, model_type=self.target_column.name.upper())
        last_data_date, _ = history
        last_data_date = pd.to_datetime(last_data_date) if isinstance(
            last_data_date, str) else last_data_date
        latest_data_date = self.market_predictor.preprocessor._data.index[-1]

        if last_data_date and last_data_date >= latest_data_date:
            print("Model has already been trained on the latest data.")
            return True

        model_path, _, _, _, _ = self._generate_paths()
        if skip_existing and os.path.exists(f'{model_path}.h5'):
            print(f"Skipping {self.target_column.name} as model already exists.")
            return True

        return False

    def _train_and_save_model(self):
        X_train, X_test, y_train, y_test = self.market_predictor.preprocessor.prepare_dataset()
        model = self.market_predictor.train_model(
            X_train, y_train, X_test, y_test)

        model_path, scaler_path, shape_path, scaler_name, shape_name = self._generate_paths(
            self.target_column)
        self.market_predictor.save_models(model, model_path)
        self.market_predictor.preprocessor.save_scaler(
            scaler_filename=scaler_path)

        latest_data_date = self.market_predictor.preprocessor._data.index[-1]
        self.market_predictor.model_history.insert_new_training_data(
            model_table=self.model_type,
            model_name=f'{self.asset_name}_{self.model_type.value.lower()}_{self.target_column.name.lower()}',
            model_type=self.target_column.name,
            data_date=str(latest_data_date),
            model_date=str(datetime.now()),
            model_path=os.path.relpath(model_path, BASE_DIR),
            scaler_path=os.path.relpath(scaler_path, BASE_DIR),
            shape_path=os.path.relpath(shape_path, BASE_DIR),
            scaler_name=scaler_name,
            shape_name=shape_name,
            notes='Training completed'
        )

    def _generate_paths(self):
        model_subdir, scaler_subdir, shape_subdir = self._get_subdirectories()

        self._ensure_directory_exists(model_subdir)
        self._ensure_directory_exists(scaler_subdir)
        self._ensure_directory_exists(shape_subdir)

        model_filename = f'{self.asset_name}_{self.model_type.value.lower()}_{self.target_column.name.lower()}.h5'
        scaler_filename = f'{self.asset_name}_{self.model_type.value.lower()}_{self.target_column.name.lower()}_scaler.pkl'
        shape_filename = f'{self.asset_name}_{self.model_type.value.lower()}_{self.target_column.name.lower()}_shape.pkl'

        model_path = os.path.join(model_subdir, model_filename)
        scaler_path = os.path.join(scaler_subdir, scaler_filename)
        shape_path = os.path.join(shape_subdir, shape_filename)

        return model_path, scaler_path, shape_path, scaler_filename, shape_filename

    def _get_subdirectories(self):
        base_model_path = f'models/{self.asset_name.lower()}/{self.model_type.value.lower()}/{self.target_column.name.lower()}'
        base_scaler_path = f'models/{self.asset_name.lower()}/scalers/{self.model_type.value.lower()}/{self.target_column.name.lower()}'
        base_shape_path = f'models/{self.asset_name.lower()}/shapes/{self.model_type.value.lower()}/{self.target_column.name.lower()}'

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
