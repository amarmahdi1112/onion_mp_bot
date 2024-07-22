import argparse
import os
import unittest
import tensorflow as tf
from settings import BASE_DIR
from model_gens.lstm_model_gen import LSTMModel
from model_gens.gru_model_gen import GRUModel
from model_gens.utils.static.model_type import ModelType
from model_gens.utils.model_trainer import ModelTrainer
from model_gens.utils.preprocessor import Preprocessor
from model_gens.utils.static.processing_type import ProcessingType

def configure_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

def configure_memory_fraction(memory_limit_mb):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mb)]
                )
        except RuntimeError as e:
            print(e)

def run_tests():
    loader = unittest.TestLoader()
    start_dir = '/home/amar/Documents/Projects/MarketPredictor/test'
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner()
    runner.run(suite)

def main():
    parser = argparse.ArgumentParser(description='Market Predictor')
    parser.add_argument('action', choices=['train', 'process'], help='Action to perform: train or process')
    parser.add_argument('--test', action='store_true', help='Run tests')
    args = parser.parse_args()

    if args.test:
        run_tests()
    else:

        configure_memory_growth()
        memory_limit_mb = 3072
        configure_memory_fraction(memory_limit_mb)

        # os.system('cls' if os.name == 'nt' else 'clear')

        if args.action == 'train':
            data_path = os.path.join(BASE_DIR, 'Datas/BTCUSDT/preprocessed_data')
            # Train LSTM model
            lstm_trainer = ModelTrainer(model_class=LSTMModel, model_type=ModelType.LSTM, data_path=data_path)
            lstm_trainer.train()

            # Train GRU model
            gru_trainer = ModelTrainer(model_class=GRUModel, model_type=ModelType.GRU, data_path=data_path)
            gru_trainer.train()
        elif args.action == 'process':
            data_path = os.path.join(BASE_DIR, 'Datas/BTCUSDT/combined.csv')
            initial_preprocessor = Preprocessor(data_path, processing_type=ProcessingType.INITIAL)
            initial_preprocessor.preprocess_data_for_initial()

if __name__ == "__main__":
    main()
