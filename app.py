from model_gens.lstm_model_gen import LSTMModel
from model_gens.gru_model_gen import GRUModel
from model_gens.utils.static.model_type import ModelType
from model_gens.utils.model_trainer import ModelTrainer
from model_gens.utils.preprocessor import Preprocessor
import tensorflow as tf
from settings import BASE_DIR
import os
import unittest


def configure_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(len(gpus), "Physical GPUs,", len(
                    logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def configure_memory_fraction(memory_limit_mb):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit_mb)]
                )
        except RuntimeError as e:
            print(e)


def run_tests():
    loader = unittest.TestLoader()
    # Replace with the actual path to your test directory
    start_dir = '/home/amar/Documents/Projects/MarketPredictor/test'
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner()
    runner.run(suite)


def main(test=False):
    if test:
        # Run tests
        run_tests()
    else:
        data_path = os.path.join(
            BASE_DIR, 'Datas/BTCUSDT/combined.csv')

        # Enable memory growth for GPU
        configure_memory_growth()

        # Set a fixed amount of GPU memory to use (in MB)
        memory_limit_mb = 3072  # Example: 3 GB
        configure_memory_fraction(memory_limit_mb)

        # Clear terminal screen
        os.system('cls' if os.name == 'nt' else 'clear')
        # Train LSTM model
        # lstm_trainer = ModelTrainer(
        #     model_class=LSTMModel, model_type=ModelType.LSTM, data_path=data_path)
        # lstm_trainer.train()

        # # Train GRU model
        # gru_trainer = ModelTrainer(
        #     model_class=GRUModel, model_type=ModelType.GRU, data_path=data_path)
        # gru_trainer.train()
        
        initial_preprocessor = Preprocessor(data_path)
        initial_preprocessor.analyze_and_predict_high()
        initial_preprocessor.analyze_and_predict_low()
        initial_preprocessor.analyze_and_predict_close()


if __name__ == "__main__":
    main()
