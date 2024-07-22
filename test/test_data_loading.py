import unittest
from model_gens.utils.preprocessor import Preprocessor  # Ensure this path is correct
from model_gens.utils.static.columns import Columns
from model_gens.utils.static.processing_type import ProcessingType
import os


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        # Path to the combined CSV file and preprocessed data directory
        self.combined_data_path = "/home/amar/Documents/Projects/MarketPredictor/Datas/BTCUSDT/combined.csv"
        self.preprocessed_data_dir = "/home/amar/Documents/Projects/MarketPredictor/Datas/BTCUSDT/preprocessed_data"

    def test_load_data_initial(self):
        # Create a Preprocessor instance
        preprocessor = Preprocessor(self.combined_data_path)
        # Set processing type to INITIAL
        preprocessor.processing_type = ProcessingType.INITIAL

        # Call the _load_data method
        preprocessor._load_data()

        # Check if data is loaded correctly
        self.assertIsNotNone(preprocessor._data,
                             "Data should not be None after loading.")
        self.assertFalse(preprocessor._data.empty,
                         "Data should not be empty after loading.")
        required_columns = [Columns.Open.name, Columns.High.name,
                            Columns.Low.name, Columns.Close.name, Columns.Volume.name]
        for column in required_columns:
            self.assertIn(column, preprocessor._data.columns, f"Column {column} should be present in the data.")

    def test_load_data_training_high(self):
        # Create a Preprocessor instance
        preprocessor = Preprocessor(
            path=self.preprocessed_data_dir, 
            processing_type=ProcessingType.TRAINING,
            target_column=Columns.High
        )
        # Set processing type to TRAINING and target column to Close

        # Call the _load_data method
        preprocessor._load_data()

        # Check if data is loaded correctly
        self.assertIsNotNone(preprocessor._data,
                             "Data should not be None after loading.")
        print(preprocessor.path)
        self.assertFalse(preprocessor._data.empty,
                         "Data should not be empty after loading.")
        required_columns = [Columns.Open.name, Columns.High.name, Columns.Low.name,
                            Columns.Close.name, 'expected_next_high', 'true_high_diff']
        for column in required_columns:
            self.assertIn(column, preprocessor._data.columns, f"Column {column} should be present in the data.")

    def test_load_data_training_low(self):
        # Create a Preprocessor instance
        preprocessor = Preprocessor(
            path=self.preprocessed_data_dir, 
            processing_type=ProcessingType.TRAINING,
            target_column=Columns.Low)

        # Call the _load_data method
        preprocessor._load_data()

        # Check if data is loaded correctly
        self.assertIsNotNone(preprocessor._data,
                             "Data should not be None after loading.")
        self.assertFalse(preprocessor._data.empty,
                         "Data should not be empty after loading.")
        required_columns = [Columns.Open.name, Columns.High.name,
                            Columns.Low.name, Columns.Close.name, Columns.Volume.name]
        for column in required_columns:
            self.assertIn(column, preprocessor._data.columns, f"Column {column} should be present in the data.")

    def tearDown(self):
        # Cleanup if necessary
        pass


if __name__ == '__main__':
    unittest.main()
