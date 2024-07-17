import unittest
from model_gens.utils.preprocessor import Preprocessor  # Ensure this path is correct
import pandas as pd
import os
from model_gens.utils.static.columns import Columns
from model_gens.utils.static.processing_type import ProcessingType

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        # Path to the CSV file
        self.raw_data_path = "/home/amar/Documents/Projects/MarketPredictor/Datas/BTCUSDT/combined.csv"
        # Initialize the Preprocessor class
        self.preprocessor = Preprocessor(self.raw_data_path)

    def test_load_data(self):
        # Test if data is loaded correctly
        self.preprocessor._load_data()  # Assuming load_data is a method for loading data
        self.assertIsNotNone(self.preprocessor._data, "Data should not be None after loading.")
        self.assertFalse(self.preprocessor._data.empty, "Data should not be empty after loading.")
    
    def test_required_columns_initial(self):
        # Test if the required columns are present for INITIAL processing type
        self.preprocessor.processing_type = ProcessingType.INITIAL
        self.preprocessor._load_data()
        required_columns = [Columns.Open.name, Columns.High.name, Columns.Low.name, Columns.Close.name, Columns.Volume.name]
        for column in required_columns:
            self.assertIn(column, self.preprocessor._data.columns, f"Column {column} should be present in the data.")

    def test_fill_missing_values(self):
        # Test filling missing values
        self.preprocessor._load_data()
        self.preprocessor._fill_missing_values(self.preprocessor._data)
        self.assertFalse(self.preprocessor._data.isna().any().any(), "There should be no NaN values after filling missing values.")

    def test_drop_unnecessary_columns(self):
        # Test dropping unnecessary columns
        self.preprocessor._load_data()
        self.preprocessor._drop_unnecessary_columns(self.preprocessor._data)
        unnecessary_columns = ["Change", 'Open Interest']
        for column in unnecessary_columns:
            self.assertNotIn(column, self.preprocessor._data.columns, f"Column {column} should be dropped from the data.")

    def tearDown(self):
        # Cleanup if necessary
        pass

if __name__ == '__main__':
    unittest.main()
