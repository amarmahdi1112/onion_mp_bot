import sqlite3
from model_gens.utils.static.model_type import ModelType


class ModelTrainingTracker:
    def __init__(self, db_name='model_training_history.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        for table in ModelType:
            self.conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {table.value} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT UNIQUE,
                    model_type TEXT,
                    last_trained_data_date TEXT,
                    last_trained_model_date TEXT,
                    model_path TEXT,
                    scaler_path TEXT,
                    shape_path TEXT,
                    scaler_name TEXT,
                    shape_name TEXT,
                    notes TEXT
                )
            ''')
        self.conn.commit()

    def insert_new_training_data(self, model_table, model_name, model_type, data_date, model_date, model_path, scaler_path, shape_path, scaler_name, shape_name, notes):
        self.conn.execute(f'''
            INSERT OR REPLACE INTO {model_table.value} (model_name, model_type, last_trained_data_date, last_trained_model_date, model_path, scaler_path, shape_path, scaler_name, shape_name, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_name, model_type, data_date, model_date, model_path, scaler_path, shape_path, scaler_name, shape_name, notes))
        self.conn.commit()

    def get_last_training_date(self, model_table, model_type):
        cursor = self.conn.execute(f'''
            SELECT last_trained_data_date, last_trained_model_date
            FROM {model_table.value}
            WHERE model_type = ?
            ORDER BY last_trained_data_date DESC
            LIMIT 1
        ''', (model_type,))
        row = cursor.fetchone()
        return row if row else (None, None)

    def get_model_directory(self, model_table, column_name):
        cursor = self.conn.execute(f'''
            SELECT model_path, scaler_path, shape_path FROM {model_table.value}
            WHERE model_type = ?
            ORDER BY last_trained_model_date DESC LIMIT 1
        ''', (column_name,))
        row = cursor.fetchone()
        return row if row else (None, None, None)

# # Usage example
# tracker = ModelTrainingTracker()

# # Insert new training data
# tracker.insert_new_training_data(
#     ModelType.LSTM, 'LSTM_Model_1', 'Close',
#     '2024-07-01', '2024-07-01',
#     'models/lstm/Close', 'scalers/lstm/Close_scaler.pkl',
#     'shapes/lstm/Close_shape.pkl', 'Close_scaler.pkl', 'Close_shape.pkl',
#     'Initial training'
# )

# # Get last training date
# last_data_date, last_model_date = tracker.get_last_training_date(ModelType.LSTM)
# print(f'Last Data Date: {last_data_date}, Last Model Date: {last_model_date}')
