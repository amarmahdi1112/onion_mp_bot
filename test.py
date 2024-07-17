# %%
from model_gens.lstm_model_gen import LSTMModel
from model_gens.utils.static.columns import Columns
from settings import BASE_DIR
import os
from model_gens.utils.preprocessor import Preprocessor

# %%
# market_predictor_open = LSTMModel(
#     base_data_path=os.path.join(BASE_DIR, 'last_60_steps.csv'),
#     new_data_path=os.path.join(BASE_DIR, 'single_test_data.csv'),
# )

# %%
# market_predictor_open.load_model(column=Columns.Open)

# %%
# predicted_open = market_predictor_open.prepare_latest_data_and_predict_open()

# %%
# predicted_open

# %%
# market_predictor_open.preprocessor._data

# %%
# last_close_val = market_predictor_open.preprocessor._data['Close'][-1]

# %%
# pred_next_diff = predicted_open[0]

# %%
# next_open = last_close_val + pred_next_diff

# %%
# print('Last Close:', last_close_val)
# print('Predicted Next Open:', next_open)

# %%
# market_predictor_low = LSTMModel(
#     base_data_path=os.path.join(BASE_DIR, 'last_60_steps.csv'),
#     new_data_path=os.path.join(BASE_DIR, 'single_test_data.csv'),
# )

# %%
# market_predictor_low.load_model(column=Columns.Low)

# %%
# predicted_low = market_predictor_low.prepare_latest_data_and_predict_low(next_open)

# %%
# predicted_low

# %%
market_predictor_high = LSTMModel(
    base_data_path=os.path.join(BASE_DIR, 'Datas/BTCUSDT/preprocessed_data/high/test/last_60_steps.csv'),
    new_data_path=os.path.join(BASE_DIR, 'single_test_data.csv'),
)

# %%
market_predictor_high.load_model(column=Columns.High)

# %%
predicted_high = market_predictor_high.prepare_latest_data_and_predict_high()

# %% [markdown]
# 

# %%
# market_predictor_high.preprocessor._data['High'][-1] = predicted_high[0]

# %%
# market_predictor_high.preprocessor._data


