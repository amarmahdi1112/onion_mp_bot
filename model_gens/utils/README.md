
# Market Data Preprocessor

## Overview

The `Preprocessor` class is designed to load, preprocess, and scale market data from CSV files. It provides various methods to add technical indicators, manage data, and prepare datasets for machine learning models.

## Properties

- **data**: Property to get and set the market data.
- **High**: Property to get the High prices.
- **Low**: Property to get the Low prices.
- **Close**: Property to get the Close prices.
- **Volume**: Property to get the Volume data.
- **Open**: Property to get the Open prices.
- **change**: Property to get the change data.

## Methods

### `load_data`
Loads data from the specified CSV file.

```python
def load_data(self, data_path):
    """Loads and returns the data from the specified CSV file."""
```
- **data_path**: Path to the CSV file.

### `remove_columns`
Removes specified columns from the data.

```python
def remove_columns(self, data, columns, save=False):
    """Removes specified columns from the data."""
```
- **data**: DataFrame to manipulate.
- **columns**: List of columns to remove.
- **save**: If True, saves the changes to the instance's data property.

### `scale_data`
Scales the data using MinMaxScaler.

```python
def scale_data(self, data=None):
    """Scales the data using MinMaxScaler."""
```
- **data**: DataFrame to scale. If None, uses the instance's data property.

### `add_columns`
Adds new columns to the data DataFrame.

```python
def add_columns(self, data, column_data, save=False):
    """Adds new columns to the data DataFrame."""
```
- **data**: DataFrame to add columns to.
- **column_data**: Dictionary of columns to add.
- **save**: If True, saves the changes to the instance's data property.

### `add_moving_averages`
Adds SMA and EMA columns to the data.

```python
def add_moving_averages(self, data, sma_window, ema_window):
    """Adds SMA and EMA columns to the data."""
```
- **data**: DataFrame to add columns to.
- **sma_window**: Window size for SMA.
- **ema_window**: Window size for EMA.

### `add_rsi`
Adds RSI column to the data.

```python
def add_rsi(self, data, rsi_window):
    """Adds RSI column to the data."""
```
- **data**: DataFrame to add columns to.
- **rsi_window**: Window size for RSI.

### `add_macd`
Adds MACD columns to the data.

```python
def add_macd(self, data, macd_window):
    """Adds MACD columns to the data."""
```
- **data**: DataFrame to add columns to.
- **macd_window**: Window size for MACD.

### `add_bollinger_bands`
Adds Bollinger Bands columns to the data.

```python
def add_bollinger_bands(self, data, bb_window):
    """Adds Bollinger Bands columns to the data."""
```
- **data**: DataFrame to add columns to.
- **bb_window**: Window size for Bollinger Bands.

### `add_atr`
Adds ATR column to the data.

```python
def add_atr(self, data, atr_window):
    """Adds ATR column to the data."""
```
- **data**: DataFrame to add columns to.
- **atr_window**: Window size for ATR.

### `add_stochastic_oscillator`
Adds Stochastic Oscillator columns to the data.

```python
def add_stochastic_oscillator(self, data, stoch_window):
    """Adds Stochastic Oscillator columns to the data."""
```
- **data**: DataFrame to add columns to.
- **stoch_window**: Window size for Stochastic Oscillator.

### `add_donchian_channel`
Adds Donchian Channel columns to the data.

```python
def add_donchian_channel(self, data, donchian_window):
    """Adds Donchian Channel columns to the data."""
```
- **data**: DataFrame to add columns to.
- **donchian_window**: Window size for Donchian Channel.

### `add_additional_indicators`
Adds additional technical indicators to the data.

```python
def add_additional_indicators(self, data, rsi_window, bb_window):
    """Adds additional technical indicators to the data."""
```
- **data**: DataFrame to add columns to.
- **rsi_window**: Window size for RSI.
- **bb_window**: Window size for Bollinger Bands.

### `add_lag_features`
Adds lag features to the data.

```python
def add_lag_features(self, data):
    """Adds lag features to the data."""
```
- **data**: DataFrame to add columns to.

### `add_rolling_window_statistics`
Adds rolling window statistics to the data.

```python
def add_rolling_window_statistics(self, data, bb_window):
    """Adds rolling window statistics to the data."""
```
- **data**: DataFrame to add columns to.
- **bb_window**: Window size for rolling statistics.

### `add_interaction_features`
Adds interaction features to the data.

```python
def add_interaction_features(self, data):
    """Adds interaction features to the data."""
```
- **data**: DataFrame to add columns to.

### `add_differencing`
Adds differencing features to the data.

```python
def add_differencing(self, data):
    """Adds differencing features to the data."""
```
- **data**: DataFrame to add columns to.

### `add_range_features`
Adds range features to the data.

```python
def add_range_features(self, data):
    """Adds range features to the data."""
```
- **data**: DataFrame to add columns to.

### `add_previous_values`
Adds previous values as features to the data.

```python
def add_previous_values(self, data):
    """Adds previous values as features to the data."""
```
- **data**: DataFrame to add columns to.

### `add_volatility`
Adds volatility feature to the data.

```python
def add_volatility(self, data, data_size):
    """Adds volatility feature to the data."""
```
- **data**: DataFrame to add columns to.
- **data_size**: Size of the data.

### `Close_price_data_processor`
Processes the data for Close price prediction.

```python
def Close_price_data_processor(self):
    """Processes the data for Close price prediction."""
```

### `high_low_price_features`
Processes the data for High or Low price prediction.

```python
def high_low_price_features(self, target_column):
    """Processes the data for High or Low price prediction."""
```
- **target_column**: Target column to remove (High or Low).

### `split_train_test`
Splits the data into training and validation sets.

```python
def split_train_test(self, X, y):
    """Splits the data into training and validation sets."""
```
- **X**: The input data.
- **y**: The target data.
- **Returns**: Tuple of training and validation datasets (X_train, X_test, y_train, y_test).

### `prepare_dataset`
Prepares the dataset for training by scaling and creating sequences.

```python
def prepare_dataset(self, target=Columns.Close):
    """Prepares the dataset for training by scaling and creating sequences."""
```
- **target**: The target column to predict (default is `Columns.Close`).
- **Returns**: Split training and validation datasets.

## Usage Example

```python
preprocessor = Preprocessor(path='/path/to/your/data.csv')
X_train, X_test, y_train, y_test = preprocessor.prepare_dataset(target=Columns.Close)
```
