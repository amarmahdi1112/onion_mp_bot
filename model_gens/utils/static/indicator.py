from enum import Enum

class Indicator(Enum):
    SMA = 'SMA'
    EMA = 'EMA'
    RSI = 'RSI'
    MACD = 'MACD'
    BOLLINGER_BANDS = 'BollingerBands'
    ATR = 'ATR'
    STOCHASTIC_OSCILLATOR = 'StochasticOscillator'
    DONCHIAN_CHANNEL = 'DonchianChannel'
    ADDITIONAL_INDICATORS = 'AdditionalIndicators'
    LAG_FEATURES = 'LagFeatures'
    ROLLING_STATISTICS = 'RollingStatistics'
    INTERACTION_FEATURES = 'InteractionFeatures'
    DIFFERENCING = 'Differencing'
    RANGE_FEATURES = 'RangeFeatures'
    PREVIOUS_VALUES = 'PreviousValues'
    VOLATILITY = 'Volatility'
