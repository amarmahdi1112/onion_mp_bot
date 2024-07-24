# Description: Enum class for processing type
from enum import Enum

class ProcessingType(Enum):
    INITIAL = 'Initial'
    TRAINING = 'Training'
    PREDICTION = 'Prediction'

