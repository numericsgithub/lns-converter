from enum import Enum

class QuantizerLocation(Enum):
    AT_INPUT_ACTIVATIONS = 0,
    AT_WEIGHTS = 1,
    AT_MULTIPLICATION = 2,
    AT_AFTER_SUM = 3
    AT_BIASES = 4,
    AT_OUTPUT_ACTIVATIONS = 5,
    EVERYWHERE = 6