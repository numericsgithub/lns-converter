from abc import ABC
import numpy as np
from scipy import stats

class BitDistribution:
    """
    This class holds the number of integer and fractional bits for a specific variable name
    """

    def __init__(self, integer_bits, fractional_bits, name=""):
        """
        Args:
            integer_bits: Number of integer bits
            fractional_bits: Number of fractional bits
            name: Name of the specific variable
        """
        self.name = name
        self.integer_bits = integer_bits
        self.fractional_bits = fractional_bits

class LayerBitDistribution:
    """
    This class holds the bit distribution for weights/filters, biases and activations of an entire layer
    """
    def __init__(self, weights_filters: BitDistribution, bias: BitDistribution, activation: BitDistribution, name):
        """

        Args:
            weights_filters: BitDistribution for Weights/Filters
            bias: BitDistribution for Biases
            activation: BitDistribution for Activations
            name:
        """
        self.weights_filters = weights_filters
        self.bias = bias
        self.activation = activation
        self.name = name

class PostTrainingQuantizer(ABC):

    """
    Abstract class for post-training quantizer. The class contains functions to determine statistical parameters
    """

    def __init__(self, debug=False):
        """
        Constructor
        Args:
            debug: Used for output of debug information. If true, then debug information is output, otherwise no output is made.
        """
        self.debug = debug

    def quantize(self, net, word_width, validation_data, favor_integer=True):
        """
        This method quantizes the passed neural network to the word width word_width.

        Args:
            net: Neuronal network which shall be quantized
            word_width: Sets the quantization word width
            validation_data: Validation data of the training dataset which was used for training.
            favor_integer: Specifies whether the integer part is to be favored when quantizing

        Returns:
            Quantized neuronal network
        """
        return net

    def get_variables(self, net, validation_data):
        """
        Returns the Layer variables for weigts/filters, biases and activations

        Args:
            net: Neuronal Network
            validation_data: Validation data of the training dataset

        Returns:
            Weights, Biases and Activations of the Network
        """

        # Getting one sample from Validation Data to calculate the activation
        dataset = list(validation_data)
        sample_data = dataset[0][0]

        # Getting Weights/Filters and Biases
        weights, biases = net.getWeightsAndBiases()

        # Getting Activations
        activations = net.getActivations(sample_data)

        return weights, biases, activations

    @staticmethod
    def get_variance(data_as_array):
        """
        This method calculates the variance of the given input array
        Args:
            data_as_array: layer data as array

        Returns:
            Variance
        """
        return np.var(data_as_array)

    @staticmethod
    def get_std_deviation(data_as_array):
        """
        This method calculates the standard deviation of the given input array
        Args:
            data_as_array: layer data as array

        Returns:
            Standard Deviation
        """
        return np.std(data_as_array)

    @staticmethod
    def get_span(data_as_array):
        """
        This method calculates the span, the min and the max value of the given input array
        Args:
            data_as_array: layer data as array

        Returns:
            Span, Min and Max Value
        """
        span = abs(max(data_as_array) - min(data_as_array))
        return span, min(data_as_array), max(data_as_array)

    @staticmethod
    def get_mean(data_as_array):
        """
        This method calculates the mean value of the given input array
        Args:
            data_as_array: layer data as array

        Returns:
            Mean Value
        """
        return np.mean(data_as_array)

    @staticmethod
    def get_median(data_as_array):
        """
        This method calculates the median value of the given input array. The median is the middle value of the the
        dataset.
        Args:
            data_as_array: layer data as array

        Returns:
            Median Value
        """
        return np.median(data_as_array)

    @staticmethod
    def get_mode(data_as_array):
        """
        This method calculates the mode value of the given input array. The mode is the value that appears
        most often in a set of data values.
        Args:
            data_as_array: layer data as array

        Returns:
            Mode Value
        """
        return stats.mode(data_as_array)[0][0]

    @staticmethod
    def get_percentiles(data_as_array, q1=25, q2=50, q3=75):
        """
        This method calculates the range of values between the percentiles q1 and q3, where q2 is the median (50%).
        Args:
            data_as_array: layer data as array
            q1: percentile q1, default value 25%
            q2: percentile q2, default value 50%
            q3: percentile q3, default vaule 75%

        Returns:
            q1_val: Quantile value of Q1
            median_val: Median value (Q2)
            q3_val: Quantile value of Q3
            min_val: Min value of the dataset
            max_val: Max value of the dataset
            whiskers_min_val: Min value of the whiskers bound or None, if no value exists
            whiskers_max_val: Max value of the whiskers bound or None, if no value exists
        """
        q1_val, median_val, q3_val = np.percentile(np.asarray(data_as_array), [q1, q2, q3])

        # Calculating the range between Q1 and Q3 with lower and upper bounds
        interpercentile_range = q3_val - q1_val
        lower_bound = q1_val - (1.5 * interpercentile_range)
        upper_bound = q3_val + (1.5 * interpercentile_range)

        # Filtering all values which are equal or lower than the lower bound
        whiksers_lower_bound_values = np.compress(data_as_array <= lower_bound, data_as_array)

        # Filtering all values which are equal or higher than the upper bound
        whiskers_upper_bound_values = np.compress(data_as_array >= upper_bound, data_as_array)

        # Getting the max value of the lower bound values, if exists
        whiskers_min_val = max(whiksers_lower_bound_values) if len(whiksers_lower_bound_values) > 0 else None

        # Getting the min value of the upper bound values, if exists
        whiskers_max_val = min(whiskers_upper_bound_values) if len(whiskers_upper_bound_values) > 0 else None

        # Getting the min value of the dataset
        min_val = min(data_as_array)

        # Getting the max value of the dataset
        max_val = max(data_as_array)

        return q1_val, median_val, q3_val, min_val, max_val, whiskers_min_val, whiskers_max_val
