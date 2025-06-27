# -*- coding: utf-8 -*-
import math

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import Variable
import numpy as np

from .Utilities import to_variable, to_value
from .Quantizer import Quantizer
import random
import time
from .QuantizerBase import DEFAULT_DATATYPE


class PartialQuantizer(Quantizer):
    """fixed point quantisizer with self adjusting bits_before and bits_after

    it is using a signed fixed point representation and is s based on the
    Quantizer.
    the quantisation range and number of steps is calculated by the
    two bitsizes.

    Parameters:
        name (string):
            the name of the layer in the TensorFlow graph.
        total_bits (int):
            the amount of bits for the fixed point quantisizer
        leak_clip (float):
            leak factor for backpropagation.
            applied when values are clipped to quantization range.
        dtype (tf.dtypes.DType):
            datatype of the layers's operations.
            default is float32.
    """

    def __init__(self, seed, quantized_portion, quantizer:Quantizer, dtype=DEFAULT_DATATYPE):
        super().__init__("portion_" + quantizer.name, dtype=dtype)
        self.quantized_portion: tf.Variable = to_variable(quantized_portion, tf.float32)
        self.seed = seed
        self.quantizer:Quantizer = quantizer


    def reset(self):
        """
        Resets the Quantizer. The next inference triggers the coefficient search again.
        Returns:

        """
        self.quantizer.reset()
        #self.b_frc_trend_setter.assign(0)


    def getCoeffs(self):
        return self.quantizer.getCoeffs()

    def isNonUniform(self):
        return self.quantizer.isNonUniform()

    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = []
        variables.extend([self.quantized_portion])
        variables.extend(self.quantizer.getQuantVariables())
        return variables

    def quant(self, inputs):
        """quantisation function

        applies the quantization to the input.

        Parameters:
            inputs (list of tensors):
                list of all input tensors.

        Returns:
            (tensor):
                the output of layer.
        """
        quantized_portion = self.quantized_portion # Do not ask me why we need this. I do not know either!

        @tf.custom_gradient
        # custom function, that modifies the gradient computation in the
        # backward pass
        def _quant(inputs): # how to remove those parameters without errors?
            inputs = tf.cast(inputs, tf.float32)
            limit = tf.cast(tf.size(inputs), tf.float32) + 1.0
            selection = tf.range(start=1.0, limit=limit) / limit
            selection = tf.random.shuffle(selection, seed=self.seed)
            selection = tf.cast(tf.reshape(selection, tf.shape(inputs)), tf.float32)
            #
            tmp = tf.where(selection <= quantized_portion, self.quantizer(inputs), inputs)
            #tmp = tf.where(selection <= quantized_portion, 0.0, inputs)
            # tmp = inputs
            #tmp = tf.where(selection <= self.quantized_portion, inputs, 0.0)

            # define the gradient calculation
            @tf.function
            def grad(dy):
                is_out_of_range = tf.logical_or(inputs < self.quantizer.min_value, inputs > self.quantizer.max_value)
                is_out_of_range = tf.logical_and(is_out_of_range, selection <= quantized_portion)
                return tf.where(is_out_of_range, self.quantizer.leak_clip * dy, dy)

            return tmp, grad

        return _quant(inputs)
