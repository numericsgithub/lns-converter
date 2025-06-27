# -*- coding: utf-8 -*-
import math

import tensorflow as tf
from tensorflow import Variable
import numpy as np

from .PerChannelQuantizer import PerChannelQuantizer
from .Utilities import to_variable, to_value
from .Quantizer import Quantizer


class PerKernelQuantizer(Quantizer):
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
    def __init__(self, name, create_quanziter_func, channel_split_axis, channel_splits, kernel_split_axis, kernel_splits, debug=False, dtype=tf.float32):
        super().__init__(name, dtype=dtype)
        self.all_per_channel_quantizers = []
        self.channel_split_axis = channel_split_axis
        self.channel_splits = channel_splits
        self.kernel_split_axis = kernel_split_axis
        self.kernel_splits = kernel_splits
        #self.is_init = False #tf.Variable(initial_value=False,trainable=False)
        self.create_quanziter_func = create_quanziter_func
        self.debug = debug
        for cur_channel in range(self.channel_splits):
            def create_channel_quant(name, cur_kernel):
                return self.create_quanziter_func(name, cur_channel, cur_kernel)
            cur_channel_quant = PerChannelQuantizer(self.name + f"_channel{cur_channel}", create_channel_quant, kernel_split_axis, kernel_splits)
            self.all_per_channel_quantizers.append(cur_channel_quant)
        # self.reset()

    def getQuantVariables(self):
        """get all variables of the layer.

        Returns:
            (list of Varaiables):
                list contains the weight and the bias Variable.
        """
        variables = []
        for quan in self.all_per_channel_quantizers:
            variables.extend(quan.getQuantVariables())
        return variables

    def reset(self):
        """
        Resets the Quantizer. The next inference triggers the coefficient search again.
        Returns:

        """
        for quan in self.all_per_channel_quantizers:
            quan.reset()

    def call(self, inputs, quantisize=None):
        """forward propagation

        if the quantisizer is active it calls the quant function on the input
        and returns the result else it returns the input.

        Parameters:
            inputs (list of tensors):
                list of all input tensors.

        Returns:
            (tensor):
                quantisized imput.
        """
        if self.isQuantisizing():
            split_inputs = tf.split(inputs, self.channel_splits, axis=self.channel_split_axis, name="split_inputs")
            for cur_chan, cur_quan in enumerate(self.all_per_channel_quantizers):
                split_inputs[cur_chan] = cur_quan(split_inputs[cur_chan], quantisize)
            result = tf.concat(split_inputs, axis=self.channel_split_axis, name="concat_result")
            return result
        else:
            return inputs
